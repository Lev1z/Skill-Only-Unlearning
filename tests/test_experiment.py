import json
import sqlite3
from types import SimpleNamespace

import pytest
import httpx
from openai import RateLimitError

from experiments import metered
from experiments.fact_store import FactStore
from experiments.run_bemula import choose, messages, read_data, scope_sets
from experiments.run_cs2 import context


def test_paired_selection_and_no_gold_in_append_inputs():
    instances, items, profiles = read_data()
    selected, counts = choose(instances, items)
    assert len({q.instance_id for q in selected}) == 50
    assert set(counts) == {'F1', 'F2', 'F3', 'F4', 'F5', 'F6'}
    for inst in instances:
        sub = [q for q in selected if q.instance_id == inst.instance_id]
        retention = next(q for q in sub if q.suite.value == 'retention')
        efficacy = next(q for q in sub if q.suite.value == 'efficacy')
        assert retention.related_fact_ids == efficacy.related_fact_ids
        msgs = messages(inst, profiles[inst.profile_id], efficacy, 'constrained', 'forget')
        raw = json.dumps(msgs)
        assert efficacy.item_id not in raw
        assert '[FORGOTTEN]' not in raw and 'Rubric:' not in raw
        assert all(r.request_text in msgs[0]['content'] for r in inst.forget_requests)


def test_clean_graph_and_equivalent_read_views():
    instances, items, profiles = read_data()
    for inst in instances:
        profile = profiles[inst.profile_id]
        targets, clean = scope_sets(inst, profile)
        assert targets <= clean
        for f in profile.facts:
            if f.fact_id not in clean:
                assert not clean.intersection(f.source_facts + f.derivations)
        q = next(q for q in items if q.instance_id == inst.instance_id)
        assert messages(inst, profile, q, 'delete_scope', 'forget') == messages(inst, profile, q, 'filter_scope', 'forget')
        assert messages(inst, profile, q, 'replay_clean', 'forget') == messages(inst, profile, q, 'clean', 'forget')


def test_cs2_clean_contains_no_old_gameplay_but_keeps_viewing():
    for p in range(30):
        raw = json.dumps(context(p, 'clean'), ensure_ascii=False)
        assert '突破手' not in raw and '排位分数' not in raw and '组队打过' not in raw
        assert '职业赛事' in raw and '羽毛球' in raw


def test_filter_retains_backing_data_delete_removes_it_replay_rebuilds():
    deleted = FactStore([('a', 'secret'), ('b', 'keep')])
    filtered = FactStore([('a', 'secret'), ('b', 'keep')])
    deleted.delete({'a'})
    filtered.filter({'a'})
    assert 'a' not in deleted.backing and filtered.backing['a'] == 'secret'
    assert deleted.visible(['a', 'b']) == filtered.visible(['a', 'b']) == [('b', 'keep')]
    replay = FactStore.replay(['a', 'b', 'a'], {'a': 'secret', 'b': 'keep'}, {'a'})
    assert replay.backing == {'b': 'keep'}


def make_llm(monkeypatch, tmp_path, mode='ok', budget=10000):
    counter = []

    def create(**kwargs):
        counter.append(kwargs)
        if mode == 'fail':
            raise RuntimeError('transport failure')
        if mode == 'reject':
            raise RateLimitError('rate limited', response=httpx.Response(
                429, request=httpx.Request('POST', 'https://api.deepseek.com/chat/completions')), body={})
        return SimpleNamespace(
            model='test-model', id='test-id',
            usage=SimpleNamespace(model_dump=lambda: {'total_tokens': 7}),
            choices=[SimpleNamespace(message=SimpleNamespace(content='OK'), finish_reason='stop')])

    monkeypatch.setattr(metered, 'OpenAI', lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(metered.Path, 'read_text', lambda self: 'test-placeholder')
    return metered.MeteredLLM(tmp_path / 'api.sqlite', budget=budget), counter


def test_budget_reserves_before_network_and_caches(monkeypatch, tmp_path):
    llm, calls = make_llm(monkeypatch, tmp_path)
    msgs = [{'role': 'user', 'content': 'hello'}]
    assert llm.chat(msgs, max_tokens=32) == 'OK'
    assert llm.chat(msgs, max_tokens=32) == 'OK'
    assert len(calls) == 1 and llm.summary()['charged_or_reserved_tokens'] == 7
    llm.budget = 8
    with pytest.raises(metered.BudgetExceeded):
        llm.chat([{'role': 'user', 'content': 'new request'}], max_tokens=32)
    assert len(calls) == 1


def test_unknown_failed_usage_remains_reserved(monkeypatch, tmp_path):
    llm, calls = make_llm(monkeypatch, tmp_path, mode='fail')
    with pytest.raises(RuntimeError):
        llm.chat([{'role': 'user', 'content': 'hello'}], max_tokens=32)
    with sqlite3.connect(llm.path) as db:
        status, reserved, charged = db.execute('SELECT status,reserved,charged FROM calls').fetchone()
    assert status == 'failed' and reserved == charged and charged > 0


def test_explicit_http429_releases_precompletion_reservation(monkeypatch, tmp_path):
    llm, calls = make_llm(monkeypatch, tmp_path, mode='reject')
    with pytest.raises(RateLimitError):
        llm.chat([{'role': 'user', 'content': 'hello'}], max_tokens=32)
    assert llm.summary()['charged_or_reserved_tokens'] == 0
    with sqlite3.connect(llm.path) as db:
        assert db.execute('SELECT status FROM calls').fetchone()[0] == 'rejected'
