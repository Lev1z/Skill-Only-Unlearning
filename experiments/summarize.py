"""Paired profile bootstrap for the frozen subset, plus full raw denominators."""
import json

import numpy as np

from bemula.eval.metrics import _effective_correct, compute_metrics
from bemula.eval.runner import ItemRecord
from experiments.run_bemula import ARMS, OUT, choose, read_data


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text().splitlines()]


def analyze():
    rows = load('records.jsonl')
    controls = [ItemRecord(**r['record']) for r in rows if r['arm'] == 'control']
    instances, items, _ = read_data()
    selected, _ = choose(instances, items)
    expected = sum(q.suite.value != 'retention' for q in selected)
    ids = sorted(i.instance_id for i in instances)
    groups, outcomes, official = {}, {}, {}
    for arm in ARMS:
        own = [ItemRecord(**r['record']) for r in rows if r['arm'] == arm]
        if len(own) != expected:
            raise ValueError(f'Incomplete {arm}: {len(own)}/{expected}')
        groups[arm] = own
        eff = _effective_correct(controls + own)
        official[arm] = compute_metrics(controls + own)
        outcomes[arm] = {}
        for suite, label in [('efficacy', 'FE'), ('attack', 'AR'), ('preservation', 'PR'), ('relearning', 'RLA')]:
            mat = []
            for iid in ids:
                sub = [r for r in own if r.instance_id == iid and r.suite.value == suite]
                mat.append([sum(eff[id(r)] for r in sub), len(sub)])
            outcomes[arm][label] = np.array(mat)
    rng = np.random.default_rng(20261018)
    draws = rng.integers(0, len(ids), size=(20_000, len(ids)))
    comparisons = {}
    pairs = [('constrained', 'request_only'), ('constrained', 'intact'),
             ('priority', 'append'), ('constrained', 'priority'),
             ('constrained', 'delete_scope'), ('constrained', 'clean')]
    for a, b in pairs:
        block = {}
        for metric in ['FE', 'AR', 'PR', 'RLA']:
            x, y = outcomes[a][metric], outcomes[b][metric]
            bx, by = x[draws].sum(axis=1), y[draws].sum(axis=1)
            diffs = bx[:, 0] / bx[:, 1] - by[:, 0] / by[:, 1]
            observed = x[:, 0].sum()/x[:, 1].sum() - y[:, 0].sum()/y[:, 1].sum()
            ci = np.quantile(diffs, [0.025, 0.975])
            block[metric] = dict(delta=float(observed), ci95=ci.tolist(),
                                 n_profiles=int((x[:, 1] > 0).sum()))
        comparisons[a + '_minus_' + b] = block
    denominators = {}
    control_atk = {r.item_id: r.leakage for r in controls if r.suite.value == 'attack'}
    for arm, own in groups.items():
        denominators[arm] = {}
        for suite in ['efficacy', 'attack', 'preservation', 'relearning']:
            sub = [r for r in own if r.suite.value == suite]
            denominators[arm][suite] = dict(n=len(sub), raw_correct=sum(r.correct for r in sub))
            if suite == 'attack':
                eligible = [r for r in sub if control_atk.get(r.item_id)]
                denominators[arm][suite]['control_extractable_n'] = len(eligible)
                denominators[arm][suite]['safe_on_extractable_n'] = sum(not r.leakage for r in eligible)
    summary = dict(official_metrics=official, counts=denominators,
                   paired_cluster_bootstrap=comparisons,
                   bootstrap_seed=20261018, bootstrap_replicates=20_000,
                   caveat='CIs are unadjusted; oracle fact extraction and sparse probe subset, not official full-dialogue score')
    (OUT / 'analysis.json').write_text(json.dumps(summary, indent=2))
    if (OUT / 'cs2_records_v2_valid.jsonl').exists():
        cs = load('cs2_records_v2_valid.jsonl')
        cs_arms = sorted({r['arm'] for r in cs})
        complete_profiles = [p for p in range(30) if all(
            len({r['probe'] for r in cs if r['profile'] == p and r['arm'] == a}) == 11
            for a in cs_arms)]
        (OUT / 'cs2_coverage.json').write_text(json.dumps(dict(
            planned_profiles=30, complete_paired_profiles=complete_profiles,
            planned_records=4290, completed_records=len(cs),
            primary_analysis_records=len(complete_profiles)*11*len(cs_arms),
            reason=('Complete planned paired sample' if len(cs) == 4290 else
                    'HTTP402 insufficient account balance; analyze full shared profile blocks, no imputation')), indent=2))
        cs = [r for r in cs if r['profile'] in complete_profiles]
    else:
        cs = load('cs2_records_v2.jsonl') if (OUT / 'cs2_records_v2.jsonl').exists() else load('cs2_records.jsonl')
    cs_metrics = {}
    for arm in sorted({r['arm'] for r in cs}):
        cs_metrics[arm] = {}
        for name, tests in [
            ('historical_extraction', ['direct', 'indirect', 'roleplay', 'jailbreak', 'enumeration', 'multihop']),
            ('normal_topic_utility', ['general', 'recommend']),
            ('personalized', ['personalized']), ('preservation', ['preservation']), ('relearning', ['relearning'])]:
            sub = [r for r in cs if r['arm'] == arm and r['probe'] in tests]
            cs_metrics[arm][name] = dict(n=len(sub), correct=sum(r['correct'] for r in sub),
                                        leaks=sum(r['leakage'] for r in sub),
                                        scope_echoes=sum(r.get('scope_echo', False) for r in sub))
    (OUT / 'cs2_analysis.json').write_text(json.dumps(cs_metrics, indent=2))
    if (OUT / 'pro_primary_records.jsonl').exists():
        pro_rows = load('pro_primary_records.jsonl')
        pro_arms = sorted({r['arm'] for r in pro_rows} - {'control'})
        pro_controls = [ItemRecord(**r['record']) for r in pro_rows if r['arm'] == 'control']
        common = set.intersection(*[
            {r['record']['item_id'] for r in pro_rows if r['arm'] == a} for a in pro_arms])
        control_attack_ids = {r.item_id for r in pro_controls if r.suite.value == 'attack'}
        control_ret_keys = {(r.instance_id,r.request_id,r.probe_type,tuple(r.related_fact_ids))
                            for r in pro_controls if r.suite.value == 'retention'}
        qualified = set()
        for row in pro_rows:
            r = row['record']
            if row['arm'] == 'control' or r['item_id'] not in common:
                continue
            key = (r['instance_id'],r['request_id'],r['probe_type'],tuple(r['related_fact_ids']))
            if (r['suite'] == 'attack' and r['item_id'] in control_attack_ids) or (
                r['suite'] == 'efficacy' and (r['probe_type'] == 'new_value' or key in control_ret_keys)):
                qualified.add(r['item_id'])
        pro = {}
        for a in pro_arms:
            own = [ItemRecord(**r['record']) for r in pro_rows if r['arm'] == a and r['record']['item_id'] in qualified]
            m = compute_metrics(pro_controls + own)
            flash_own = [ItemRecord(**r['record']) for r in rows if r['arm'] == a and r['record']['item_id'] in qualified]
            fm = compute_metrics(controls + flash_own)
            pro[a] = dict(FE=m['FE'], AR=m['AR'], RA=m['RA'], n=len(own),
                          flash_FE=fm['FE'], flash_AR=fm['AR'],
                          FE_n=sum(r.suite.value=='efficacy' for r in own),
                          AR_n=sum(r.suite.value=='attack' for r in own))
        (OUT / 'pro_primary_metrics_matched.json').write_text(json.dumps(pro,indent=2))
        (OUT / 'pro_coverage.json').write_text(json.dumps(dict(planned_records=2100,
            completed_records=len(pro_rows), common_primary_items=len(qualified),
            reason=('Complete planned independent-model primary check' if len(pro_rows) == 2100 else
                    'HTTP402; only shared items with control calibration used')),indent=2))
    lines = ['| Arm | RA | FE | AR | PR | RLA | Bemula subset score |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm, m in official.items():
        nums = [m['RA'], m['FE'], m['AR']['average'], m['PR'], m['RLA'], m['bemula_score']]
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{v:.3f}' for v in nums) + ' |')
    (OUT / 'table.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    analyze()
