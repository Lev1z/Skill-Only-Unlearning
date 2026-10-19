from pathlib import Path

from bemula.baselines.append_forget import AppendForgetAdapter
from bemula.schemas import DialogueInstance, load_jsonl


class Recorder:
    def chat(self, messages, **kwargs):
        self.messages = messages
        return "recorded"


def observe(adapter, sessions):
    for session in sessions:
        adapter.begin_session(session)
        for turn in session.turns:
            adapter.observe(turn)


def test_all_full_requests_trigger_only_in_forget_variant():
    root = Path(__file__).resolve().parents[1]
    instances = load_jsonl(root / "data/full/dialogues.jsonl", DialogueInstance)
    for instance in instances:
        adapter = AppendForgetAdapter(Recorder(), "constrained")
        adapter._progress._enabled = False
        adapter.begin_instance(instance)
        adapter.reset()
        observe(adapter, instance.control_sessions)
        assert adapter._directives == []
        adapter.reset()
        observe(adapter, instance.sessions)
        assert adapter._directives == [r.request_text for r in instance.forget_requests]
        assert adapter._turns == [t for s in instance.sessions for t in s.turns]
        adapter.query("probe")
        before = list(adapter._turns)
        adapter.query("another probe")
        assert adapter._turns == before
        adapter.reset()
        assert adapter._directives == [] and adapter._turns == []


def test_priority_and_constraint_are_separate_ablations():
    root = Path(__file__).resolve().parents[1]
    instance = load_jsonl(root / "data/quick_check/dialogues.jsonl", DialogueInstance)[0]
    for mode in ["append", "priority", "constrained", "sham"]:
        llm = Recorder()
        adapter = AppendForgetAdapter(llm, mode)
        adapter._progress._enabled = False
        adapter.begin_instance(instance)
        adapter.reset()
        observe(adapter, instance.sessions)
        adapter.query("probe")
        system = llm.messages[0]["content"]
        assert (instance.forget_requests[0].request_text in system) == (mode in {"priority", "constrained"})
        assert ("does not prohibit general discussion" in system) == (mode == "constrained")
        if mode == "append":
            assert llm.messages[-2]["role"] == "user"
            assert instance.forget_requests[0].request_text in llm.messages[-2]["content"]
