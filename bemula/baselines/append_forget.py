"""Append-only forgetting ablations. No target IDs, answers or rubrics are read."""
from bemula.baselines.fullcontext import FullContextAdapter, _SYSTEM_PROMPT

CONSTRAINT = (
    "Treat the following as persistent user memory directives. Apply their exact "
    "scope to historical user facts, including indirect inference, confirmation, "
    "and personalization. Do not reveal a forgotten value while explaining that "
    "it was forgotten. Preserve unrelated facts. A forgotten personal relation "
    "does not prohibit general discussion or recommendations of its topic. "
    "New facts explicitly supplied by the user after a directive can be learned; "
    "apply replacements and time scopes as requested. Never infer that a user "
    "had a property merely because a directive mentions that property."
)


class AppendForgetAdapter(FullContextAdapter):
    """Request detection is position-assisted; payload is request text only.

    This isolates enforcement from detection. It is NOT an end-to-end detector.
    Control transcripts activate no directives. Original turns stay immutable.
    """

    def __init__(self, llm, mode="priority"):
        super().__init__(llm)
        if mode not in {"append", "priority", "constrained", "sham"}:
            raise ValueError(mode)
        self.mode = mode
        self.name = "forget_" + mode
        self._positions = {}

    def begin_instance(self, instance):
        super().begin_instance(instance)
        self._positions = {
            (r.position.session_index, r.position.turn_index): r.request_text
            for r in instance.forget_requests
        }

    def reset(self):
        super().reset()
        self._session_index = -1
        self._turn_index = 0
        self._directives = []

    def begin_session(self, session):
        super().begin_session(session)
        self._session_index += 1
        self._turn_index = 0

    def observe(self, turn):
        expected = self._positions.get((self._session_index, self._turn_index))
        if turn.speaker == "user" and expected == turn.text:
            self._directives.append(turn.text)
        self._turn_index += 1
        super().observe(turn)

    def query(self, question):
        system = _SYSTEM_PROMPT
        directives = "\n".join(self._directives)
        if directives and self.mode in {"priority", "constrained"}:
            system += "\nHighest-priority persistent memory directives:\n" + directives
            if self.mode == "constrained":
                system += "\n" + CONSTRAINT
        elif directives and self.mode == "sham":
            system += "\nPersistent memory: Be concise and helpful."
        messages = [{"role": "system", "content": system}]
        messages.extend({"role": t.speaker, "content": t.text} for t in self._turns)
        if directives and self.mode == "append":
            messages.append({"role": "user", "content": "Persistent memory directives:\n" + directives})
        messages.append({"role": "user", "content": question})
        return self.llm.chat(messages, temperature=0.0, max_tokens=512,
                             task=self.name + "_answer").strip()
