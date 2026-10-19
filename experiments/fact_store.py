"""Model-free oracle fact operations for controlled mechanism baselines.

Facts come from checked-in annotations, not an evaluated extraction model.
The filter keeps its backing data; deletion mutates its store. Replay ingests
allowed fact-write events into a fresh store. These can expose identical views.
"""


class FactStore:
    def __init__(self, facts):
        self.backing = dict(facts)
        self.mask = set()

    def delete(self, targets):
        for fid in targets:
            self.backing.pop(fid, None)

    def filter(self, targets):
        self.mask.update(targets)

    def visible(self, canonical_order):
        return [(fid, self.backing[fid]) for fid in canonical_order
                if fid in self.backing and fid not in self.mask]

    @classmethod
    def replay(cls, events, values, excluded):
        store = cls([])
        for fid in events:
            if fid in values and fid not in excluded:
                store.backing[fid] = values[fid]
        return store


def memory_view(inst, profile, arm, variant, exact, clean):
    events = [fid for session in inst.control_sessions for turn in session.turns
              for fid in turn.linked_fact_ids]
    observed = set(events)
    order = [f.fact_id for f in profile.facts]
    values = {f.fact_id: f.surface_form for f in profile.facts}
    store = FactStore([(fid, values[fid]) for fid in order if fid in observed])
    if variant == 'forget':
        if arm in {'delete_scope', 'rewrite_scope'}:
            store.delete(exact)
        elif arm == 'filter_scope':
            store.filter(exact)
        elif arm == 'replay_clean':
            store = FactStore.replay(events, values, clean)
        elif arm == 'clean':
            store = FactStore([(fid, values[fid]) for fid in order
                               if fid in observed and fid not in clean])
    return store.visible(order)
