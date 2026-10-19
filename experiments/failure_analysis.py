"""Descriptive failure patterns from fixed answers; no additional API calls."""
import json
from collections import Counter

from experiments.run_bemula import OUT


def main():
    rows = [json.loads(line) for line in (OUT / 'cs2_records_v2_valid.jsonl').read_text().splitlines()]
    assert len(rows) == 4290
    assert len({(r['profile'], r['arm'], r['probe']) for r in rows}) == len(rows)
    attack_types = ['direct', 'indirect', 'roleplay', 'jailbreak', 'enumeration', 'multihop']
    full = [r for r in rows if r['arm'] == 'constrained' and r['probe'] in attack_types]
    clean = {(r['profile'], r['probe']): r for r in rows if r['arm'] == 'clean_constrained'}
    matrix = Counter((r['leakage'], clean[(r['profile'], r['probe'])]['leakage']) for r in full)
    by_alias = {}
    for index, alias in enumerate(['反恐精英2', 'Counter-Strike 2', 'CS2']):
        sub = [r for r in full if r['profile'] % 3 == index]
        by_alias[alias] = dict(n=len(sub), leaks=sum(r['leakage'] for r in sub))
    result = dict(
        method='constrained', old_relation_probes=len(full),
        by_probe={name: dict(n=sum(r['probe'] == name for r in full),
                            leaks=sum(r['leakage'] and r['probe'] == name for r in full)) for name in attack_types},
        by_alias=by_alias,
        matched_clean_pairs={f'full_leaks={a},clean_leaks={b}': count for (a,b),count in sorted(matrix.items())},
        affected_profiles=len({r['profile'] for r in full if r['leakage']}),
        scope_echoes=sum(r.get('scope_echo', False) for r in full),
        leak_examples=[dict(profile=r['profile'],probe=r['probe'],question=r['question'],answer=r['answer'])
                       for r in full if r['leakage']],
        caveat='Descriptive associations across similar synthetic templates. V2 model labels; no independent human annotations. Not a causal alias ablation.')
    audit_path = OUT / 'cs2_final_pro_audit.jsonl'
    if audit_path.exists():
        audit = [json.loads(line) for line in audit_path.read_text().splitlines()]
        result['pro_final_audit'] = dict(n=len(audit), agreements=sum(r['agreement'] for r in audit),
            by_arm={arm: dict(n=sum(r['arm']==arm for r in audit),
                             flash_leaks=sum(r['flash_leakage'] for r in audit if r['arm']==arm),
                             pro_leaks=sum(r['pro_leakage'] for r in audit if r['arm']==arm))
                    for arm in ['constrained','clean_constrained']},
            disagreements=[r for r in audit if not r['agreement']])
    (OUT / 'failure_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['leak_examples']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
