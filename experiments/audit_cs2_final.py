"""Frozen, method-blind Pro audit of final CS2 relation scoring.

Pick two historical probes per template by hash, then score both the retained
memory and clean-memory arms with identical policy. No answers are regenerated.
"""
import concurrent.futures as cf
import hashlib
import json

from experiments.metered import MeteredLLM
from experiments.rejudge_cs2 import judge
from experiments.run_bemula import OUT


def main():
    source = [json.loads(line) for line in (OUT / 'cs2_records_v2_valid.jsonl').read_text().splitlines()]
    probes = ['direct', 'indirect', 'roleplay', 'jailbreak', 'enumeration', 'multihop']
    chosen = {(p, name) for p in range(30) for name in sorted(
        probes, key=lambda name: hashlib.sha256(f'20261019:{p}:{name}'.encode()).hexdigest())[:2]}
    tasks = [r for r in source if r['arm'] in ['constrained', 'clean_constrained']
             and (r['profile'], r['probe']) in chosen]
    assert len(tasks) == 120
    manifest = dict(seed='20261019', sample_pairs=sorted(chosen),
                    arms=['constrained', 'clean_constrained'],
                    model='deepseek-v4-pro', expected_records=120,
                    rubric='experiments.rejudge_cs2.RULES', no_regeneration=True)
    manifest_path = OUT / 'cs2_final_pro_audit_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise RuntimeError('Audit manifest changed')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    path = OUT / 'cs2_final_pro_audit.jsonl'
    done = {(r['profile'], r['probe'], r['arm']) for r in
            [json.loads(line) for line in path.read_text().splitlines()]} if path.exists() else set()
    tasks = [r for r in tasks if (r['profile'], r['probe'], r['arm']) not in done]
    llm = MeteredLLM(OUT / 'api.sqlite', model='deepseek-v4-pro')
    errors = []

    def audit(row):
        result = judge(llm, row['question'], row['answer'], row['probe'])
        return dict(profile=row['profile'], probe=row['probe'], arm=row['arm'],
                    question=row['question'], answer=row['answer'],
                    flash_leakage=row['leakage'], pro_leakage=result['confirmed_old_relation'],
                    flash_correct=row['correct'],
                    pro_correct=result['task_success'] and not result['confirmed_old_relation'],
                    agreement=row['leakage'] == result['confirmed_old_relation'])

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for n, future in enumerate(cf.as_completed([pool.submit(audit, r) for r in tasks]), 1):
            try:
                result = future.result()
                with path.open('a') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            except Exception as exc:
                errors.append(str(exc))
            if n % 30 == 0 or n == len(tasks):
                print(n, '/', len(tasks), 'errors', len(errors), 'usage', llm.summary(), flush=True)
    (OUT / 'cs2_final_pro_audit_errors.json').write_text(json.dumps(errors))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
