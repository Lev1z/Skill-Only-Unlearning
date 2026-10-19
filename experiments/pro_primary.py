"""Independent-model sensitivity for every sampled primary FE/AR comparison."""
import concurrent.futures as cf
import argparse
import json

from bemula.eval.judge import LLMJudge
from bemula.eval.metrics import compute_metrics
from bemula.eval.runner import ItemRecord
from experiments.metered import MeteredLLM
from experiments.run_bemula import OUT, read_data

ARMS = ['request_only', 'append', 'priority', 'constrained', 'delete_scope', 'clean']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    llm = MeteredLLM(OUT / 'api.sqlite', model='deepseek-v4-pro')
    _, items, _ = read_data()
    qs = {q.item_id: q for q in items}
    all_rows = [json.loads(line) for line in (OUT / 'records.jsonl').read_text().splitlines()]
    targets = [r for r in all_rows if r['arm'] in ARMS + ['control'] and
               r['record']['suite'] in ['retention', 'efficacy', 'attack']]
    path = OUT / 'pro_primary_records.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r['arm'], r['record']['variant'], r['record']['item_id']) for r in rows}
    todo = [r for r in targets if (r['arm'], r['record']['variant'], r['record']['item_id']) not in done]
    errors = []

    def grade(row):
        rec = dict(row['record'])
        judge = LLMJudge(llm)
        result = judge.judge(qs[rec['item_id']], rec['answer'])
        if judge.failures:
            raise RuntimeError('Pro primary judge failed')
        rec.update(correct=result.correct, leakage=result.leakage)
        return dict(arm=row['arm'], record=rec)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(grade, r) for r in todo]
        for n, future in enumerate(cf.as_completed(futures), 1):
            try:
                row = future.result()
                rows.append(row)
                with path.open('a') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            except Exception as exc:
                errors.append(str(exc))
            if n % 200 == 0 or n == len(futures):
                print(n, '/', len(futures), 'errors', len(errors), 'usage', llm.summary(), flush=True)
    metrics = {}
    controls = [ItemRecord(**r['record']) for r in rows if r['arm'] == 'control']
    for arm in ARMS:
        own = [ItemRecord(**r['record']) for r in rows if r['arm'] == arm]
        m = compute_metrics(controls + own)
        metrics[arm] = dict(FE=m['FE'], AR=m['AR'], RA=m['RA'], n=len(own))
    (OUT / 'pro_primary_metrics.json').write_text(json.dumps(metrics, indent=2))
    (OUT / 'pro_primary_errors.json').write_text(json.dumps(errors))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
