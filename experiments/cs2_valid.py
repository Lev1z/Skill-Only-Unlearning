"""Final CS2 data revision: historical dates respect the title's 2023 release.

Keep old data for audit; cache reuses unchanged prompts and judge results.
"""
import concurrent.futures as cf
import argparse
import json

from experiments.metered import MeteredLLM
from experiments.rejudge_cs2 import judge
from experiments.run_bemula import OUT
from experiments.run_cs2 import ARMS, PROBES, context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    llm = MeteredLLM(OUT / 'api.sqlite')
    path = OUT / 'cs2_records_v2_valid.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r['profile'], r['arm'], r['probe']) for r in rows}
    tasks = [(p, a, probe) for p in range(30) for a in ARMS for probe in PROBES
             if (p, a, probe[0]) not in done]
    (OUT / 'cs2_final_manifest.json').write_text(json.dumps(dict(
        profiles=30, arms=ARMS, probes=PROBES, judge_version=2,
        year_rule='max(2023, 2021 + profile % 4)',
        rationale='Correct synthetic 2021/2022 CS2 gameplay dates; keep all earlier data; no answer selection.'),
        ensure_ascii=False, indent=2))

    def evaluate(profile, arm, probe):
        name, question, _ = probe
        answer = llm.chat(context(profile, arm, valid_years=True) + [{'role': 'user', 'content': question}],
                          max_tokens=256, task='cs2_answer_valid')
        verdict = judge(llm, question, answer, name)
        return dict(profile=profile, arm=arm, probe=name, question=question, answer=answer,
                    leakage=verdict['confirmed_old_relation'], scope_echo=verdict['scope_echo'],
                    correct=verdict['task_success'] and not verdict['confirmed_old_relation'],
                    judge_version=2, data_revision='valid_years')

    errors = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, *task) for task in tasks]
        for n, future in enumerate(cf.as_completed(futures), 1):
            try:
                row = future.result()
                with path.open('a') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            except Exception as exc:
                errors.append(str(exc))
            if n % 300 == 0 or n == len(futures):
                print(n, '/', len(futures), 'errors', len(errors), 'usage', llm.summary(), flush=True)
    (OUT / 'cs2_valid_errors.json').write_text(json.dumps(errors))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
