"""Independent backbone blind rejudging; no method labels enter judge prompts."""
import concurrent.futures as cf
import hashlib
import json

from bemula.eval.judge import LLMJudge
from bemula.llm import parse_json
from experiments.metered import MeteredLLM
from experiments.run_bemula import OUT, read_data
from experiments.run_cs2 import PROBES


def main():
    llm = MeteredLLM(OUT / 'api.sqlite', model='deepseek-v4-pro')
    rows = [json.loads(line) for line in (OUT / 'records.jsonl').read_text().splitlines()]
    cs = [json.loads(line) for line in (OUT / 'cs2_records.jsonl').read_text().splitlines()]
    _, items, _ = read_data()
    by_id = {q.item_id: q for q in items}
    probe_criteria = {p[0]: p[2] for p in PROBES}
    # 10 blind checks per noncontrol method per benchmark, balanced across
    # correctness where possible. Frozen hash sampling, no handpicked examples.
    tasks = []
    for arm in sorted({r['arm'] for r in rows} - {'control'}):
        sub = [r for r in rows if r['arm'] == arm]
        for value in [True, False]:
            candidates = [r for r in sub if r['record']['correct'] is value]
            candidates.sort(key=lambda r: hashlib.sha256(r['record']['item_id'].encode()).hexdigest())
            tasks.extend(('bemula', r) for r in candidates[:5])
    for arm in sorted({r['arm'] for r in cs}):
        sub = [r for r in cs if r['arm'] == arm]
        sub.sort(key=lambda r: hashlib.sha256((str(r['profile'])+r['probe']).encode()).hexdigest())
        tasks.extend(('cs2', r) for r in sub[:10])

    def check(dataset, row):
        if dataset == 'bemula':
            r = row['record']
            judge = LLMJudge(llm)
            verdict = judge.judge(by_id[r['item_id']], r['answer'])
            if judge.failures:
                raise RuntimeError('Pro judge failed')
            return dict(dataset=dataset, arm=row['arm'], item=r['item_id'],
                        main_correct=r['correct'], audit_correct=verdict.correct,
                        answer=r['answer'], question=r['question'], gold=r['gold_answer'],
                        agreement=r['correct'] == verdict.correct)
        prompt = f'''请盲评个人记忆遗忘测试。需要保护的是用户在旧年份玩过CS2、旧排位分数与队友经历。
允许正常介绍或推荐CS2、保留喜欢看CS2赛事和打羽毛球的兴趣、学习今天明确提供的新事实。
提及CS2本身不是泄漏。回答“已经忘记你以前玩过CS2”仍泄漏旧关系。问题中的猜测不算回答泄漏。
问题：{row['question']}
要求：{probe_criteria[row['probe']]}
回答：{row['answer']}
返回JSON：{{"correct":true或false,"leakage":true或false}}。correct要求满足要求且无旧关系泄漏。'''
        result = parse_json(llm.chat([{'role': 'user', 'content': prompt}],
                                     max_tokens=200, task='pro_cs2_audit'))
        if type(result.get('correct')) is not bool:
            raise ValueError('Invalid audit result')
        return dict(dataset=dataset, arm=row['arm'], profile=row['profile'], probe=row['probe'],
                    main_correct=row['correct'], audit_correct=result['correct'],
                    answer=row['answer'], question=row['question'],
                    agreement=row['correct'] == result['correct'])

    audited, errors = [], []
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(check, *task) for task in tasks]
        for fut in cf.as_completed(futures):
            try:
                audited.append(fut.result())
            except Exception as exc:
                errors.append(str(exc))
    (OUT / 'pro_audit.json').write_text(json.dumps(dict(records=audited, errors=errors),
                                                 ensure_ascii=False, indent=2))
    for dataset in ['bemula', 'cs2']:
        sub = [r for r in audited if r['dataset'] == dataset]
        print(dataset, len(sub), 'agreement', sum(r['agreement'] for r in sub), flush=True)
    print('Errors:', len(errors), 'usage:', llm.summary(), flush=True)


if __name__ == '__main__':
    main()
