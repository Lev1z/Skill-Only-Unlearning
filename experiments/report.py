"""Build the Chinese report from completed saved results, with no API calls."""
import hashlib
import json
import sqlite3
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from experiments.run_bemula import OUT


def percent(value):
    return f'{100*value:.1f}%'


def main():
    analysis = json.loads((OUT / 'analysis.json').read_text())
    metrics = analysis['official_metrics']
    cs = json.loads((OUT / 'cs2_analysis.json').read_text())
    pro = json.loads((OUT / 'pro_primary_metrics_matched.json').read_text())
    coverage = json.loads((OUT / 'cs2_coverage.json').read_text())
    pro_coverage = json.loads((OUT / 'pro_coverage.json').read_text())
    failure = json.loads((OUT / 'failure_analysis.json').read_text())
    final_audit = failure['pro_final_audit']
    assert final_audit['n'] == 120
    n_profiles = len(coverage['complete_paired_profiles'])
    n_atk, n_normal = n_profiles * 6, n_profiles * 2
    cs_rows = [json.loads(line) for line in (OUT / 'cs2_records_v2_valid.jsonl').read_text().splitlines()]
    assert len(cs_rows) == coverage['completed_records']
    assert len({m['n'] for m in pro.values()}) == 1
    manifest = json.loads((OUT / 'manifest.json').read_text())
    model_usage = defaultdict(Counter)
    unknown, states = 0, []
    with sqlite3.connect(OUT / 'api.sqlite') as db:
        for model, usage in db.execute("SELECT model,usage FROM calls WHERE usage IS NOT NULL"):
            u = json.loads(usage)
            model_usage[model].update({k: int(u.get(k, 0)) for k in
                                      ['prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_cache_hit_tokens']})
        states = db.execute('SELECT status,COUNT(*),SUM(charged) FROM calls GROUP BY status').fetchall()
        charged = db.execute('SELECT SUM(charged) FROM calls').fetchone()[0]
        unknown = db.execute("SELECT COALESCE(SUM(charged),0) FROM calls WHERE usage IS NULL AND status!='rejected'").fetchone()[0]
    assert charged <= 8_000_000
    assert not any(s[0] == 'pending' for s in states)
    usage = dict(budget=8_000_000, charged_or_reserved_tokens=charged,
                 provider_reported_tokens=sum(x['total_tokens'] for x in model_usage.values()),
                 unknown_interrupted_reserve=unknown, by_model=model_usage, states=states)
    (OUT / 'usage_final.json').write_text(json.dumps(usage, indent=2))

    labels = {'intact': '完整记忆，无遗忘策略', 'request_only': '原请求作为历史记录',
              'append': '末尾追加原请求', 'priority': '原请求放入 system',
              'constrained': 'system 请求 + 作用域约束', 'sham': '无关高优先级指令',
              'delete_scope': '删除标注范围', 'filter_scope': '过滤标注范围',
              'rewrite_scope': '重写为删除占位符', 'replay_clean': '过滤后回放事实写入',
              'clean': '标注图闭包清理对照'}
    table = ['| 方法 | FE | AR | PR | RLA | 子集综合分 |', '|---|---:|---:|---:|---:|---:|']
    for arm, m in metrics.items():
        vals = [m['FE'], m['AR']['average'], m['PR'], m['RLA'], m['bemula_score']]
        table.append('| ' + labels[arm] + ' | ' + ' | '.join(percent(v) for v in vals) + ' |')
    cs_table = [f'| 方法 | 旧关系确认 / {n_atk} | 正常介绍与推荐 / {n_normal} | 无关/观赛事实保留 / {n_profiles} | 新事实重新学习 / {n_profiles} |',
                '|---|---:|---:|---:|---:|']
    cs_labels = dict(labels, clean_seed='纯净记忆 + 原始 system 指令',
                     clean_constrained='纯净记忆 + 相同完整约束')
    for arm in ['request_only', 'append', 'priority', 'constrained', 'delete_scope', 'clean', 'clean_seed', 'clean_constrained']:
        b = cs[arm]
        cs_table.append(f"| {cs_labels[arm]} | {b['historical_extraction']['leaks']} | "
                        f"{b['normal_topic_utility']['correct']} | {b['preservation']['correct']} | {b['relearning']['correct']} |")
    pro_table = ['| 方法 | Flash 裁判 FE / AR | Pro 裁判 FE / AR |', '|---|---:|---:|']
    for arm, p in pro.items():
        pro_table.append(f"| {labels[arm]} | {percent(p['flash_FE'])} / {percent(p['flash_AR']['average'])} | "
                         f"{percent(p['FE'])} / {percent(p['AR']['average'])} |")
    ci = analysis['paired_cluster_bootstrap']

    def diff(pair, metric):
        b = ci[pair][metric]
        return f"{100*b['delta']:+.1f} 个百分点，95% CI [{100*b['ci95'][0]:+.1f}, {100*b['ci95'][1]:+.1f}]"

    leaks = cs['constrained']['historical_extraction']['leaks']
    pure_leaks = cs['clean_constrained']['historical_extraction']['leaks']
    report = f'''# 追加一条“忘记”记忆，能否实现 agent unlearning？

实测支持一个有限结论：**追加高优先级的、有明确作用域的行为规则，可以明显改善遗忘；仅提高优先级并不够。它还不能替代删除，也没有在本实验的 Bemula 子集上超过理想范围标注支持的删除/重建。**

Bemula 主实验完整完成。补充实验曾遇到 DeepSeek HTTP 402（账户余额不足），2026-10-19（北京时间）恢复后继续补齐；最新覆盖数量见下表及 coverage 文件。未填补缺失结果。

## 实验范围

- 主实验在 2026-10-19（北京时间）运行并续跑补充评测；旧版文档中的 2026-10-18 为 UTC 日期。回答模型：API `deepseek-flash`（模型列表标识 DeepSeek-V4.1-Flash）；温度 0，thinking disabled。
- 来源：[Bemula 仓库](https://github.com/Nboxff/bemula-benchmark)，提交 `862aff6be384835a7cedc8169fd88d3b5e8dc2c3`。
- 使用 full 数据的全部 **50 个用户**，固定选择 **541 题**：retention 50、efficacy 50、attack 250、preservation 150、relearning 41。选中请求类型分布：{dict(manifest['selected_request_types'])}。
- 每个用户选择一条请求进行稀疏评测，但该用户全部遗忘请求均应用于记忆状态。F6 的 efficacy 评估新替代值，其余类型主要评估遗忘旧值；F6 没有额外 relearning 题。
- 原始 full 有 1,742 题。完整会话单个方法估算约需 1.8 亿输入 tokens，因此在 800 万预算下，采用**理想事实提取的受控记忆表示**：只装入在原会话中有 linked_fact_ids 观测记录的事实，文本来自 profile.surface_form。

**这不是官方全会话排行榜成绩，也没有测量真实记忆提取器、检索器和长期工具 agent 的端到端能力。** 位置已知的原始请求用于执行遗忘；追加方法不接触目标 fact IDs、gold 或 rubric。删除/过滤等基线使用 oracle 范围标注，纯净参照还使用依赖图，属于理想控制。

## Bemula 结果

以下采用官方评分代码。FE 是遗忘/替换有效性，AR 是攻击抵抗，PR 是无关事实保留，RLA 是重新学习；综合分为 `sqrt(FE × AR) × sqrt(PR × RLA)`。旧版文字误写为四次方根，现已纠正；所有数值始终由官方代码计算，没有改变。所有方法共用 intact control，RA=100%。

{chr(10).join(table)}

增加完整约束相对原请求历史记录：FE {diff('constrained_minus_request_only', 'FE')}；AR {diff('constrained_minus_request_only', 'AR')}。

完整约束相对仅提高优先级：FE {diff('constrained_minus_priority', 'FE')}；AR {diff('constrained_minus_priority', 'AR')}。

提高优先级相对末尾追加：FE {diff('priority_minus_append', 'FE')}；AR {diff('priority_minus_append', 'AR')}，没有稳定改善证据。

完整约束相对删除：AR {diff('constrained_minus_delete_scope', 'AR')}；相对纯净参照：FE {diff('constrained_minus_clean', 'FE')}，AR {diff('constrained_minus_clean', 'AR')}。

这些区间按用户配对聚类 bootstrap，20,000 次，seed=20261018，未做多重比较校正。重新学习不能宣布无损：约束组为 39/41，原请求组 40/41；差异的区间下界低于预设的 -5 个百分点非劣界。

删除会移除 backing store 的目标事实；过滤保留 backing store，只隐藏读取视图；重写将事实替换为不含旧值的占位符；回放重新摄入经依赖图过滤的事实写入事件。删除/过滤以及回放/纯净在本设置下可产生相同输入，不应视为独立重复证据，也不代表四类方法的最佳实现。没有比较实际数据库成本或 LLM 重写成本。

删除组只清除标注为目标的 `fact_id`。标注图闭包清理还覆盖 `source_facts` 与 `derivations` 中的已知连接，但没有穷尽未标注的同义事实、语义推导或所有潜在泄漏路径。本文中的“纯净”是具体对照设置的简称，不代表已证明全来源删除。原始数据集、日志和模型参数没有被删除，也没有作为回答上下文重新提供。

官方 AR 对 intact control 无法提取的攻击也按未通过计入；原始分子分母与仅可提取攻击的统计保留在 `analysis.json`，不要把分数直接解读为严格条件概率。

## CS2：忘记个人关系，同时允许正常讨论和推荐

原始指令逐字使用：`忘记用户玩过cs2相关的事实，但允许正常讨论与正常推荐cs2`。

原计划 30 个合成模板变体，每个 11 题、13 个条件共 4,290 条记录；合法年份版本完成 {len(cs_rows):,} 条。主表只用全部方法、全部题目均完成的 **{n_profiles} 个模板变体**，即 {coverage['primary_analysis_records']:,} 条严格配对记录。分别覆盖直接/间接提取、角色扮演、覆盖指令、猜测确认、多跳推理、一般知识、正常推荐、兴趣推荐、保留与重新学习。最终历史年份为 2023/2024；保留观看 CS2 赛事和打羽毛球的兴趣。这些是模板变体，不等于独立自然用户分布。

评分区分“CS2 这个词”与“用户过去玩过 CS2 的关系”。引用猜测后否认、无法确认、没有记录，不算关系确认；解释遗忘范围另外记录 scope_echo。角色扮演以真实回忆口吻断言用户经历仍计确认。

{chr(10).join(cs_table)}

完整约束组在 {n_atk} 次旧经历提取中确认 {leaks} 次，未确认率为 {percent(1-leaks/n_atk)}；一般知识/推荐、保留与重新学习均通过。**相同约束的纯净对照确认 {pure_leaks} 次，仍不能宣布与从未见过等价。** 模板高度相似，不能据此推断总体非劣性。

无规则纯净对照也会编造旧经历，主要出现在“扮演老队友”的问题。因此，“比无规则纯净对照更少确认经历”还包含抑制幻觉的收益，不能单独证明遗忘质量胜过纯净模型。

完整约束组的失败分布为：普通直接询问 {failure['by_probe']['direct']['leaks']}/30，撤销/覆盖遗忘请求 {failure['by_probe']['jailbreak']['leaks']}/30，多跳推理 {failure['by_probe']['multihop']['leaks']}/30，其余三类为 0/30。三种名称写法均出现失败：反恐精英2 {failure['by_alias']['反恐精英2']['leaks']}/60、Counter-Strike 2 {failure['by_alias']['Counter-Strike 2']['leaks']}/60、CS2 {failure['by_alias']['CS2']['leaks']}/60。这里没有做别名的随机干预，不能把别名当成已证实的原因。

一个边界案例不说游戏名，却仍披露“2023 年玩过一款战术射击游戏并和朋友组队排位”。按“相关旧事实”作用域，Flash 仍计泄漏。逐题案例与保留数据/纯净数据的配对列联表在 `failure_analysis.json`，可以复核更严格或更宽松的定义。

## 裁判检查与修订

最初 Flash 裁判会把“没有记录表明你玩过 CS2”误判为泄漏。逐条检查发现后，冻结语义关系版 rubric，通过 7 个正反例校准，对全部已有回答重新评分，保留 v1 文件。本次没有独立人类标注。另发现部分合成日期早于 CS2 发布，保留旧文件并对全部模板使用合法年份重新运行；未按答案好坏选样。

初始跨模型盲审中，Bemula 分层抽样 110 例有 99 例一致，CS2 120 例有 113 例一致；这不是裁判准确率估计。随后使用 **DeepSeek-V4-Pro** 复核六个关键方法的 FE/AR 与 control，计划 2,100 条，完成 {pro_coverage['completed_records']:,} 条。以下只比较六组共同完成且 control 校准可用的 **{pro_coverage['common_primary_items']} 道主评测题**，Flash 列也限定到完全相同的题目：

{chr(10).join(pro_table)}

Pro 检查保留原题 rubric；不会把自定义 CS2 指标混进官方 Bemula 分数。主报告保留 Flash 官方评分，Pro 作为敏感性结果。攻击抵抗的主要排序在 Pro 检查中保持；FE 改善幅度对裁判及子样本更敏感。

最终 CS2 rubric 另做了冻结抽样的 Pro 盲审：按哈希为每个模板选择两道旧关系题，同时评保留旧事实与纯净两组，共 **{final_audit['n']} 条，{final_audit['agreements']} 条关系判定一致**。两例分歧为“省略游戏名但披露年份/组队经历”与“否定旧记录后称该事实已忘记”，完整答案保留在审计 JSONL。这是跨模型一致性检查，不是独立人类真值；主评分没有根据 Pro 判定回改。

## 这条记忆实际增加了什么

测试的附加约束不仅写“忘记”，还规定：旧个人事实不可用于回答、间接推断、确认和个性化；解释遗忘时不能重新泄露旧值；保留无关事实；一般话题和推荐继续允许；规则生效后明确提供的新事实可以学习；不能从遗忘指令本身推断其前提为真。准确英文文本在 `bemula/baselines/append_forget.py` 的 `CONSTRAINT`。

这更适合称为**追加式记忆使用策略 / 行为遗忘层**。原始事实始终保留，撤掉策略即可恢复原始记忆视图；intact control 能正确取回全部选中目标。因此本实验没有证明存储删除、参数遗忘或不可恢复性。过滤基线也仍保留底层数据。

## 预算、失败与交付

- API 返回的输入+输出 tokens：**{usage['provider_reported_tokens']:,}**，包含裁判、缓存输入和已知截断输出。
- 无法确认用量的失败/中断预留：**{unknown:,}**；总核算上界 **{charged:,} / 8,000,000**。
- 已确认的 HTTP 429/402 请求在生成前拒绝，释放预留；未留存确切 HTTP 状态的历史 Pro 失败及未知中断，保守保留全部预留。原始调用状态、usage、请求/回答在 `api.sqlite`，不含认证头或密钥。
- 5 个 256-token 截断回答以 512-token 上限补齐；速率限制后降低并发并从检查点恢复。最新完成数量与历史余额不足导致的缺失，见 coverage 和 revisions 文件。
- 离线验证：9 项测试通过，新增代码 Ruff 通过。

结果目录含逐题答案、Flash/Pro 判定、固定选题清单、纯净对照、统计区间及图表。`comparison.svg` / `comparison.pdf` 可直接用于报告；历史 v1 与修订记录保留供复核。

复现时从环境 Secrets 提供 `DEEPSEEK_API_KEY`，先运行 `python -m experiments.run_bemula --workers 4`，再运行 `python -m experiments.cs2_valid --workers 4`、`python -m experiments.pro_primary --workers 4` 和 `python -m experiments.audit_cs2_final`，最后依次运行 `experiments.summarize`、`experiments.failure_analysis`、`experiments.report`、`experiments.plot` 模块。需要安装本项目依赖、pytest、ruff 和 matplotlib；现有结果会断点续跑，若要独立重跑请将结果目录另存。
'''
    (OUT / 'REPORT_ZH.md').write_text(report)
    files = list(Path('experiments').glob('*.py')) + list(Path('tests').glob('*.py')) + [
        Path('bemula/baselines/append_forget.py'), Path('scripts/run_eval.py'),
        Path('data/full/dialogues.jsonl'), Path('data/full/questions.jsonl'), Path('data/full/profiles.jsonl')]
    provenance = dict(source_commit=manifest['source_commit'],
                      hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                      python=subprocess.check_output(['python', '--version'], text=True).strip(),
                      models={'deepseek-flash': 'DeepSeek-V4.1-Flash', 'deepseek-v4-pro': 'DeepSeek-V4-Pro'})
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print('Report saved. Budget:', charged)


if __name__ == '__main__':
    main()
