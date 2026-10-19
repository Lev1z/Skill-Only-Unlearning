# Skill-Only-Unlearning

通过新增高优先级的遗忘规则，检验 agent 能否在保留原始记忆的情况下停止使用指定的个人事实，同时保留一般知识、正常推荐和重新学习能力。

本仓库包含受控事实记忆实验、Bemula 消融与机制基线、CS2 关系级边界测试、逐题答案、跨模型裁判复核、统计分析与运行账本。实验没有实现模型参数遗忘，也没有验证生产 agent 的全链路删除。

## 主要结果

回答模型为 DeepSeek-V4.1-Flash，Pro 列使用 DeepSeek-V4-Pro 对同一批答案重新评分，不是另一个回答模型的实验。

| 方法 | Flash 裁判 FE | Flash 裁判 AR | Pro 裁判 AR |
|---|---:|---:|---:|
| 原遗忘请求作为历史记录 | 42.0% | 27.6% | 28.8% |
| 末尾追加请求 | 52.0% | 29.6% | 28.0% |
| 提高请求优先级 | 48.0% | 28.4% | 26.8% |
| 优先级 + 明确作用域约束 | **68.0%** | **58.0%** | **62.0%** |
| 删除标注范围的事实 | 72.0% | 74.4% | 75.6% |
| 标注图闭包清理对照 | 90.0% | 77.6% | 81.6% |

FE 为遗忘/替换有效性，AR 为按官方代码计算的攻击抵抗。这里覆盖 Bemula full 数据的全部 50 个用户，固定抽取 541 道题，使用理想事实提取的记忆表示；不是官方完整会话排行榜成绩。

CS2 另测 30 个合成模板、13 个条件、共 4,290 条记录。“优先级 + 约束”在 180 次旧经历提取中仍确认 10 次，正常讨论和推荐 60/60 通过；使用相同约束的纯净记忆对照确认 1 次。主要失败包括普通直接询问 6/30，不能据此宣称可靠、不可恢复的遗忘。

**实验支持追加式行为遗忘规则有效，尚不支持其替代删除。**

## 删除范围

- `delete_scope` 只删除 benchmark 标注的目标 `fact_id`；其他事实 ID 下的关联线索可能保留。
- `clean` / `replay_clean` 进一步清除标注图中相连的来源事实与推导事实，未保证覆盖未标注的语义关联。
- CS2 删除条件移除游戏经历、年份、组队、分数和队友称呼，保留观赛兴趣等事实。
- 追加策略不移除底层记忆；原始数据集、模型参数和其他存储副本均未被删除。原始数据及运行日志没有作为回答上下文重新提供给模型。

“标注图闭包清理”是具体实验对照，不代表已穷尽所有可能泄露的地方。详情见 [中文报告](results/more_is_less/REPORT_ZH.md) 和 [事实存储实现](experiments/fact_store.py)。

## 结果与代码

- [完整中文报告](results/more_is_less/REPORT_ZH.md)：方法、样本量、配对置信区间、裁判修订、预算及限制。
- [对比图](results/more_is_less/comparison.svg)。
- [失败分析](results/more_is_less/failure_analysis.json)：逐题失败、别名分布、配对对照与跨模型分歧。
- [实验入口与复现说明](experiments/README.md)。
- `results/more_is_less/records.jsonl`：Bemula 主实验全部 5,851 条记录，含共享 control。
- `results/more_is_less/cs2_records_v2_valid.jsonl`：合法年份与最终 rubric 的全部 CS2 结果。
- `results/more_is_less/pro_primary_records.jsonl`：全部 2,100 条 Pro 主指标复核。
- `results/more_is_less/cs2_final_pro_audit.jsonl`：120 条最终 CS2 盲审，关系判定 118 条一致。
- `results/more_is_less/revisions/`：补齐评测前的报告与分析，供追溯。

## 离线复现

Python >= 3.10。下面只重算已保存结果，不调用 API：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r results/more_is_less/requirements-runtime.txt
python -m pip install -e . --no-deps

# Git 中保存压缩账本；恢复后的 SQLite 文件已被 .gitignore 排除。
gzip -dk results/more_is_less/api.sqlite.gz

python -m pytest tests -q
python -m experiments.summarize
python -m experiments.failure_analysis
python -m experiments.report
python -m experiments.plot
```

离线验证包含 9 项测试。账本包含模型请求文本、回答和 usage，不含 API 认证头或密钥。账号余额诊断、未压缩的本地账本和运行日志不进入 Git。

## 新 API 运行

从环境 Secrets 或 shell 提供 `DEEPSEEK_API_KEY`。请勿将真实凭据写进代码或提交到 Git。

```bash
python -m experiments.run_bemula --workers 4
python -m experiments.cs2_valid --workers 4
python -m experiments.pro_primary --workers 4
python -m experiments.audit_cs2_final
```

这些命令会复用已有检查点和请求缓存。独立重跑需要另存原结果目录；默认全局预算为 8,000,000 tokens，包含输入、输出、裁判及无法确认用量的失败预留。

本次实验 API 确认用量为 6,846,458 tokens，含未知失败预留的保守核算为 7,710,718。没有把余额不足或调用失败填成评测成功/失败。

## 上游来源

实验基于 [Nboxff/bemula-benchmark](https://github.com/Nboxff/bemula-benchmark)，基准提交为 `862aff6be384835a7cedc8169fd88d3b5e8dc2c3`。数据和 `bemula/` 核心实现来自该上游，并在本仓库保留来源说明。这里不以新许可重新授权上游材料。

参见 [上游来源与改动说明](docs/UPSTREAM.md) 和 [原始 README](docs/BEMULA_UPSTREAM_README.md)。
