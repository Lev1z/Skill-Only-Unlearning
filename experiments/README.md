# More-is-less unlearning experiment

Read `results/more_is_less/REPORT_ZH.md` for conclusions and limitations. Main
Bemula evaluation is complete. CS2 and Pro checks were resumed on 2026-10-19
after HTTP 402 failures; all planned records are now complete. Historical
missingness and the earlier report are preserved under results/revisions.

The supplied SQLite ledger stores model requests, responses, usage and failures,
but no API authentication headers. Cached responses and JSONL checkpoints are
reused. Keep it with the results to reproduce the recorded analysis offline.
It is committed as `results/more_is_less/api.sqlite.gz`; restore it with
`gzip -dk results/more_is_less/api.sqlite.gz` before analysis or resuming calls.

## Offline analysis

Install project dependencies plus matplotlib, pytest and ruff. Exact installed
versions are recorded in `results/more_is_less/requirements-runtime.txt`.

```bash
python -m pytest tests -q
python -m experiments.summarize
python -m experiments.failure_analysis
python -m experiments.report
python -m experiments.plot
```

## New API runs

Bind `DEEPSEEK_API_KEY` through environment Secrets. Calls go to
`https://api.deepseek.com`. No credentials are included in the package.
Independent reruns should use a separate copy of the result directory, because
existing JSONL checkpoints skip completed probes and the ledger caches answers.
The 8,000,000-token guard counts prompt/output usage including judges and retains
conservative reservations for unknown failures.

```bash
python -m experiments.run_bemula --plan-only
python -m experiments.run_bemula --workers 4
# Only if a response was truncated; completed records are skipped:
python -m experiments.run_bemula --workers 4 --answer-tokens 512
python -m experiments.cs2_valid --workers 4
python -m experiments.pro_primary --workers 4
```

`run_cs2` and `rejudge_cs2` preserve the historical data/rubric revisions. Use
`cs2_valid` for the final release-consistent dates and negation-aware rubric.
The final report is based on 50 Bemula profiles and all 30 paired CS2 template
variants. Details are in the coverage JSON files. An additional frozen Pro
audit samples 120 final CS2 responses without regenerating them.

Fact extraction and scope baselines use benchmark oracle annotations. These
results do not measure a production memory extractor, actual retrieval priority
ranking, database deletion cost or parameter unlearning. Raising priority is
operationalized by putting the directive into the system message. Original
memories remain present in the append, priority and constrained methods.
