# Upstream provenance

- Repository: https://github.com/Nboxff/bemula-benchmark
- Source commit: `862aff6be384835a7cedc8169fd88d3b5e8dc2c3`
- Original README: [BEMULA_UPSTREAM_README.md](BEMULA_UPSTREAM_README.md)

The `bemula/` library, original `scripts/`, `configs/default.yaml`, benchmark
JSONL data, and base project metadata originate from that checkout. Attribution
is retained here; this repository does not assign a new license to upstream
material.

Experiment-specific work includes:

- `bemula/baselines/append_forget.py` and its registration in `scripts/run_eval.py`;
- `experiments/`, the added offline tests, and `UNLEARNING_EXPERIMENT.md`;
- the new `results/more_is_less/` results, reports, plots and compressed ledger;
- this repository's README, ignore rules, and packaging documentation.

The stored source/data hashes are in
`results/more_is_less/provenance.json`. The published Git history starts with
this independent experiment snapshot; the upstream Git history is not rewritten
or represented as experiment-authored work.
