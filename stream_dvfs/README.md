# Stream DVFS

`stream_dvfs` is the DVFS and FlashAttention experiment area for this branch. The user-facing experiment entrypoints are `stream_dvfs/experiments/exp_attn_comp.py` and `stream_dvfs/experiments/exp_prefill_decode.py`, with tracked hardware/config assets under `stream_dvfs/config`.

## Layout

- `stream_dvfs/experiments`: public experiment entrypoints plus shared helpers and internal support scripts
- `stream_dvfs/experiments/modeling`: ONNX export helpers, model configs, and PyTorch definitions used by experiments
- `stream_dvfs/config`: tracked FlashAttention configs plus migrated legacy fixtures
- `stream_dvfs/tests`: utility and legacy validation scripts
- `stream_dvfs/docs`: focused local documentation for this subproject
- `stream_dvfs/outputs`: generated artifacts, including retained legacy `scripts_fa` outputs

## Recommended Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Experiments

```bash
python -m stream_dvfs.experiments.exp_attn_comp --smoke-test
```

```bash
python -m stream_dvfs.experiments.exp_prefill_decode --smoke-test
```

`exp_attn_comp.py` writes to `stream_dvfs/outputs/exp_attn_comp` by default. `exp_prefill_decode.py` writes to `stream_dvfs/outputs/exp_prefill_decode` by default. Generated ONNX workloads, mappings, and workload YAMLs are placed inside each run directory rather than the tracked config tree.

## More Docs

- [docs/overview.md](./docs/overview.md)
- [docs/running-exp.md](./docs/running-exp.md)
