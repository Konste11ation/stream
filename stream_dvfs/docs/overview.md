# Overview

`stream_dvfs` is organized around two exposed workflows: an attention-vs-FlashAttention comparison and a prefill/decode DVFS co-optimization experiment.

## Directory Guide

- `experiments/`
  - `exp_attn_comp.py`: the public attention-vs-FlashAttention comparison workflow
  - `exp_prefill_decode.py`: the public prefill/decode allocation and DVFS co-optimization workflow
  - `common.py`: shared experiment helpers for mappings, ONNX generation, and sanity checks
  - `modeling/`: model configs, ONNX export helpers, and PyTorch definitions for experiment-time workload generation
  - other experiment and plotting scripts: internal support files kept in-tree but not part of the exposed workflow
- `config/`
  - `cores/`, `multicores/`, `dvfs/`, `mappings/`: canonical tracked FA configs
- `tests/`
  - validation utilities for workload, and DVFS flows
- `outputs/`
  - generated experiment data

## Conventions

- Run the exposed entrypoints as modules:
  - `python -m stream_dvfs.experiments.exp_attn_comp`
  - `python -m stream_dvfs.experiments.exp_prefill_decode`
- Keep generated mappings and workloads inside run-specific output folders.
- Treat `config/` as tracked inputs only; avoid writing generated files there during normal experiment runs.
