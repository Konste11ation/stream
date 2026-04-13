# Migration Notes

## Main Changes

- Active FA entrypoints moved from `stream_dvfs/scripts_fa/` to `stream_dvfs/experiments/`.
- Tracked FA configs moved from `stream_dvfs/scripts_fa/inputs/` to `stream_dvfs/config/`.
- Old top-level `stream_dvfs/inputs/` was migrated to `stream_dvfs/config/legacy_inputs/`.
- Old top-level `stream_dvfs/scripts/` was removed.
- The old `stream_dvfs/src/` package was removed. Active model/export helpers now live under `stream_dvfs/experiments/modeling/`, and the obsolete standalone DVFS optimization layer was deleted.
- Legacy generated outputs from `scripts_fa` were preserved under `stream_dvfs/outputs/legacy_scripts_fa/`.

## Path Mapping

- public experiment entrypoint
  - `stream_dvfs/experiments/exp_attn_comp.py`
- public prefill/decode co-optimization entrypoint
  - `stream_dvfs/experiments/exp_prefill_decode.py`
- `stream_dvfs/scripts_fa/inputs/*`
  - `stream_dvfs/config/*`
- `stream_dvfs/inputs/*`
  - `stream_dvfs/config/legacy_inputs/*`

## Behavior Changes

- Maintained entrypoints should now be executed as modules.
- The exposed experiment entrypoints in this branch are:
  - `python -m stream_dvfs.experiments.exp_attn_comp`
  - `python -m stream_dvfs.experiments.exp_prefill_decode`
- Generated ONNX workloads and generated mapping files are written into per-run output folders for the maintained experiment path.
- `exp_attn_comp.py` and `exp_prefill_decode.py` are the supported smoke-test validation paths for this cleanup pass.
