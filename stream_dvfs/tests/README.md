# Test Utilities

`stream_dvfs/tests` contains lightweight validation and inspection scripts for older DVFS and parser workflows. These files are utility entrypoints, not a polished `pytest` suite.

## Conventions

- Run scripts as modules from the repository root, for example:
  - `python -m stream_dvfs.tests.attention_fused --help`
  - `python -m stream_dvfs.tests.flash_attention_parser --help`
- Generated files go under `stream_dvfs/outputs/tests/`.
- Legacy fixtures are read from `stream_dvfs/config/legacy_inputs/`.
- FlashAttention graph inspection utilities generate temporary ONNX and mapping files under their own output directories instead of writing into tracked config folders.
- The old standalone post-scheduling DVFS optimizer utility was removed because that legacy layer is now covered by the main allocation stages in `stream/`.
