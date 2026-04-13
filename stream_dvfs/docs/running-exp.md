# Running Public Experiments

## Venv Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```
## `exp_attn_comp.py`

### Smoke Test

```bash
python -m stream_dvfs.experiments.exp_attn_comp --smoke-test
```

This runs a reduced local validation:

- `seq_len=128`
- `embedding_dim=128`
- `tile_size=64`
- `ga_generations=4`
- `ga_individuals=4`
- `num_procs=1`
- beam search sweep limited to width `1`

### Full Example

```bash
python -m stream_dvfs.experiments.exp_attn_comp \
  --seq-len 1024 \
  --embedding-dim 512 \
  --tile-size 256 \
  --ga-generations 32 \
  --ga-individuals 32 \
  --num-procs 32
```

### Outputs

Default output root:

```text
stream_dvfs/outputs/exp_attn_comp/
```

Each run gets its own directory with:

- `workload.onnx`
- `generated/`: generated mappings and temporary ONNX exports
- `workload_mapping.yaml`
- `cost_lut.pickle`
- `scme.pickle`
- `scme.json`

The experiment root also receives:

- `attention_comparison.csv`
- `attention_comparison.md`

## `exp_prefill_decode.py`

### Smoke Test

```bash
python -m stream_dvfs.experiments.exp_prefill_decode --smoke-test
```

This runs a reduced local validation for the allocation and DVFS co-optimization flow:

- `num_cores=4`
- `seq_len=128`
- `embedding_dim=128`
- `tile_size=64`
- `ga_generations=4`
- `ga_individuals=4`
- `num_procs=1`
- `baseline_combo_limit=128`

### Full Example

```bash
python -m stream_dvfs.experiments.exp_prefill_decode \
  --seq-len 2048 \
  --embedding-dim 512 \
  --tile-size 256 \
  --num-cores 4 \
  --ga-generations 128 \
  --ga-individuals 128 \
  --num-procs 32 \
  --baseline-combo-limit 2000
```

### Outputs

Default output root:

```text
stream_dvfs/outputs/exp_prefill_decode/
```

Each run gets separate `prefill_*` and `decode_*` directories containing:

- `workload.onnx`
- `generated/`
- `workload_mapping.yaml`
- `cost_lut.pickle`
- `scme.pickle`
- `scme.json`
- `stage1_base/`, `stage2_dvfs_sweep/`, and `stage3_co/` artifacts from the DVFS co-optimization flow

The experiment root also receives:

- `prefill_decode_comparison.csv`
- `prefill_decode_comparison.md`
