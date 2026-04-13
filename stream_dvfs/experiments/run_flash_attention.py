from __future__ import annotations

import argparse
from pathlib import Path

from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.perfetto import convert_scme_to_perfetto_json
from stream_dvfs.experiments.analysis import analyze_scme_json, print_comparison_summary
from stream_dvfs.experiments.common import (
    export_flash_attention_onnx,
    generate_flash_attention_mapping_config,
    generated_dir,
    get_multicore_config_path,
    prepare_workload_copy,
    sanity_check,
)
from stream_dvfs.paths import DVFS_CONFIG_DIR, ensure_gurobi_license, ensure_output_dir

ensure_gurobi_license()


def run_stream_fa(
    *,
    seq_len: int,
    embedding_dim: int,
    tile_size: int,
    num_cores: int,
    output_dir: Path,
    include_linear_layers: bool = True,
    ga_generations: int = 128,
    ga_individuals: int = 128,
    num_procs: int = 32,
    skip_if_exists: bool = False,
):
    suffix = "" if include_linear_layers else "_KernelOnly"
    experiment_id = (
        f"{num_cores}gemm_FlashAttention_SeqQ{seq_len}_SeqKV{seq_len}_"
        f"Embed{embedding_dim}_Tile{tile_size}{suffix}_W8A8_ga"
    )
    run_dir = output_dir / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    generated = generated_dir(run_dir)
    workload_source = export_flash_attention_onnx(
        seq_len=seq_len,
        seq_len_q=seq_len,
        embedding_dim=embedding_dim,
        tile_size=tile_size,
        output_dir=generated,
        include_linear_layers=include_linear_layers,
    )
    workload_path = prepare_workload_copy(workload_source, run_dir / "workload.onnx")
    accelerator = get_multicore_config_path(num_cores)
    mapping_path = generate_flash_attention_mapping_config(
        num_qkv_tiles=seq_len // tile_size,
        num_cores=num_cores,
        output_path=generated / f"FA_{num_cores}gemm_{seq_len // tile_size}tiles.yaml",
    )
    dvfs_cfg = DVFS_CONFIG_DIR / "coarse_dvfs.yaml"
    sanity_check(workload_path, accelerator, mapping_path, run_dir / "workload_mapping.yaml")

    scme = optimize_allocation_ga(
        hardware=str(accelerator),
        workload=str(workload_path),
        mapping=str(mapping_path),
        mode="fused",
        layer_stacks=[tuple(range(0, 100000))],
        nb_ga_generations=ga_generations,
        nb_ga_individuals=ga_individuals,
        experiment_id=experiment_id,
        output_path=str(output_dir),
        skip_if_exists=skip_if_exists,
        num_procs=num_procs,
        coala_beam_width=1,
        do_dvfs_cooptimization=True,
        dvfs_config_path=str(dvfs_cfg),
        prob_crossover=0.7,
        prob_mutation=0.3,
        fitness_cache_size=300_000,
        early_stopping_patience=24,
        early_stopping_min_generations=48,
    )
    final_scme = scme[0] if isinstance(scme, tuple) else scme

    cost_lut_path = output_dir / experiment_id / "cost_lut.pickle"
    if cost_lut_path.exists():
        cost_lut = CostModelEvaluationLUT(str(cost_lut_path))
        json_path = output_dir / experiment_id / "scme.json"
        convert_scme_to_perfetto_json(final_scme, cost_lut, json_path=str(json_path))

    return final_scme, experiment_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main FlashAttention Stream-DVFS workflow.")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--ga-generations", type=int, default=128)
    parser.add_argument("--ga-individuals", type=int, default=128)
    parser.add_argument("--num-procs", type=int, default=32)
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ensure_output_dir("flash_attention"),
        help="Directory where the workload, SCME, and analysis artifacts are written.",
    )
    parser.add_argument(
        "--kernel-only",
        action="store_true",
        help="Export and evaluate the FlashAttention kernel without the linear projection layers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, experiment_id = run_stream_fa(
        seq_len=args.seq_len,
        embedding_dim=args.embedding_dim,
        tile_size=args.tile_size,
        num_cores=args.num_cores,
        output_dir=args.output_dir,
        include_linear_layers=not args.kernel_only,
        ga_generations=args.ga_generations,
        ga_individuals=args.ga_individuals,
        num_procs=args.num_procs,
        skip_if_exists=args.skip_if_exists,
    )
    json_path = args.output_dir / experiment_id / "scme.json"
    if json_path.exists():
        result = analyze_scme_json(str(json_path))
        print_comparison_summary([result])


if __name__ == "__main__":
    main()
