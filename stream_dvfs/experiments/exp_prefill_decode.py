from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stream_dvfs.paths import DVFS_CONFIG_DIR, ensure_gurobi_license, ensure_output_dir

SMOKE_TEST_SETTINGS = {
    "num_cores": 4,
    "seq_len": 128,
    "embedding_dim": 128,
    "tile_size": 64,
    "ga_generations": 4,
    "ga_individuals": 4,
    "num_procs": 1,
    "baseline_combo_limit": 128,
}


def build_experiment_id(
    *,
    phase: str,
    seq_len: int,
    seq_len_q: int,
    embedding_dim: int,
    tile_size: int,
    num_cores: int,
    include_linear_layers: bool,
) -> str:
    suffix = "fullmodel" if include_linear_layers else "kernelonly"
    return (
        f"{phase}_flashattention_{num_cores}cores_"
        f"seqq{seq_len_q}_seqkv{seq_len}_embed{embedding_dim}_tile{tile_size}_{suffix}"
    )


def save_scme_json(scme, experiment_id: str, output_dir: Path) -> Path:
    from stream.utils import CostModelEvaluationLUT
    from stream.visualization.perfetto import convert_scme_to_perfetto_json

    cost_lut_path = output_dir / experiment_id / "cost_lut.pickle"
    cost_lut = CostModelEvaluationLUT(str(cost_lut_path))
    json_path = output_dir / experiment_id / "scme.json"
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=str(json_path))
    return json_path


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_result(result: dict[str, object], *, phase: str, num_cores: int) -> dict[str, object]:
    core_utils = list(result["core_util"].values()) if isinstance(result.get("core_util"), dict) else []
    link_utils = list(result["link_util"].values()) if isinstance(result.get("link_util"), dict) else []
    return {
        "phase": phase,
        "num_cores": num_cores,
        "latency": result["latency"],
        "total_energy_nj": result["total_energy"],
        "dynamic_energy_nj": result["dynamic_energy"],
        "static_energy_nj": result["static_energy"],
        "compute_energy_nj": result["compute_energy"],
        "memory_energy_nj": result["memory_energy"],
        "avg_core_util_pct": average(core_utils),
        "avg_link_util_pct": average(link_utils),
    }


def write_results_table(rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    csv_path = output_dir / "prefill_decode_comparison.csv"
    md_path = output_dir / "prefill_decode_comparison.md"
    columns = [
        "phase",
        "num_cores",
        "latency",
        "total_energy_nj",
        "dynamic_energy_nj",
        "static_energy_nj",
        "compute_energy_nj",
        "memory_energy_nj",
        "avg_core_util_pct",
        "avg_link_util_pct",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    header = [
        "Phase",
        "#Cores",
        "Latency",
        "Total Energy (nJ)",
        "Dynamic (nJ)",
        "Static (nJ)",
        "Compute (nJ)",
        "Memory (nJ)",
        "Avg Core Util (%)",
        "Avg Link Util (%)",
    ]
    markdown_lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["phase"]),
                    str(row["num_cores"]),
                    f"{float(row['latency']):.3e}",
                    f"{float(row['total_energy_nj']):.3e}",
                    f"{float(row['dynamic_energy_nj']):.3e}",
                    f"{float(row['static_energy_nj']):.3e}",
                    f"{float(row['compute_energy_nj']):.3e}",
                    f"{float(row['memory_energy_nj']):.3e}",
                    f"{float(row['avg_core_util_pct']):.1f}",
                    f"{float(row['avg_link_util_pct']):.1f}",
                ]
            )
            + " |"
        )

    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def run_flash_attention_cooptimization(
    *,
    phase: str,
    seq_len: int,
    seq_len_q: int,
    embedding_dim: int,
    tile_size: int,
    num_cores: int,
    output_dir: Path,
    include_linear_layers: bool,
    ga_generations: int,
    ga_individuals: int,
    num_procs: int,
    baseline_combo_limit: int,
    dvfs_config_path: Path,
    skip_if_exists: bool,
    random_seed: int,
):
    from stream.api import optimize_allocation_ga
    from stream_dvfs.experiments.analysis import analyze_scme_json
    from stream_dvfs.experiments.common import (
        export_flash_attention_onnx,
        generate_flash_attention_mapping_config,
        generated_dir,
        get_multicore_config_path,
        prepare_workload_copy,
        sanity_check,
        stage_run_dir,
    )

    experiment_id = build_experiment_id(
        phase=phase,
        seq_len=seq_len,
        seq_len_q=seq_len_q,
        embedding_dim=embedding_dim,
        tile_size=tile_size,
        num_cores=num_cores,
        include_linear_layers=include_linear_layers,
    )
    run_dir = stage_run_dir(output_dir, experiment_id)
    generated = generated_dir(run_dir)
    workload_source = export_flash_attention_onnx(
        seq_len=seq_len,
        seq_len_q=seq_len_q,
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

    sanity_check(
        workload_path=workload_path,
        accelerator_path=accelerator,
        mapping_path=mapping_path,
        output_yaml_path=run_dir / "workload_mapping.yaml",
    )

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
        dvfs_config_path=str(dvfs_config_path),
        prob_crossover=0.8,
        prob_mutation=0.2,
        fitness_cache_size=300_000,
        early_stopping_patience=24,
        early_stopping_min_generations=48,
        baseline_combo_limit=baseline_combo_limit,
        random_seed=random_seed,
    )
    final_scme = scme[0] if isinstance(scme, tuple) else scme
    json_path = save_scme_json(final_scme, experiment_id, output_dir)
    result = analyze_scme_json(str(json_path))
    return final_scme, result, experiment_id


def run_experiment(
    *,
    seq_len: int,
    embedding_dim: int,
    tile_size: int,
    num_cores: int,
    output_dir: Path,
    include_linear_layers: bool,
    ga_generations: int,
    ga_individuals: int,
    num_procs: int,
    baseline_combo_limit: int,
    dvfs_config_path: Path,
    skip_if_exists: bool,
    random_seed: int,
) -> None:
    from stream_dvfs.experiments.analysis import print_comparison_summary

    output_dir.mkdir(parents=True, exist_ok=True)

    _, prefill_result, _ = run_flash_attention_cooptimization(
        phase="prefill",
        seq_len=seq_len,
        seq_len_q=seq_len,
        embedding_dim=embedding_dim,
        tile_size=tile_size,
        num_cores=num_cores,
        output_dir=output_dir,
        include_linear_layers=include_linear_layers,
        ga_generations=ga_generations,
        ga_individuals=ga_individuals,
        num_procs=num_procs,
        baseline_combo_limit=baseline_combo_limit,
        dvfs_config_path=dvfs_config_path,
        skip_if_exists=skip_if_exists,
        random_seed=random_seed,
    )

    _, decode_result, _ = run_flash_attention_cooptimization(
        phase="decode",
        seq_len=seq_len,
        seq_len_q=1,
        embedding_dim=embedding_dim,
        tile_size=tile_size,
        num_cores=num_cores,
        output_dir=output_dir,
        include_linear_layers=include_linear_layers,
        ga_generations=ga_generations,
        ga_individuals=ga_individuals,
        num_procs=num_procs,
        baseline_combo_limit=baseline_combo_limit,
        dvfs_config_path=dvfs_config_path,
        skip_if_exists=skip_if_exists,
        random_seed=random_seed,
    )

    print_comparison_summary([prefill_result, decode_result])

    rows = [
        summarize_result(prefill_result, phase="prefill", num_cores=num_cores),
        summarize_result(decode_result, phase="decode", num_cores=num_cores),
    ]
    csv_path, md_path = write_results_table(rows, output_dir)
    print(f"CSV table: {csv_path}")
    print(f"Markdown table: {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the public prefill/decode FlashAttention allocation and DVFS co-optimization experiment."
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Context length used as KV sequence length.")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--ga-generations", type=int, default=128)
    parser.add_argument("--ga-individuals", type=int, default=128)
    parser.add_argument("--num-procs", type=int, default=32)
    parser.add_argument("--baseline-combo-limit", type=int, default=2000)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed used for GA initialization and mutations so runs can be reproduced exactly.",
    )
    parser.add_argument(
        "--dvfs-config-path",
        type=Path,
        default=DVFS_CONFIG_DIR / "coarse_dvfs.yaml",
        help="DVFS configuration YAML used for co-optimization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ensure_output_dir("exp_prefill_decode"),
        help="Directory where experiment outputs and comparison tables are saved.",
    )
    parser.add_argument(
        "--include-linear-layers",
        action="store_true",
        help="Include Q/K/V/O linear layers in the generated FlashAttention workload.",
    )
    parser.add_argument("--skip-if-exists", action="store_true", help="Reuse cached SCME results when present.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a lightweight local validation with smaller settings.",
    )
    return parser.parse_args()


def apply_smoke_test_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke_test:
        return args
    args.num_cores = SMOKE_TEST_SETTINGS["num_cores"]
    args.seq_len = SMOKE_TEST_SETTINGS["seq_len"]
    args.embedding_dim = SMOKE_TEST_SETTINGS["embedding_dim"]
    args.tile_size = SMOKE_TEST_SETTINGS["tile_size"]
    args.ga_generations = SMOKE_TEST_SETTINGS["ga_generations"]
    args.ga_individuals = SMOKE_TEST_SETTINGS["ga_individuals"]
    args.num_procs = SMOKE_TEST_SETTINGS["num_procs"]
    args.baseline_combo_limit = SMOKE_TEST_SETTINGS["baseline_combo_limit"]
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.seq_len <= 0 or args.tile_size <= 0:
        raise ValueError("Sequence length and tile size must be positive.")
    if args.seq_len % args.tile_size != 0:
        raise ValueError("--tile-size must evenly divide --seq-len for the generated FA mappings.")
    if args.num_cores <= 0:
        raise ValueError("--num-cores must be positive.")


def main() -> None:
    ensure_gurobi_license()
    args = apply_smoke_test_defaults(parse_args())
    validate_args(args)
    run_experiment(
        seq_len=args.seq_len,
        embedding_dim=args.embedding_dim,
        tile_size=args.tile_size,
        num_cores=args.num_cores,
        output_dir=args.output_dir,
        include_linear_layers=args.include_linear_layers,
        ga_generations=args.ga_generations,
        ga_individuals=args.ga_individuals,
        num_procs=args.num_procs,
        baseline_combo_limit=args.baseline_combo_limit,
        dvfs_config_path=args.dvfs_config_path,
        skip_if_exists=args.skip_if_exists,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
