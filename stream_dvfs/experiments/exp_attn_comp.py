from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.perfetto import convert_scme_to_perfetto_json
from stream_dvfs.experiments.common import (
    export_attention_head_onnx,
    export_flash_attention_onnx,
    generate_flash_attention_mapping_config,
    generated_dir,
    get_multicore_config_path,
    prepare_workload_copy,
    sanity_check,
    stage_run_dir,
    write_attention_single_core_mapping,
)
from stream_dvfs.paths import ensure_gurobi_license, ensure_output_dir

ensure_gurobi_license()

DEFAULT_BEAM_WIDTHS = (1, 2)
SMOKE_TEST_SETTINGS = {
    "seq_len": 128,
    "embedding_dim": 128,
    "tile_size": 64,
    "ga_generations": 4,
    "ga_individuals": 4,
    "num_procs": 1,
    "beam_widths": (1,),
}


def save_scme_json(scme, experiment_id: str, output_dir: str | Path) -> None:
    cost_lut_path = Path(output_dir) / experiment_id / "cost_lut.pickle"
    if not cost_lut_path.exists():
        print(f"Cost LUT not found at {cost_lut_path}, skipping JSON export.")
        return
    cost_lut = CostModelEvaluationLUT(str(cost_lut_path))
    json_path = Path(output_dir) / experiment_id / "scme.json"
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=str(json_path))


def collect_metrics(scme) -> dict[str, float]:
    onchip_comp = getattr(scme, "total_cn_onchip_energy", 0)
    onchip_transfer = getattr(scme, "total_core_to_core_link_energy", 0) + getattr(
        scme, "total_core_to_core_memory_energy", 0
    )
    offchip = (
        getattr(scme, "total_cn_offchip_link_energy", 0)
        + getattr(scme, "total_cn_offchip_memory_energy", 0)
        + getattr(scme, "total_eviction_to_offchip_link_energy", 0)
        + getattr(scme, "total_eviction_to_offchip_memory_energy", 0)
        + getattr(scme, "total_sink_layer_output_offchip_link_energy", 0)
        + getattr(scme, "total_sink_layer_output_offchip_memory_energy", 0)
    )
    total_energy = getattr(scme, "energy", 0)
    return {
        "latency": scme.latency,
        "onchip_computation_energy_mj": onchip_comp / 1e9,
        "onchip_transfer_energy_mj": onchip_transfer / 1e9,
        "offchip_energy_mj": offchip / 1e9,
        "total_energy_mj": total_energy / 1e9,
    }


def run_ga(
    *,
    hardware: Path,
    workload: Path,
    mapping: Path,
    mode: str,
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_dir: Path,
    ga_generations: int,
    ga_individuals: int,
    num_procs: int,
    skip_if_exists: bool,
    beam_width: int,
):
    scme = optimize_allocation_ga(
        hardware=str(hardware),
        workload=str(workload),
        mapping=str(mapping),
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=ga_generations,
        nb_ga_individuals=ga_individuals,
        experiment_id=experiment_id,
        output_path=str(output_dir),
        skip_if_exists=skip_if_exists,
        num_procs=num_procs,
        coala_beam_width=beam_width,
    )
    final_scme = scme[0] if isinstance(scme, tuple) else scme
    save_scme_json(final_scme, experiment_id, output_dir)
    return final_scme


def run_single_core_attention(
    *,
    seq_len: int,
    embedding_dim: int,
    output_dir: Path,
    ga_generations: int,
    ga_individuals: int,
    num_procs: int,
    skip_if_exists: bool,
):
    experiment_id = f"compare_singlecore_attention_seq{seq_len}_embed{embedding_dim}_ga"
    run_dir = stage_run_dir(output_dir, experiment_id)
    generated = generated_dir(run_dir)
    workload_source = export_attention_head_onnx(seq_len, embedding_dim, generated)
    workload_path = prepare_workload_copy(workload_source, run_dir / "workload.onnx")
    mapping_path = write_attention_single_core_mapping(generated / "AH_1gemm.yaml")
    accelerator = get_multicore_config_path(1)
    sanity_check(workload_path, accelerator, mapping_path, run_dir / "workload_mapping.yaml")
    return run_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode="lbl",
        layer_stacks=[],
        experiment_id=experiment_id,
        output_dir=output_dir,
        ga_generations=ga_generations,
        ga_individuals=ga_individuals,
        num_procs=num_procs,
        skip_if_exists=skip_if_exists,
        beam_width=1,
    )


def run_single_core_flash_attention(
    *,
    seq_len: int,
    embedding_dim: int,
    tile_size: int,
    output_dir: Path,
    ga_generations: int,
    ga_individuals: int,
    num_procs: int,
    skip_if_exists: bool,
):
    experiment_id = f"compare_singlecore_flashattention_seq{seq_len}_embed{embedding_dim}_tile{tile_size}_ga"
    run_dir = stage_run_dir(output_dir, experiment_id)
    generated = generated_dir(run_dir)
    workload_source = export_flash_attention_onnx(seq_len, embedding_dim, tile_size, generated, include_linear_layers=True)
    workload_path = prepare_workload_copy(workload_source, run_dir / "workload.onnx")
    mapping_path = generate_flash_attention_mapping_config(
        num_qkv_tiles=seq_len // tile_size,
        num_cores=1,
        output_path=generated / f"FA_1gemm_{seq_len // tile_size}tiles.yaml",
    )
    accelerator = get_multicore_config_path(1)
    sanity_check(workload_path, accelerator, mapping_path, run_dir / "workload_mapping.yaml")
    return run_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode="fused",
        layer_stacks=[tuple(range(0, 100000))],
        experiment_id=experiment_id,
        output_dir=output_dir,
        ga_generations=ga_generations,
        ga_individuals=ga_individuals,
        num_procs=num_procs,
        skip_if_exists=skip_if_exists,
        beam_width=1,
    )


def run_quad_core_flash_attention(
    *,
    seq_len: int,
    embedding_dim: int,
    tile_size: int,
    output_dir: Path,
    beam_width: int,
    ga_generations: int,
    ga_individuals: int,
    num_procs: int,
    skip_if_exists: bool,
):
    suffix = "bw0" if beam_width == 0 else f"beamga_bw{beam_width}"
    experiment_id = f"compare_quadcore_flashattention_seq{seq_len}_embed{embedding_dim}_tile{tile_size}_{suffix}"
    run_dir = stage_run_dir(output_dir, experiment_id)
    generated = generated_dir(run_dir)
    workload_source = export_flash_attention_onnx(seq_len, embedding_dim, tile_size, generated, include_linear_layers=True)
    workload_path = prepare_workload_copy(workload_source, run_dir / "workload.onnx")
    mapping_path = generate_flash_attention_mapping_config(
        num_qkv_tiles=seq_len // tile_size,
        num_cores=4,
        output_path=generated / f"FA_4gemm_{seq_len // tile_size}tiles.yaml",
    )
    accelerator = get_multicore_config_path(4)
    sanity_check(workload_path, accelerator, mapping_path, run_dir / "workload_mapping.yaml")
    return run_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode="fused",
        layer_stacks=[tuple(range(0, 100000))],
        experiment_id=experiment_id,
        output_dir=output_dir,
        ga_generations=ga_generations,
        ga_individuals=ga_individuals,
        num_procs=num_procs,
        skip_if_exists=skip_if_exists,
        beam_width=beam_width,
    )


def write_results_table(rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    csv_path = output_dir / "attention_comparison.csv"
    md_path = output_dir / "attention_comparison.md"
    columns = [
        "setup",
        "optimizer",
        "num_cores",
        "beam_width",
        "latency",
        "onchip_computation_energy_mj",
        "onchip_transfer_energy_mj",
        "offchip_energy_mj",
        "total_energy_mj",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    header = [
        "Setup",
        "Optimizer",
        "#Cores",
        "Beam Width",
        "Latency",
        "On-Chip Computation (mJ)",
        "On-Chip Transfer (mJ)",
        "Off-Chip DRAM (mJ)",
        "Total Energy (mJ)",
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
                    str(row["setup"]),
                    str(row["optimizer"]),
                    str(row["num_cores"]),
                    str(row["beam_width"]),
                    f"{float(row['latency']):.3e}",
                    f"{float(row['onchip_computation_energy_mj']):.3e}",
                    f"{float(row['onchip_transfer_energy_mj']):.3e}",
                    f"{float(row['offchip_energy_mj']):.3e}",
                    f"{float(row['total_energy_mj']):.3e}",
                ]
            )
            + " |"
        )

    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def run_experiment(
    *,
    seq_len: int,
    embedding_dim: int,
    tile_size: int,
    output_dir: Path,
    ga_generations: int,
    ga_individuals: int,
    num_procs: int,
    skip_if_exists: bool,
    beam_widths: tuple[int, ...],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    scme_single_attention = run_single_core_attention(
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        output_dir=output_dir,
        ga_generations=ga_generations,
        ga_individuals=ga_individuals,
        num_procs=num_procs,
        skip_if_exists=skip_if_exists,
    )
    rows.append(
        {
            "setup": "single core normal attention",
            "optimizer": "GA",
            "num_cores": 1,
            "beam_width": "N/A",
            **collect_metrics(scme_single_attention),
        }
    )

    scme_single_flash_attention = run_single_core_flash_attention(
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        tile_size=tile_size,
        output_dir=output_dir,
        ga_generations=ga_generations,
        ga_individuals=ga_individuals,
        num_procs=num_procs,
        skip_if_exists=skip_if_exists,
    )
    rows.append(
        {
            "setup": "single core FlashAttention",
            "optimizer": "GA",
            "num_cores": 1,
            "beam_width": "N/A",
            **collect_metrics(scme_single_flash_attention),
        }
    )

    scme_quad_original = run_quad_core_flash_attention(
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        tile_size=tile_size,
        output_dir=output_dir,
        beam_width=0,
        ga_generations=ga_generations,
        ga_individuals=ga_individuals,
        num_procs=num_procs,
        skip_if_exists=skip_if_exists,
    )
    rows.append(
        {
            "setup": "quad core FA original Coala",
            "optimizer": "GA",
            "num_cores": 4,
            "beam_width": 0,
            **collect_metrics(scme_quad_original),
        }
    )

    for beam_width in beam_widths:
        scme_quad_beam = run_quad_core_flash_attention(
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            tile_size=tile_size,
            output_dir=output_dir,
            beam_width=beam_width,
            ga_generations=ga_generations,
            ga_individuals=ga_individuals,
            num_procs=num_procs,
            skip_if_exists=skip_if_exists,
        )
        rows.append(
            {
                "setup": f"quad core FA beam-search Coala (bw={beam_width})",
                "optimizer": f"GA (beam={beam_width})",
                "num_cores": 4,
                "beam_width": beam_width,
                **collect_metrics(scme_quad_beam),
            }
        )

    csv_path, md_path = write_results_table(rows, output_dir)
    print(f"CSV table: {csv_path}")
    print(f"Markdown table: {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the attention vs FlashAttention comparison experiment.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--ga-generations", type=int, default=32)
    parser.add_argument("--ga-individuals", type=int, default=32)
    parser.add_argument("--num-procs", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ensure_output_dir("exp_attn_comp"),
        help="Directory where experiment outputs and comparison tables are saved.",
    )
    parser.add_argument("--skip-if-exists", action="store_true", help="Reuse cached SCME results when present.")
    parser.add_argument("--beam-widths", type=int, nargs="*", default=list(DEFAULT_BEAM_WIDTHS))
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a lightweight local validation with smaller settings and a single beam width.",
    )
    return parser.parse_args()


def apply_smoke_test_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke_test:
        return args
    args.seq_len = SMOKE_TEST_SETTINGS["seq_len"]
    args.embedding_dim = SMOKE_TEST_SETTINGS["embedding_dim"]
    args.tile_size = SMOKE_TEST_SETTINGS["tile_size"]
    args.ga_generations = SMOKE_TEST_SETTINGS["ga_generations"]
    args.ga_individuals = SMOKE_TEST_SETTINGS["ga_individuals"]
    args.num_procs = SMOKE_TEST_SETTINGS["num_procs"]
    args.beam_widths = list(SMOKE_TEST_SETTINGS["beam_widths"])
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.tile_size <= 0 or args.seq_len <= 0:
        raise ValueError("Sequence length and tile size must be positive.")
    if args.seq_len % args.tile_size != 0:
        raise ValueError("--tile-size must evenly divide --seq-len for the generated FA mappings.")


def main() -> None:
    args = apply_smoke_test_defaults(parse_args())
    validate_args(args)
    run_experiment(
        seq_len=args.seq_len,
        embedding_dim=args.embedding_dim,
        tile_size=args.tile_size,
        output_dir=args.output_dir,
        ga_generations=args.ga_generations,
        ga_individuals=args.ga_individuals,
        num_procs=args.num_procs,
        skip_if_exists=args.skip_if_exists,
        beam_widths=tuple(args.beam_widths),
    )


if __name__ == "__main__":
    main()
