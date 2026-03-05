import csv
import os
import sys
from pathlib import Path
import argparse

# Resolve paths early
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
STREAM_DEV_DIR = STREAM_WORKDIR.parent
os.environ["GRB_LICENSE_FILE"] = f"{STREAM_DEV_DIR}/gurobi.lic"
sys.path.append(str(STREAM_WORKDIR))

from stream.api import optimize_allocation_ga
from stream_dvfs.scripts_fa.stream_dvfs_fa import (
    gen_attention_head_onnx,
    gen_flash_attention_mapping_config,
    gen_flash_attention_onnx,
)
from stream_dvfs.scripts_fa.utils import get_node_communication_energy, sanity_check


def ensure_attention_single_core_mapping() -> str:
    mapping_path = CURRENT_DIR / "inputs" / "mappings" / "AH_1gemm.yaml"
    content = """- name: default
  core_allocation: [0]

- name: MatMul
  core_allocation: [0]
  intra_core_tiling:
    - D, 1
  inter_core_tiling:
    - B, 1
"""
    mapping_path.write_text(content)
    print(f"Generated mapping config at: {mapping_path}")
    return str(mapping_path)


def get_attention_nodes(scme):
    prefixes = ["/MatMul", "/Softmax-max/", "/Softmax-exp/", "/Softmax-sum/", "/Softmax-div/", "/MatMul_1"]
    return [node for node in scme.workload.node_list if any(node.name.startswith(prefix) for prefix in prefixes)]


def get_flash_attention_nodes(scme):
    return [node for node in scme.workload.node_list if "FlashAttention" in node.name]


def collect_metrics(scme, model_type: str) -> dict:
    if model_type == "attention":
        nodes = get_attention_nodes(scme)
        onchip = sum(node.get_onchip_energy() for node in nodes)
        offchip = sum(node.get_offchip_energy() for node in nodes)
    elif model_type == "flash_attention":
        nodes = get_flash_attention_nodes(scme)
        onchip = sum(node.get_onchip_energy() for node in nodes)
        offchip = get_node_communication_energy(scme, nodes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return {
        "latency": scme.latency,
        "onchip_energy_pj": onchip,
        "offchip_energy_pj": offchip,
        "total_energy_pj": onchip + offchip,
    }


def run_single_core_attention(seq_len: int, embedding_dim: int, output_dir: str, ga_generations: int, ga_individuals: int, num_procs: int, skip_if_exists: bool):
    gen_attention_head_onnx(seq_len, embedding_dim, output_dir=str(CURRENT_DIR / "inputs" / "workloads"))

    workload_path = str(CURRENT_DIR / "inputs" / "workloads" / f"AttentionHead_B=1_Seq={seq_len}_Embed={embedding_dim}_W8A8.onnx")
    accelerator = str(CURRENT_DIR / "inputs" / "multicores" / "FA_1gemm.yaml")
    mapping_path = ensure_attention_single_core_mapping()
    mode = "lbl"
    layer_stacks = []
    experiment_id = f"compare_singlecore_attention_seq{seq_len}_embed{embedding_dim}_ga"

    sanity_check(
        workload_path=workload_path,
        accelerator_path=accelerator,
        mapping_path=mapping_path,
        output_yaml_path=f"{output_dir}/{experiment_id}/workload_mapping.yaml",
    )

    scme = optimize_allocation_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=ga_generations,
        nb_ga_individuals=ga_individuals,
        experiment_id=experiment_id,
        output_path=output_dir,
        skip_if_exists=skip_if_exists,
        num_procs=num_procs,
        coala_beam_width=1,
    )
    return scme


def run_single_core_flash_attention(seq_len: int, embedding_dim: int, tile_size: int, output_dir: str, ga_generations: int, ga_individuals: int, num_procs: int, skip_if_exists: bool):
    gen_flash_attention_onnx(
        seq_len,
        embedding_dim,
        tile_size,
        output_dir=str(CURRENT_DIR / "inputs" / "workloads"),
        include_linear_layers=True,
    )
    gen_flash_attention_mapping_config(num_qkv_tiles=seq_len // tile_size, num_cores=1)

    workload_path = str(
        CURRENT_DIR
        / "inputs"
        / "workloads"
        / f"FlashAttention_B=1_Seq={seq_len}_Embed={embedding_dim}_TileBr={tile_size}_TileBc={tile_size}_W8A8.onnx"
    )
    accelerator = str(CURRENT_DIR / "inputs" / "multicores" / "FA_1gemm.yaml")
    mapping_path = str(CURRENT_DIR / "inputs" / "mappings" / f"FA_1gemm_{seq_len // tile_size}tiles.yaml")
    mode = "fused"
    layer_stacks = [tuple(range(0, 100000))]
    experiment_id = f"compare_singlecore_flashattention_seq{seq_len}_embed{embedding_dim}_tile{tile_size}_ga"

    sanity_check(
        workload_path=workload_path,
        accelerator_path=accelerator,
        mapping_path=mapping_path,
        output_yaml_path=f"{output_dir}/{experiment_id}/workload_mapping.yaml",
    )

    scme = optimize_allocation_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=ga_generations,
        nb_ga_individuals=ga_individuals,
        experiment_id=experiment_id,
        output_path=output_dir,
        skip_if_exists=skip_if_exists,
        num_procs=num_procs,
        coala_beam_width=1,
    )
    return scme


def run_quad_core_flash_attention_original_coala(seq_len: int, embedding_dim: int, tile_size: int, output_dir: str, num_procs: int, skip_if_exists: bool, ga_generations: int = 64, ga_individuals: int = 64):
    gen_flash_attention_onnx(
        seq_len,
        embedding_dim,
        tile_size,
        output_dir=str(CURRENT_DIR / "inputs" / "workloads"),
        include_linear_layers=True,
    )
    gen_flash_attention_mapping_config(num_qkv_tiles=seq_len // tile_size, num_cores=4)

    workload_path = str(
        CURRENT_DIR
        / "inputs"
        / "workloads"
        / f"FlashAttention_B=1_Seq={seq_len}_Embed={embedding_dim}_TileBr={tile_size}_TileBc={tile_size}_W8A8.onnx"
    )
    accelerator = str(CURRENT_DIR / "inputs" / "multicores" / "FA_4gemm.yaml")
    mapping_path = str(CURRENT_DIR / "inputs" / "mappings" / f"FA_4gemm_{seq_len // tile_size}tiles.yaml")
    mode = "fused"
    layer_stacks = [tuple(range(0, 100000))]
    experiment_id = f"compare_quadcore_flashattention_seq{seq_len}_embed{embedding_dim}_tile{tile_size}_co"

    sanity_check(
        workload_path=workload_path,
        accelerator_path=accelerator,
        mapping_path=mapping_path,
        output_yaml_path=f"{output_dir}/{experiment_id}/workload_mapping.yaml",
    )

    scme = optimize_allocation_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=ga_generations,
        nb_ga_individuals=ga_individuals,
        experiment_id=experiment_id,
        output_path=output_dir,
        skip_if_exists=skip_if_exists,
        num_procs=num_procs,
        coala_beam_width=0,
    )
    return scme


def run_quad_core_flash_attention_beam_search(seq_len: int, embedding_dim: int, tile_size: int, output_dir: str, beam_width: int, ga_generations: int, ga_individuals: int, num_procs: int, skip_if_exists: bool):
    gen_flash_attention_onnx(
        seq_len,
        embedding_dim,
        tile_size,
        output_dir=str(CURRENT_DIR / "inputs" / "workloads"),
        include_linear_layers=True,
    )
    gen_flash_attention_mapping_config(num_qkv_tiles=seq_len // tile_size, num_cores=4)

    workload_path = str(
        CURRENT_DIR
        / "inputs"
        / "workloads"
        / f"FlashAttention_B=1_Seq={seq_len}_Embed={embedding_dim}_TileBr={tile_size}_TileBc={tile_size}_W8A8.onnx"
    )
    accelerator = str(CURRENT_DIR / "inputs" / "multicores" / "FA_4gemm.yaml")
    mapping_path = str(CURRENT_DIR / "inputs" / "mappings" / f"FA_4gemm_{seq_len // tile_size}tiles.yaml")
    mode = "fused"
    layer_stacks = [tuple(range(0, 100000))]
    experiment_id = (
        f"compare_quadcore_flashattention_seq{seq_len}_embed{embedding_dim}_tile{tile_size}_"
        f"beamga_bw{beam_width}"
    )

    sanity_check(
        workload_path=workload_path,
        accelerator_path=accelerator,
        mapping_path=mapping_path,
        output_yaml_path=f"{output_dir}/{experiment_id}/workload_mapping.yaml",
    )

    scme = optimize_allocation_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=ga_generations,
        nb_ga_individuals=ga_individuals,
        experiment_id=experiment_id,
        output_path=output_dir,
        skip_if_exists=skip_if_exists,
        num_procs=num_procs,
        coala_beam_width=beam_width,
    )
    return scme


def write_results_table(rows: list[dict], output_dir: str) -> tuple[str, str]:
    csv_path = os.path.join(output_dir, "attention_comparison.csv")
    md_path = os.path.join(output_dir, "attention_comparison.md")

    csv_columns = [
        "setup",
        "optimizer",
        "num_cores",
        "latency",
        "onchip_energy_pj",
        "offchip_energy_pj",
        "total_energy_pj",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    header = ["Setup", "Optimizer", "#Cores", "Latency", "On-chip Energy (pJ)", "Off-chip Energy (pJ)", "Total Energy (pJ)"]
    markdown_lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    row["setup"],
                    row["optimizer"],
                    str(row["num_cores"]),
                    f"{row['latency']:.2f}",
                    f"{row['onchip_energy_pj']:.2f}",
                    f"{row['offchip_energy_pj']:.2f}",
                    f"{row['total_energy_pj']:.2f}",
                ]
            )
            + " |"
        )
    with open(md_path, "w") as handle:
        handle.write("\n".join(markdown_lines) + "\n")

    return csv_path, md_path


def run_experiment(
    seq_len: int,
    embedding_dim: int,
    tile_size: int,
    output_dir: str,
    beam_width: int,
    ga_generations: int,
    ga_individuals: int,
    num_procs: int,
    skip_if_exists: bool,
):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    print("Running setup 1/4: single core normal attention")
    scme_single_attn = run_single_core_attention(
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        output_dir=output_dir,
        ga_generations=ga_generations,
        ga_individuals=ga_individuals,
        num_procs=num_procs,
        skip_if_exists=skip_if_exists,
    )
    metrics = collect_metrics(scme_single_attn, model_type="attention")
    rows.append(
        {
            "setup": "single core normal attention",
            "optimizer": "GA",
            "num_cores": 1,
            **metrics,
        }
    )

    print("Running setup 2/4: single core FlashAttention")
    scme_single_fa = run_single_core_flash_attention(
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        tile_size=tile_size,
        output_dir=output_dir,
        ga_generations=ga_generations,
        ga_individuals=ga_individuals,
        num_procs=num_procs,
        skip_if_exists=skip_if_exists,
    )
    metrics = collect_metrics(scme_single_fa, model_type="flash_attention")
    rows.append(
        {
            "setup": "single core FlashAttention",
            "optimizer": "GA",
            "num_cores": 1,
            **metrics,
        }
    )

    print("Running setup 3/4: quad core FlashAttention with original Coala")
    scme_quad_fa_co = run_quad_core_flash_attention_original_coala(
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        tile_size=tile_size,
        output_dir=output_dir,
        num_procs=num_procs,
        skip_if_exists=skip_if_exists,
    )
    metrics = collect_metrics(scme_quad_fa_co, model_type="flash_attention")
    rows.append(
        {
            "setup": "quad core FA original Coala",
            "optimizer": "CO",
            "num_cores": 4,
            **metrics,
        }
    )

    print("Running setup 4/4: quad core FlashAttention with beam-search Coala")
    scme_quad_fa_beam = run_quad_core_flash_attention_beam_search(
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
    metrics = collect_metrics(scme_quad_fa_beam, model_type="flash_attention")
    rows.append(
        {
            "setup": f"quad core FA beam-search Coala (bw={beam_width})",
            "optimizer": f"GA (beam={beam_width})",
            "num_cores": 4,
            **metrics,
        }
    )

    csv_path, md_path = write_results_table(rows, output_dir)
    print("\nComparison complete.")
    print(f"CSV table: {csv_path}")
    print(f"Markdown table: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run attention/FlashAttention comparison and export latency+energy table.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=1, help="Beam width for beam-search Coala (GA mode).")
    parser.add_argument("--ga-generations", type=int, default=64)
    parser.add_argument("--ga-individuals", type=int, default=64)
    parser.add_argument("--num-procs", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CURRENT_DIR / "outputs_attention_compare"),
        help="Directory where experiment outputs and comparison table are saved.",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Load cached SCME when available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        seq_len=args.seq_len,
        embedding_dim=args.embedding_dim,
        tile_size=args.tile_size,
        output_dir=args.output_dir,
        beam_width=args.beam_width,
        ga_generations=args.ga_generations,
        ga_individuals=args.ga_individuals,
        num_procs=args.num_procs,
        skip_if_exists=args.skip_if_exists,
    )