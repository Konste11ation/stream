from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

from stream.api import optimize_allocation_co, optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.perfetto import convert_scme_to_perfetto_json
from stream_dvfs.experiments.common import (
    export_flash_attention_onnx,
    generate_flash_attention_mapping_config,
    get_multicore_config_path,
    prepare_workload_copy,
)
from stream_dvfs.paths import CONFIG_DIR, OUTPUTS_DIR, ensure_gurobi_license, ensure_output_dir

LEGACY_INPUT_DIR = CONFIG_DIR / "legacy_inputs"
TEST_OUTPUT_DIR = OUTPUTS_DIR / "tests"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s",
        force=True,
    )


def default_test_output_dir(name: str) -> Path:
    return ensure_output_dir("tests", name)


def legacy_input_path(*parts: str) -> Path:
    return LEGACY_INPUT_DIR.joinpath(*parts)


def workload_stem(workload_path: str | Path) -> str:
    path = Path(workload_path)
    return path.stem if path.suffix else path.name


def build_experiment_id(
    *,
    accelerator: str | Path,
    workload: str | Path,
    mode: str,
    optimizer: Literal["co", "ga"],
) -> str:
    accelerator_name = Path(accelerator).stem
    return f"{accelerator_name}-{workload_stem(workload)}-{mode}-{optimizer}"


def export_scme_json(scme, *, output_dir: str | Path, experiment_id: str) -> Path:
    output_path = Path(output_dir)
    cost_lut_path = output_path / experiment_id / "cost_lut.pickle"
    json_path = output_path / experiment_id / "scme.json"
    cost_lut = CostModelEvaluationLUT(str(cost_lut_path))
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=str(json_path))
    return json_path


def run_allocation(
    *,
    hardware: str | Path,
    workload: str | Path,
    mapping: str | Path,
    mode: Literal["lbl", "fused"],
    layer_stacks: list[tuple[int, ...]],
    optimizer: Literal["co", "ga"],
    output_dir: str | Path,
    experiment_id: str | None = None,
    skip_if_exists: bool = False,
    ga_generations: int = 8,
    ga_individuals: int = 8,
    num_procs: int = 1,
    beam_width: int = 1,
):
    ensure_gurobi_license()
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    if experiment_id is None:
        experiment_id = build_experiment_id(
            accelerator=hardware,
            workload=workload,
            mode=mode,
            optimizer=optimizer,
        )

    common_kwargs = {
        "hardware": str(hardware),
        "workload": str(workload),
        "mapping": str(mapping),
        "mode": mode,
        "layer_stacks": layer_stacks,
        "experiment_id": experiment_id,
        "output_path": str(resolved_output),
        "skip_if_exists": skip_if_exists,
    }

    if optimizer == "co":
        scme = optimize_allocation_co(
            **common_kwargs,
            num_procs=num_procs,
        )
    else:
        scme = optimize_allocation_ga(
            **common_kwargs,
            nb_ga_generations=ga_generations,
            nb_ga_individuals=ga_individuals,
            num_procs=num_procs,
            coala_beam_width=beam_width,
        )

    export_scme_json(scme, output_dir=resolved_output, experiment_id=experiment_id)
    return scme, experiment_id


def prepare_flash_attention_test_case(
    *,
    num_cores: int,
    seq_len: int,
    embedding_dim: int,
    tile_size: int,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    run_dir = Path(output_dir)
    generated_dir = run_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    workload_source = export_flash_attention_onnx(
        seq_len,
        embedding_dim,
        tile_size,
        generated_dir,
        include_linear_layers=True,
    )
    workload_path = prepare_workload_copy(workload_source, run_dir / "workload.onnx")
    mapping_path = generate_flash_attention_mapping_config(
        num_qkv_tiles=seq_len // tile_size,
        num_cores=num_cores,
        output_path=generated_dir / f"FA_{num_cores}gemm_{seq_len // tile_size}tiles.yaml",
    )
    accelerator_path = get_multicore_config_path(num_cores)
    return workload_path, accelerator_path, mapping_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed
