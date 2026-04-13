from __future__ import annotations

import argparse
import logging
from pathlib import Path

from stream_dvfs.tests.common import (
    configure_logging,
    default_test_output_dir,
    legacy_input_path,
    run_allocation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the legacy FlashAttention parser validation workload.")
    parser.add_argument(
        "--workload",
        type=Path,
        default=legacy_input_path(
            "workloads",
            "FlashAttentionTest_B=1_FULL_PREFILL_SIZE=1_DECODE_SIZE=1_W8A8_Prefill.onnx",
        ),
        help="Path to the legacy FlashAttention ONNX workload.",
    )
    parser.add_argument(
        "--accelerator",
        type=Path,
        default=legacy_input_path("multicore_system", "FA_4gemm_4simd.yaml"),
        help="Path to the accelerator YAML file.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=legacy_input_path("multicore_mapping", "FA_4gemm_4simd_hand_mapping.yaml"),
        help="Path to the mapping YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_test_output_dir("flash_attention_parser"),
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("co", "ga"),
        default="ga",
        help="Allocation backend to run.",
    )
    parser.add_argument("--ga-generations", type=int, default=8, help="GA generations when using `ga`.")
    parser.add_argument("--ga-individuals", type=int, default=8, help="GA population size when using `ga`.")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--skip-if-exists", action="store_true", help="Reuse existing SCME output when available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    _, experiment_id = run_allocation(
        hardware=args.accelerator,
        workload=args.workload,
        mapping=args.mapping,
        mode="fused",
        layer_stacks=[tuple(range(0, 1000))],
        optimizer=args.optimizer,
        output_dir=args.output_dir,
        skip_if_exists=args.skip_if_exists,
        ga_generations=args.ga_generations,
        ga_individuals=args.ga_individuals,
        num_procs=args.num_procs,
    )
    print(f"Saved outputs to {Path(args.output_dir) / experiment_id}")


if __name__ == "__main__":
    main()
