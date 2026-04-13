from __future__ import annotations

import argparse
import logging
from pathlib import Path

from stream_dvfs.tests.common import configure_logging, run_allocation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a generic LLM allocation workload.")
    parser.add_argument("-w", "--workload-path", type=Path, required=True, help="Path to the ONNX workload file.")
    parser.add_argument("-a", "--accelerator-yaml", type=Path, required=True, help="Path to the accelerator YAML file.")
    parser.add_argument("-m", "--mapping-yaml", type=Path, required=True, help="Path to the mapping YAML file.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Directory for generated outputs.")
    parser.add_argument("--optimizer", choices=("co", "ga"), default="ga", help="Allocation backend to run.")
    parser.add_argument("--mode", choices=("lbl", "fused"), default="fused", help="Allocation mode.")
    parser.add_argument("--layer-stack-end", type=int, default=67, help="Exclusive upper bound for the default layer stack.")
    parser.add_argument("--ga-generations", type=int, default=8, help="GA generations when using `ga`.")
    parser.add_argument("--ga-individuals", type=int, default=8, help="GA population size when using `ga`.")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--skip-if-exists", action="store_true", help="Reuse existing SCME output when available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    _, experiment_id = run_allocation(
        hardware=args.accelerator_yaml,
        workload=args.workload_path,
        mapping=args.mapping_yaml,
        mode=args.mode,
        layer_stacks=[tuple(range(0, args.layer_stack_end))],
        optimizer=args.optimizer,
        output_dir=args.output_dir,
        skip_if_exists=args.skip_if_exists,
        ga_generations=args.ga_generations,
        ga_individuals=args.ga_individuals,
        num_procs=args.num_procs,
    )
    print(f"Saved outputs to {args.output_dir / experiment_id}")


if __name__ == "__main__":
    main()
