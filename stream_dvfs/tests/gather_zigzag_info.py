from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

from zigzag.visualization import bar_plot_cost_model_evaluations_breakdown


def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    filename = re.sub(r"[\s(),]", "_", filename)
    filename = re.sub(r"_+", "_", filename)
    return filename.strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a ZigZag cost LUT and emit per-node breakdown plots.")
    parser.add_argument(
        "cost_lut",
        type=Path,
        nargs="?",
        default=Path(
            "stream_dvfs/outputs/attention_head-AttentionHeadTest_B=1_FULL_PREFILL_SIZE=1_DECODE_SIZE=1_W8A8_Decode-fused-ga/cost_lut.pickle"
        ),
        help="Path to a `cost_lut.pickle` file.",
    )
    parser.add_argument(
        "--breakdown-dir",
        type=Path,
        default=None,
        help="Directory for breakdown plots. Defaults to `<cost_lut dir>/breakdown`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    breakdown_dir = args.breakdown_dir or args.cost_lut.parent / "breakdown"
    breakdown_dir.mkdir(parents=True, exist_ok=True)

    with args.cost_lut.open("rb") as handle:
        cmes = pickle.load(handle)

    total_stall = 0
    for node, node_cmes in cmes.items():
        node_name = str(node)
        last_cme = None
        for cme in node_cmes.values():
            last_cme = cme
            mem_names = [mem_level.memory_instance.name for mem_level in cme.mem_level_list]
            node_stall = cme.latency_total0 - cme.ideal_temporal_cycle
            total_stall += node_stall
            for mem_name, ports_slack in zip(mem_names, cme.stall_slack_comb_collect):
                print(f"{node_name} | {mem_name}: {ports_slack}")
            print(
                "Latency: "
                f"{cme.latency_total2:.3e} "
                f"(ideal={cme.ideal_temporal_cycle}, "
                f"spatial_stall={cme.ideal_temporal_cycle - cme.ideal_cycle}, "
                f"temporal_stall={cme.latency_total0 - cme.ideal_temporal_cycle}, "
                f"onload={cme.latency_total1 - cme.latency_total0}, "
                f"offload={cme.latency_total2 - cme.latency_total1})"
            )

        if last_cme is None:
            continue

        save_path = breakdown_dir / f"breakdown_{sanitize_filename(node_name)}.png"
        bar_plot_cost_model_evaluations_breakdown([last_cme], save_path=str(save_path))

    print(f"Total temporal stall across all nodes: {total_stall}")


if __name__ == "__main__":
    main()
