from __future__ import annotations

import argparse
import logging
from pathlib import Path

from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.tiled_workload_generation import TiledWorkloadGenerationStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.stage import MainStage, Stage
from stream.workload.utils import visualize_computation_workload
from stream_dvfs.tests.common import configure_logging, default_test_output_dir, prepare_flash_attention_test_case

logger = logging.getLogger(__name__)


class GeometricPropagationValidationStage(Stage):
    def is_leaf(self) -> bool:
        return True

    def run(self):
        tiled_workload = self.kwargs["workload"]
        logger.info("Validating geometric propagation on tiled workload with %s nodes.", len(tiled_workload.node_list))

        nodes_by_layer: dict[str, list] = {}
        for node in tiled_workload.node_list:
            nodes_by_layer.setdefault(node.name, []).append(node)

        total_edges = tiled_workload.number_of_edges()
        logger.info("Layers found: %s", list(nodes_by_layer.keys()))
        logger.info("Total edges in tiled graph: %s", total_edges)
        if total_edges == 0:
            raise RuntimeError("No dependency edges were generated in the tiled workload.")

        projection_layers = ("/q_proj", "/k_proj", "/v_proj", "/o_proj")
        for producer, consumer, data in tiled_workload.edges(data=True):
            producer_name = str(producer.name)
            consumer_name = str(consumer.name)
            if any(tag in producer_name for tag in projection_layers) or any(tag in consumer_name for tag in projection_layers):
                logger.info("Edge: %s -> %s | operand=%s", producer, consumer, data.get("operand"))

        output_dir = Path(self.kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        visualize_computation_workload(
            tiled_workload,
            filepath=str(output_dir / "tiled_workload_geometric.png"),
            cluster_by=None,
        )
        yield tiled_workload, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate geometric propagation for a generated FlashAttention workload.")
    parser.add_argument("--num-cores", type=int, default=1, help="Number of GEMM cores in the generated hardware config.")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length for the generated workload.")
    parser.add_argument("--embedding-dim", type=int, default=16, help="Embedding dimension for the generated workload.")
    parser.add_argument("--tile-size", type=int, default=16, help="Tile size for generated FlashAttention tiles.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_test_output_dir("geometric_propagation"),
        help="Directory for generated test artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    workload_path, accelerator_path, mapping_path = prepare_flash_attention_test_case(
        num_cores=args.num_cores,
        seq_len=args.seq_len,
        embedding_dim=args.embedding_dim,
        tile_size=args.tile_size,
        output_dir=args.output_dir,
    )

    mainstage = MainStage(
        [
            AcceleratorParserStage,
            StreamONNXModelParserStage,
            LayerStacksGenerationStage,
            TilingGenerationStage,
            TiledWorkloadGenerationStage,
            GeometricPropagationValidationStage,
        ],
        accelerator=str(accelerator_path),
        workload_path=str(workload_path),
        mapping_path=str(mapping_path),
        mode="fused",
        layer_stacks=[tuple(range(0, 1000))],
        loma_lpf_limit=6,
        nb_ga_individuals=4,
        nb_ga_generations=4,
        tiled_workload_path=str(args.output_dir / "tiled_workload_test.pickle"),
        dump_filename_pattern=str(args.output_dir / "final_output"),
        plot_hof=True,
        plot_file_name=str(args.output_dir / "hof_plot"),
        plot_per_core=True,
        output_dir=str(args.output_dir),
    )
    mainstage.run()
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
