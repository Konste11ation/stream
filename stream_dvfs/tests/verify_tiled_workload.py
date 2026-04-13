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
from zigzag.datatypes import Constants
from stream_dvfs.tests.common import configure_logging, default_test_output_dir, prepare_flash_attention_test_case

logger = logging.getLogger(__name__)


class WorkloadInspectionStage(Stage):
    def is_leaf(self) -> bool:
        return True

    def run(self):
        tiled_workload = self.kwargs["workload"]
        logger.info("Inspecting tiled workload with %s nodes.", len(tiled_workload.node_list))

        nodes_by_name: dict[str, list] = {}
        for node in tiled_workload.node_list:
            nodes_by_name.setdefault(node.name, []).append(node)

        for name, nodes in nodes_by_name.items():
            logger.info("Node group '%s': %s tiles", name, len(nodes))

        for name, nodes in nodes_by_name.items():
            if not any(tag in name for tag in ("q_proj", "k_proj", "v_proj", "gemm_qk")):
                continue

            logger.info("--- Node detail group: %s ---", name)
            ir_dims = nodes[0].loop_relevancy_info.get_ir_layer_dims(Constants.OUTPUT_LAYER_OP)
            logger.info("Output irrelevant dims: %s", ir_dims)

            for node in nodes:
                logger.info("Node: %s", node)
                logger.info("Loop ranges: %s", node.loop_ranges)
                for operand, tensor in node.operand_tensors.items():
                    logger.info("Operand %s: %s | loop_ranges=%s", operand, tensor, tensor.loop_ranges)
                    logger.info("Operand %s dim order: %s", operand, node.operand_dimensionality_order.get(operand))
                logger.info("Dimension relations: %s", node.dimension_relations)

        edges = list(tiled_workload.edges(data=True))
        logger.info("Total edges: %s", len(edges))
        for producer, consumer, data in edges:
            if "MatMul" in producer.name and "gemm" in consumer.name:
                logger.info("MatMul -> gemm edge: %s -> %s | %s", producer, consumer, data)
        for index, (producer, consumer, data) in enumerate(edges[:20]):
            logger.info("Sample edge %s: %s -> %s | operand=%s", index, producer, consumer, data.get("operand", "N/A"))

        yield tiled_workload, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a generated tiled FlashAttention workload.")
    parser.add_argument("--num-cores", type=int, default=1, help="Number of GEMM cores in the generated hardware config.")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length for the generated workload.")
    parser.add_argument("--embedding-dim", type=int, default=16, help="Embedding dimension for the generated workload.")
    parser.add_argument("--tile-size", type=int, default=16, help="Tile size for generated FlashAttention tiles.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_test_output_dir("verify_tiled_workload"),
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
            WorkloadInspectionStage,
        ],
        accelerator=str(accelerator_path),
        workload_path=str(workload_path),
        mapping_path=str(mapping_path),
        loma_lpf_limit=6,
        mode="fused",
        layer_stacks=[tuple(range(0, 1000))],
        tiled_workload_path=str(args.output_dir / "tiled_workload_test.pickle"),
        cost_lut_path=str(args.output_dir / "cost_lut.pickle"),
        temporal_mapping_type="uneven",
    )
    mainstage.run()
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
