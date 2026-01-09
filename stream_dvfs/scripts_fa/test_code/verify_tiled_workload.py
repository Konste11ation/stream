import sys
import os
from pathlib import Path
import logging

# Resolve paths
CURRENT_DIR = Path(__file__).resolve().parent
SCRIPTS_FA_DIR = CURRENT_DIR.parent
STREAM_DVFS_DIR = SCRIPTS_FA_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent

sys.path.append(str(STREAM_WORKDIR))

from stream.stages.stage import Stage, MainStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.generation.tiled_workload_generation import TiledWorkloadGenerationStage
from stream.workload.computation.computation_node import ComputationNode

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# LoopOffsetsStage removed as we fixed the core TiledWorkloadGenerationStage


class WorkloadInspectionStage(Stage):
    """
    Stage to inspect the generated tiled workload.
    """
    def is_leaf(self) -> bool:
        return True

    def run(self):
        tiled_workload = self.kwargs["workload"]
        logger.info(f"Inspecting Tiled Workload: {tiled_workload}")
        
        node_list = tiled_workload.node_list
        logger.info(f"Total Nodes: {len(node_list)}")
        
        # Group nodes by type or name
        nodes_by_name = {}
        for node in node_list:
            if node.name not in nodes_by_name:
                nodes_by_name[node.name] = []
            nodes_by_name[node.name].append(node)
            
        for name, nodes in nodes_by_name.items():
            logger.info(f"Node Group '{name}': {len(nodes)} tiles")

        # Inspect details of q_proj, k_proj, v_proj and gemm_qk
        for name, nodes in nodes_by_name.items():
            if "q_proj" in name or "k_proj" in name or "gemm_qk" in name or "v_proj" in name:
                 logger.info(f"--- Node Detail Group: {name} ---")
                 # Print Irrelevant Dimensions for Output
                 from zigzag.datatypes import Constants
                 ir_dims = nodes[0].loop_relevancy_info.get_ir_layer_dims(Constants.OUTPUT_LAYER_OP)
                 logger.info(f"  Output Irrelevant Dims: {ir_dims}")
                 
                 for node in nodes:
                     logger.info(f"  Node: {node}")
                     logger.info(f"    Loop Ranges: {node.loop_ranges}")
                     for op, tensor in node.operand_tensors.items():
                         logger.info(f"    Operand {op}: {tensor} | loop_ranges: {tensor.loop_ranges}")
                         # Print dim order for all interactions
                         logger.info(f"    Operand {op} Dim Order: {node.operand_dimensionality_order.get(op)}")
                     
                     logger.info(f"    Dimension Relations: {node.dimension_relations}")

        # Inspect Edges
        logger.info("\n--- Dependency Inspection ---")
        edges = list(tiled_workload.edges(data=True))
        logger.info(f"Total Edges: {len(edges)}")
        
        # Specifically look for MatMul -> GEMM connections (Flash Attention logic)
        for u, v, data in edges:
            if "MatMul" in u.name and "gemm" in v.name:
                 logger.info(f"Edge: {u} -> {v} | Data: {data}")

        # Also just print some general sampling of edges
        for i, (u, v, data) in enumerate(edges[:20]):
            logger.info(f"Sample Edge {i}: {u} -> {v} | Operand: {data.get('operand', 'N/A')}")

        yield tiled_workload, None

def run_test():
    # Parameters matching stream_dvfs_fa.py test code
    num_cores = 1
    seq_len = 32
    embedding_dim = 16
    tile_size = 16
    
    # Paths
    workload_path = str(SCRIPTS_FA_DIR / "inputs" / "workloads" / f"FlashAttention_B=1_Seq={seq_len}_Embed={embedding_dim}_TileBr={tile_size}_TileBc={tile_size}_W8A8.onnx")
    accelerator_path = str(SCRIPTS_FA_DIR / "inputs" / "multicores" / f"FA_{num_cores}gemm_{num_cores}simd.yaml")
    mapping_path = str(SCRIPTS_FA_DIR / "inputs" / "mappings" / f"FA_{num_cores}gemm_{num_cores}simd_{seq_len//tile_size}tiles.yaml")
    
    output_dir = str(CURRENT_DIR / "outputs")
    experiment_id = "test_verification"
    
    # Ensure inputs exist (assuming stream_dvfs_fa.py has been run or files exist)
    if not os.path.exists(workload_path):
        logger.error(f"Workload file not found: {workload_path}. Please run stream_dvfs_fa.py first to generate inputs.")
        return

    # Setup Stages
    # Mimic api.py Optimize Allocation logic but swap out the estimation/optimization stages for our inspector
    mainstage = MainStage(
        [
            AcceleratorParserStage,
            StreamONNXModelParserStage,
            # FixFlashAttentionDimensionsStage,
            LayerStacksGenerationStage,
            TilingGenerationStage,
            TiledWorkloadGenerationStage,
            WorkloadInspectionStage  # Our custom inspector
        ],
        accelerator=accelerator_path,
        workload_path=workload_path,
        mapping_path=mapping_path,
        loma_lpf_limit=6,
        mode="fused",
        layer_stacks=[tuple(range(0, 1000))], # Single stack
        tiled_workload_path=f"{output_dir}/tiled_workload_test.pickle",
        # Dummy paths for arguments required by stages we might not fully utilize but need initialization
        cost_lut_path=f"{output_dir}/cost_lut.pickle",
        temporal_mapping_type="uneven",
    )

    logger.info("Starting MainStage...")
    mainstage.run()
    logger.info("Test Complete.")

if __name__ == "__main__":
    run_test()
