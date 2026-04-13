import sys
import os
from pathlib import Path
import argparse

# Resolve paths early
CURRENT_DIR = Path(__file__).resolve().parent  
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
STREAM_DEV_DIR = STREAM_WORKDIR.parent
os.environ['GRB_LICENSE_FILE'] = f'{STREAM_DEV_DIR}/gurobi.lic'
sys.path.append(str(STREAM_WORKDIR))  
import logging as _logging
from stream_dvfs.src.config_library import W8A8
from stream_dvfs.src.config import AttentionHeadConfig, FlashAttentionConfig
from stream_dvfs.src.util import Stage, get_onnx_path  # noqa: E402 
from stream_dvfs.src.export_onnx import export_model_to_onnx  # noqa: E402 
from stream.stream.api import optimize_allocation_ga, optimize_allocation_co
from stream.stream.utils import CostModelEvaluationLUT
from stream.stream.visualization.perfetto import convert_scme_to_perfetto_json
import re
from stream_dvfs.scripts_fa.utils import get_node_communication_energy, sanity_check, compare_energy
import pickle
from stream_dvfs.src.dvfs_optimization import DvfsOptimizationStage
from stream_dvfs.scripts_fa.analyze_dvfs_potential import analyze_scme_json, print_comparison_summary
_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


def gen_flash_attention_onnx(seq_len:int, embedding_dim:int, tile_size:int, output_dir: str, include_linear_layers: bool = True):
    flash_attention_config = FlashAttentionConfig(
        seq_len=seq_len,
        input_dim=embedding_dim,
        dim_k=embedding_dim,
        dim_v=embedding_dim,
        tile_Br=tile_size,
        tile_Bc=tile_size,
        batch_size=1,
        name=f"FlashAttention",
        include_linear_layers=include_linear_layers
    )
    onnx_output_path = get_onnx_path(output_dir=output_dir,
                              model=flash_attention_config,
                              quant=W8A8)
    if os.path.exists(onnx_output_path):
        print(f"ONNX model already exists at: {onnx_output_path}, skipping export.")
        return
    export_model_to_onnx(
        model_config=flash_attention_config,
        quant_config=W8A8,
        output_path=onnx_output_path
    )
    print(f"Exported Flash Attention ONNX model to: {onnx_output_path}")

def gen_attention_head_onnx(seq_len:int, embedding_dim:int, output_dir: str):
    # The attention head does not have tiling parameters
    attention_head_config = AttentionHeadConfig(
        seq_len=seq_len,
        input_dim=embedding_dim,
        dim_k=embedding_dim,
        dim_v=embedding_dim,
        batch_size=1,
        name=f"AttentionHead"
    )
    onnx_output_path = get_onnx_path(output_dir=output_dir,
                              model=attention_head_config,
                              quant=W8A8)
    if os.path.exists(onnx_output_path):
        print(f"ONNX model already exists at: {onnx_output_path}, skipping export.")
        return
    export_model_to_onnx(
        model_config=attention_head_config,
        quant_config=W8A8,
        output_path=onnx_output_path
    )
    print(f"Exported Attention Head ONNX model to: {onnx_output_path}")

def gen_flash_attention_mapping_config(num_qkv_tiles: int, num_cores: int =1):
    # We should generate the mapping files here if needed
    tpl_mapping_path = str(CURRENT_DIR / "inputs" / "mappings" / f"FA_{num_cores}gemm.yaml.tpl")
    mapping_output_path = str(CURRENT_DIR / "inputs" / "mappings" / f"FA_{num_cores}gemm_{num_qkv_tiles}tiles.yaml")
    if os.path.exists(mapping_output_path):
        print(f"Mapping config already exists at: {mapping_output_path}, skipping generation.")
        return
    with open(tpl_mapping_path, 'r') as tpl_file:
        tpl_content = tpl_file.read()
        mapping_content = tpl_content.replace("<num_qkv_tiles>", str(num_qkv_tiles))
    with open(mapping_output_path, 'w') as mapping_file:
        mapping_file.write(mapping_content)
    print(f"Generated mapping config at: {mapping_output_path}")

def gen_flash_attention_multicore_config(output_dir: str):
    # We should generate the multicore config files here if needed
    pass

def run_stream_fa(seq_len:int, embedding_dim:int, tile_size:int, num_cores: int, output_dir: str, include_linear_layers: bool = True):
    # Build a FlashAttentionConfig to compute the correct workload path using the util helper
    flash_attention_cfg = FlashAttentionConfig(
        seq_len=seq_len,
        input_dim=embedding_dim,
        dim_k=embedding_dim,
        dim_v=embedding_dim,
        tile_Br=tile_size,
        tile_Bc=tile_size,
        batch_size=1,
        name="FlashAttention",
        include_linear_layers=include_linear_layers
    )
    workload_path = get_onnx_path(output_dir=str(CURRENT_DIR / "inputs" / "workloads"),
                                  model=flash_attention_cfg,
                                  quant=W8A8)
    accelerator = str(CURRENT_DIR / "inputs" / "multicores" / f"FA_{num_cores}gemm.yaml")
    mapping_path = str(CURRENT_DIR / "inputs" / "mappings" / f"FA_{num_cores}gemm_{seq_len//tile_size}tiles.yaml")
    dvfs_cfg = str(CURRENT_DIR / "inputs" / "dvfs" / "coarse_dvfs.yaml")
    # output_dir is passed as argument
    mode = "fused"
    layer_stacks = [tuple(range(0, 100000))]
    # reuse the config's parameterized name for consistency
    experiment_id = f"{num_cores}gemm_{flash_attention_cfg.parameterized_name}_{W8A8.name}_ga"
    # Optimization Strategy:
    nb_ga_generations = 128
    nb_ga_individuals = 128
    fitness_cache_size = 300_000
    early_stopping_patience = 24
    early_stopping_min_generations = 48
    sanity_check(
        workload_path=workload_path,
        accelerator_path=accelerator,
        mapping_path=mapping_path,
        output_yaml_path=f"{output_dir}/{experiment_id}/workload_mapping.yaml"
    )
    scme = optimize_allocation_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=nb_ga_generations,
        nb_ga_individuals=nb_ga_individuals,
        experiment_id=experiment_id,
        output_path=output_dir,
        skip_if_exists=False,
        num_procs=32,
        coala_beam_width=1,
        do_dvfs_cooptimization=True,
        dvfs_config_path=dvfs_cfg,
        # Tuned GA parameters for convergence
        prob_crossover=0.7,
        prob_mutation=0.3, # Sum must be <= 1.0 for DEAP varOr
        fitness_cache_size=fitness_cache_size,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_generations=early_stopping_min_generations,
    )
    # scme = optimize_allocation_co(
    #     hardware=accelerator,
    #     workload=workload_path,
    #     mapping=mapping_path,
    #     mode=mode,
    #     layer_stacks=layer_stacks,
    #     experiment_id=experiment_id,
    #     output_path=output_dir,
    #     skip_if_exists=False,
    #     num_procs=32,
    # )
    cost_lut_path = f"{output_dir}/{experiment_id}/cost_lut.pickle"
    cost_lut = CostModelEvaluationLUT(cost_lut_path)
    json_path = f"{output_dir}/{experiment_id}/scme.json"
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)
    return scme

def run_stream_attention(seq_len:int, embedding_dim:int, num_cores: int, output_dir: str):
    # Use get_onnx_path to ensure naming stays consistent with exports
    attention_cfg = AttentionHeadConfig(
        seq_len=seq_len,
        input_dim=embedding_dim,
        dim_k=embedding_dim,
        dim_v=embedding_dim,
        batch_size=1,
        name="AttentionHead"
    )
    workload_path = get_onnx_path(output_dir=str(CURRENT_DIR / "inputs" / "workloads"),
                                  model=attention_cfg,
                                  quant=W8A8)
    accelerator = str(CURRENT_DIR / "inputs" / "multicores" / f"FA_{num_cores}gemm.yaml")
    mapping_path = str(CURRENT_DIR / "inputs" / "mappings" / f"AH_{num_cores}gemm.yaml")
    dvfs_cfg = str(CURRENT_DIR / "inputs" / "dvfs" / "coarse_dvfs.yaml")
    # respect the output_dir argument rather than hard-coding
    mode = "fused"
    layer_stacks = [tuple(range(0, 100000))]
    experiment_id = f"{num_cores}gemm_{attention_cfg.parameterized_name}_{W8A8.name}_ga"
    nb_ga_generations = 32
    nb_ga_individuals = 32
    sanity_check(
        workload_path=workload_path,
        accelerator_path=accelerator,
        mapping_path=mapping_path,
        output_yaml_path=f"{output_dir}/{experiment_id}/workload_mapping.yaml"
    )
    scme = optimize_allocation_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=nb_ga_generations,
        nb_ga_individuals=nb_ga_individuals,
        experiment_id=experiment_id,
        output_path=output_dir,
        skip_if_exists=True,
        num_procs=32,
        coala_beam_width=1,
        do_dvfs_cooptimization=True,
        dvfs_config_path=dvfs_cfg,
        # Tuned GA parameters for convergence
        prob_crossover=0.7,
        prob_mutation=0.3, # Sum must be <= 1.0 for DEAP varOr
    )
    cost_lut_path = f"{output_dir}/{experiment_id}/cost_lut.pickle"
    cost_lut = CostModelEvaluationLUT(cost_lut_path)
    json_path = f"{output_dir}/{experiment_id}/scme.json"
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)
    return scme

if __name__ == "__main__":

    # Test code
    num_cores = 4
    seq_len = 4096
    embedding_dim = 512
    tile_size = 1024
    
    # Run the FA test
    gen_flash_attention_onnx(seq_len, embedding_dim, tile_size, output_dir=str(CURRENT_DIR / "inputs" / "workloads"), include_linear_layers=True)
    gen_flash_attention_mapping_config(num_qkv_tiles=seq_len//tile_size, num_cores=num_cores)
    scme_fa = run_stream_fa(
        seq_len,
        embedding_dim,
        tile_size=tile_size,
        num_cores=num_cores,
        output_dir=str(CURRENT_DIR / "outputs"),
        include_linear_layers=True,
    )
    # reconstruct the experiment_id exactly as `run_stream_fa` would have
    fa_cfg = FlashAttentionConfig(
        seq_len=seq_len,
        input_dim=embedding_dim,
        dim_k=embedding_dim,
        dim_v=embedding_dim,
        tile_Br=tile_size,
        tile_Bc=tile_size,
        batch_size=1,
        name="FlashAttention",
        include_linear_layers=True,
    )
    experiment_id = f"{num_cores}gemm_{fa_cfg.parameterized_name}_{W8A8.name}_ga"
    json_path = f"{CURRENT_DIR}/outputs/{experiment_id}/scme.json"
    if os.path.exists(json_path):
        res = analyze_scme_json(json_path)
        print_comparison_summary([res])
        
    # Run the Attention Head test
    # gen_attention_head_onnx(seq_len, embedding_dim, output_dir=str(CURRENT_DIR / "inputs" / "workloads"))
    # scme_ah = run_stream_attention(seq_len, embedding_dim, num_cores=num_cores, output_dir=str(CURRENT_DIR / "outputs"))
    
    # # Compare
    # compare_energy(scme_fa, scme_ah)
