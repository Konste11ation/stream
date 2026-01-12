import sys
import os
from pathlib import Path
import argparse

# Resolve paths early
CURRENT_DIR = Path(__file__).resolve().parent  
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent

sys.path.append(str(STREAM_WORKDIR))  
import logging as _logging
from stream_dvfs.src.config_library import W8A8
from stream_dvfs.src.config import AttentionHeadConfig, FlashAttentionConfig
from stream_dvfs.src.util import Stage, get_onnx_path  # noqa: E402 
from stream_dvfs.src.export_onnx import export_model_to_onnx  # noqa: E402 
from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.perfetto import convert_scme_to_perfetto_json
import re
from stream_dvfs.scripts_fa.utils import sanity_check, compare_energy
_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


def gen_flash_attention_onnx(seq_len:int, embedding_dim:int, tile_size:int, output_dir: str):
    flash_attention_config = FlashAttentionConfig(
        seq_len=seq_len,
        input_dim=embedding_dim,
        dim_k=embedding_dim,
        dim_v=embedding_dim,
        tile_Br=tile_size,
        tile_Bc=tile_size,
        batch_size=1,
        name=f"FlashAttention"
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
    tpl_mapping_path = str(CURRENT_DIR / "inputs" / "mappings" / f"FA_{num_cores}gemm_{num_cores}simd.yaml.tpl")
    mapping_output_path = str(CURRENT_DIR / "inputs" / "mappings" / f"FA_{num_cores}gemm_{num_cores}simd_{num_qkv_tiles}tiles.yaml")
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

def run_stream_dvfs_fa(seq_len:int, embedding_dim:int, tile_size:int, num_cores: int, output_dir: str):
    workload_path = str(CURRENT_DIR / "inputs" / "workloads" / f"FlashAttention_B=1_Seq={seq_len}_Embed={embedding_dim}_TileBr={tile_size}_TileBc={tile_size}_W8A8.onnx")
    accelerator = str(CURRENT_DIR / "inputs" / "multicores" / f"FA_{num_cores}gemm_{num_cores}simd.yaml")
    mapping_path = str(CURRENT_DIR / "inputs" / "mappings" / f"FA_{num_cores}gemm_{num_cores}simd_{seq_len//tile_size}tiles.yaml")

    output_dir = str(CURRENT_DIR / "outputs/")
    mode = "fused"
    layer_stacks = [tuple(range(0, 1000))]
    experiment_id = f"{num_cores}gemm_{num_cores}simd_FlashAttention_Seq{seq_len}_Embed{embedding_dim}_Tile{tile_size}_W8A8_ga"
    nb_ga_generations = 8
    nb_ga_individuals = 8
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
    )
    cost_lut_path = f"{output_dir}/{experiment_id}/cost_lut.pickle"
    cost_lut = CostModelEvaluationLUT(cost_lut_path)
    json_path = f"{output_dir}/{experiment_id}/scme.json"
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)
    return scme

def run_stream_dvfs_attention(seq_len:int, embedding_dim:int, num_cores: int, output_dir: str):
    workload_path = str(CURRENT_DIR / "inputs" / "workloads" / f"AttentionHead_B=1_Seq={seq_len}_Embed={embedding_dim}_W8A8.onnx")
    accelerator = str(CURRENT_DIR / "inputs" / "multicores" / f"FA_{num_cores}gemm_{num_cores}simd.yaml")
    mapping_path = str(CURRENT_DIR / "inputs" / "mappings" / f"AH_{num_cores}gemm_{num_cores}simd.yaml")
    output_dir = str(CURRENT_DIR / "outputs/")
    mode = "lbl"
    layer_stacks = []
    experiment_id = f"{num_cores}gemm_{num_cores}simd_AttentionHead_Seq{seq_len}_Embed{embedding_dim}_W8A8_ga"
    nb_ga_generations = 8
    nb_ga_individuals = 8
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
    )
    cost_lut_path = f"{output_dir}/{experiment_id}/cost_lut.pickle"
    cost_lut = CostModelEvaluationLUT(cost_lut_path)
    json_path = f"{output_dir}/{experiment_id}/scme.json"
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)
    return scme


if __name__ == "__main__":
    # future main code
    # for seq_len in [32, 64, 128]:
    #     embedding_dim = 128
    #     tile_size = 16
    #     gen_flash_attention_onnx(seq_len, embedding_dim, output_dir=str(CURRENT_DIR / "inputs" / "workloads"))
    #     gen_flash_attention_mapping_config(num_qkv_tiles=seq_len//tile_size, output_dir=str(CURRENT_DIR / "inputs" / "mappings"))
    #     run_stream_dvfs_fa(seq_len, embedding_dim, tile_size=tile_size, output_dir=str(CURRENT_DIR / "outputs"))
    
    # Test code
    num_cores = 1
    seq_len = 256
    embedding_dim = 1024
    tile_size = 64
    # Run the FA test
    gen_flash_attention_onnx(seq_len, embedding_dim, tile_size, output_dir=str(CURRENT_DIR / "inputs" / "workloads"))
    gen_flash_attention_mapping_config(num_qkv_tiles=seq_len//tile_size, num_cores=num_cores)
    scme_fa = run_stream_dvfs_fa(seq_len, embedding_dim, tile_size=tile_size, num_cores=num_cores, output_dir=str(CURRENT_DIR / "outputs"))
    # Run the Attention Head test
    gen_attention_head_onnx(seq_len, embedding_dim, output_dir=str(CURRENT_DIR / "inputs" / "workloads"))
    scme_ah = run_stream_dvfs_attention(seq_len, embedding_dim, num_cores=num_cores, output_dir=str(CURRENT_DIR / "outputs"))
    
    # Compare
    compare_energy(scme_fa, scme_ah)