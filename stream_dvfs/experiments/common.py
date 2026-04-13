from __future__ import annotations

import os
from pathlib import Path

import yaml
from zigzag.datatypes import Constants
from zigzag.utils import open_yaml

from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.mapping_parser import MappingParser
from stream.parser.onnx.model import ONNXModelParser
from stream_dvfs.experiments.modeling.config import AttentionHeadConfig, FlashAttentionConfig
from stream_dvfs.experiments.modeling.config_library import W8A8
from stream_dvfs.experiments.modeling.export_onnx import export_model_to_onnx
from stream_dvfs.experiments.modeling.util import get_onnx_path
from stream_dvfs.paths import MAPPING_CONFIG_DIR, MULTICORE_CONFIG_DIR, ensure_gurobi_license

ensure_gurobi_license()


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_yaml(path: str | Path, data: object) -> Path:
    resolved = ensure_parent_dir(path)
    with resolved.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
    return resolved


def export_attention_head_onnx(seq_len: int, embedding_dim: int, output_dir: str | Path) -> Path:
    attention_head_config = AttentionHeadConfig(
        seq_len=seq_len,
        input_dim=embedding_dim,
        dim_k=embedding_dim,
        dim_v=embedding_dim,
        batch_size=1,
        name="AttentionHead",
    )
    onnx_output_path = Path(get_onnx_path(output_dir=str(output_dir), model=attention_head_config, quant=W8A8))
    if onnx_output_path.exists():
        return onnx_output_path

    export_model_to_onnx(
        model_config=attention_head_config,
        quant_config=W8A8,
        output_path=str(onnx_output_path),
    )
    return onnx_output_path


def export_flash_attention_onnx(
    seq_len: int,
    embedding_dim: int,
    tile_size: int,
    output_dir: str | Path,
    *,
    include_linear_layers: bool = True,
    seq_len_q: int | None = None,
) -> Path:
    flash_attention_config = FlashAttentionConfig(
        seq_len=seq_len,
        seq_len_q=seq_len_q if seq_len_q is not None else seq_len,
        input_dim=embedding_dim,
        dim_k=embedding_dim,
        dim_v=embedding_dim,
        tile_Br=tile_size,
        tile_Bc=tile_size,
        batch_size=1,
        name="FlashAttention",
        include_linear_layers=include_linear_layers,
    )
    onnx_output_path = Path(get_onnx_path(output_dir=str(output_dir), model=flash_attention_config, quant=W8A8))
    if onnx_output_path.exists():
        return onnx_output_path

    export_model_to_onnx(
        model_config=flash_attention_config,
        quant_config=W8A8,
        output_path=str(onnx_output_path),
    )
    return onnx_output_path


def write_attention_single_core_mapping(output_path: str | Path) -> Path:
    mapping = [
        {"name": "default", "core_allocation": [0]},
        {
            "name": "MatMul",
            "core_allocation": [0],
            "intra_core_tiling": ["D, 1"],
            "inter_core_tiling": ["B, 1"],
        },
    ]
    return write_yaml(output_path, mapping)


def generate_flash_attention_mapping_config(num_qkv_tiles: int, num_cores: int, output_path: str | Path) -> Path:
    template_path = MAPPING_CONFIG_DIR / f"FA_{num_cores}gemm.yaml.tpl"
    resolved_output = ensure_parent_dir(output_path)
    template = template_path.read_text(encoding="utf-8")
    resolved_output.write_text(template.replace("<num_qkv_tiles>", str(num_qkv_tiles)), encoding="utf-8")
    return resolved_output


def get_multicore_config_path(num_cores: int) -> Path:
    return MULTICORE_CONFIG_DIR / f"FA_{num_cores}gemm.yaml"


def stage_run_dir(output_dir: str | Path, experiment_id: str) -> Path:
    run_dir = Path(output_dir) / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def generated_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_workload_copy(onnx_path: str | Path, destination: str | Path) -> Path:
    source = Path(onnx_path)
    target = ensure_parent_dir(destination)
    if source.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        os.replace(source, target)
    return target


def sanity_check(workload_path: str | Path, accelerator_path: str | Path, mapping_path: str | Path, output_yaml_path: str | Path):
    accelerator_data = open_yaml(str(accelerator_path))
    validator = AcceleratorValidator(accelerator_data, str(accelerator_path))
    accelerator_data = validator.normalized_data
    if not validator.validate():
        raise ValueError("Failed to validate user provided accelerator.")

    accelerator = AcceleratorFactory(accelerator_data).create()
    all_mappings = MappingParser(str(mapping_path)).run()
    onnx_model_parser = ONNXModelParser(str(workload_path), all_mappings, accelerator)
    onnx_model_parser.run()
    workload = onnx_model_parser.workload

    nodes_data = []
    for node in workload.node_list:
        nodes_data.append(
            {
                "id": getattr(node, "id", None),
                "name": getattr(node, "name", None),
                "operator_type": getattr(node, "type", None),
                "equation": getattr(getattr(node, "equation", None), "data", None),
                "layer_dim_sizes": str(getattr(node, "layer_dim_sizes", {})),
                "inter_core_tiling": str(getattr(node, "inter_core_tiling", {})),
                "intra_core_tiling": str(getattr(node, "intra_core_tiling", {})),
                "input_operand_source": str(getattr(node, "input_operand_source", {})),
            }
        )
    write_yaml(output_yaml_path, {"nodes": nodes_data})


def get_node_communication_energy(scme, target_nodes):
    total_energy = 0
    target_nodes_set = set(target_nodes)
    tensor_consumers = {}

    def get_core(node):
        allocation = node.chosen_core_allocation
        if isinstance(allocation, int):
            return scme.accelerator.get_core(allocation)
        return allocation

    for node in target_nodes:
        consumer_core = get_core(node)
        for predecessor in scme.workload.predecessors(node):
            if hasattr(predecessor, "operand_tensors") and Constants.OUTPUT_LAYER_OP in predecessor.operand_tensors:
                target_tensor = predecessor.operand_tensors[Constants.OUTPUT_LAYER_OP]
                tensor_consumers.setdefault(target_tensor.id, []).append((node, consumer_core))

    active_links = set()
    if hasattr(scme.accelerator, "communication_manager"):
        communication_manager = scme.accelerator.communication_manager
        if hasattr(communication_manager, "get_all_links"):
            for link in communication_manager.get_all_links():
                if link.events:
                    active_links.add(link)
        else:
            for all_links_pairs in communication_manager.all_pair_links.values():
                for link_pair in all_links_pairs:
                    for link in link_pair:
                        if link.events:
                            active_links.add(link)

    for link in active_links:
        for event in link.events:
            if event.tensor.origin in target_nodes_set:
                total_energy += event.energy
            elif event.tensor.id in tensor_consumers:
                receiver_core = event.receiver
                if any(consumer_core == receiver_core for _, consumer_core in tensor_consumers[event.tensor.id]):
                    total_energy += event.energy

    return total_energy


def compare_energy(scme_fa, scme_ah):
    fa_nodes = [node for node in scme_fa.workload.node_list if "FlashAttention" in node.name]
    fa_onchip = sum(node.get_onchip_energy() for node in fa_nodes)
    fa_offchip = get_node_communication_energy(scme_fa, fa_nodes)

    ah_prefixes = ["/MatMul", "/Softmax-max/", "/Softmax-exp/", "/Softmax-sum/", "/Softmax-div/", "/MatMul_1"]
    ah_nodes = [node for node in scme_ah.workload.node_list if any(node.name.startswith(prefix) for prefix in ah_prefixes)]
    ah_onchip = sum(node.get_onchip_energy() for node in ah_nodes)
    ah_offchip = sum(node.get_offchip_energy() for node in ah_nodes)

    print("=" * 60)
    print("Energy Comparison: FlashAttention (Fused) vs AttentionHead (Unfused)")
    print("=" * 60)
    print(f"Flash Attention total energy: {fa_onchip + fa_offchip:,.2f} pJ")
    print(f"Attention Head total energy: {ah_onchip + ah_offchip:,.2f} pJ")
    print("=" * 60)
