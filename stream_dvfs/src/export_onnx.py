import sys
import os
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import NodeProto
from onnx.shape_inference import infer_shapes

CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
sys.path.append(str(STREAM_DVFS_DIR))

from src.config import (
    ModelConfig,
    QuantConfig,
    TransformerConfigSingleLayer,
    AttentionHeadConfig,
    FlashAttentionConfig
)
from src.config_library import OPT_2_7B, W32A32
from src.pytorch_models.transformer_model import LanguageModel
from src.pytorch_models.transformer_model_decode import LanguageModelDecode
from src.pytorch_models.attention_head import Self_Attention
from src.pytorch_models.custom_flash_attention import FlashAttentionModel
from src.util import Stage, get_onnx_path


def infer_shapes_with_custom_ops(model_onnx):
    # Run infer_shapes first pass
    model_onnx = infer_shapes(model_onnx)
    
    # Fix FlashAttention shapes
    from onnx import helper, TensorProto
    
    modified = False
    for node in model_onnx.graph.node:
        if node.op_type == 'FlashAttention':
            q_name = node.input[0]
            out_name = node.output[0]
            
            # Find q shape
            q_shape = None
            # Check value_info
            for vi in model_onnx.graph.value_info:
                if vi.name == q_name:
                    if vi.type.HasField('tensor_type') and vi.type.tensor_type.HasField('shape'):
                        q_shape = vi.type.tensor_type.shape
                        break
            # Check graph inputs
            if q_shape is None:
                for inp in model_onnx.graph.input:
                    if inp.name == q_name:
                         if inp.type.HasField('tensor_type') and inp.type.tensor_type.HasField('shape'):
                            q_shape = inp.type.tensor_type.shape
                            break
                            
            if q_shape:
                # Set output shape
                dims = []
                for d in q_shape.dim:
                    if d.HasField('dim_value'):
                        dims.append(d.dim_value)
                    elif d.HasField('dim_param'):
                        dims.append(d.dim_param)
                    else:
                        dims.append(None)
                
                # Check if it already exists in value_info
                exists = False
                for vi in model_onnx.graph.value_info:
                    if vi.name == out_name:
                        exists = True
                        break
                
                if not exists:
                    new_vi = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, dims)
                    model_onnx.graph.value_info.append(new_vi)
                    modified = True
                
                # Also update graph output if applicable
                for out in model_onnx.graph.output:
                    if out.name == out_name:
                        out.type.tensor_type.shape.Clear()
                        out.type.tensor_type.shape.CopyFrom(q_shape)
                        modified = True

    if modified:
        model_onnx = infer_shapes(model_onnx)
        
    return model_onnx


def export_model_to_onnx(
    model_config: ModelConfig,
    quant_config: QuantConfig,
    output_path: str = "outputs/custom_transformer.onnx",
    stage: Stage = Stage.PREFILL,
):

    config_single_layer = model_config.to_single_layer_config()
    match config_single_layer.type:

        case "Single_Layer_Model":
            export_transformer_to_onnx(
                config_single_layer,
                output_path,
                stage,
            )
        case "AttentionHead":
            export_attention_head_to_onnx(
                config_single_layer,
                output_path,
            )
        case "FlashAttention":
            export_flash_attention_to_onnx(
                config_single_layer,
                output_path,
            )
        case _:
            raise ValueError("config must be a single layer configuration")

    # Perform shape inference
    onnx_model = onnx.load(output_path)
    onnx_model = infer_shapes_with_custom_ops(onnx_model)

    # Add attribute with quantization info, to be used in Zigzag
    for node in onnx_model.graph.node:
        if node.op_type != "Constant":
            add_attribute_to_onnx_node(node, "weight_size", quant_config.weight_bits)
            add_attribute_to_onnx_node(node, "act_size", quant_config.act_bits)
            add_attribute_to_onnx_node(node, "output_size", quant_config.intermediate_output_bits)

    # Save the model with external data and then remove it
    # NOTE: This requires later loading it with load_external_data=False
    external_data_filename = "external.data"
    external_data_path = os.path.join(os.path.dirname(output_path), external_data_filename)
    onnx.save(onnx_model, output_path, save_as_external_data=True, location=external_data_filename)
    if os.path.exists(external_data_path):
        os.remove(external_data_path)

def export_attention_head_to_onnx(
    attention_head_config: AttentionHeadConfig,
    output_path: str = "outputs/attention_head.onnx",
):

    print(f"Generating ONNX model at {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_input = torch.randn(
        attention_head_config.batch_size,
        attention_head_config.seq_len,
        attention_head_config.input_dim,
    )

    pytorch_model = Self_Attention(
        attention_head_config.input_dim,
        attention_head_config.dim_k,
        attention_head_config.dim_v,
    )

    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_path,
        opset_version=16,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
        do_constant_folding=True,
        export_params=False,
    )

def export_flash_attention_to_onnx(
    flash_attention_config: FlashAttentionConfig,
    output_path: str = "outputs/flash_attention.onnx",
):
    print(f"Generating ONNX model at {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_input = torch.randn(
        flash_attention_config.batch_size,
        flash_attention_config.seq_len,
        flash_attention_config.input_dim,
    )
    # For now the input_dim and dim_k, dim_v must be equal
    pytorch_model = FlashAttentionModel(
        flash_attention_config.input_dim,
        flash_attention_config.dim_k,
        flash_attention_config.dim_v,
    )

    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_path,
        opset_version=16,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
        do_constant_folding=True,
        export_params=False,
    )



def export_transformer_to_onnx(
    transformer_config: TransformerConfigSingleLayer,
    path: str = "outputs/custom_transformer.onnx",
    stage: Stage = Stage.PREFILL,
):
    assert transformer_config.num_layer == 1
    print(f"Generating ONNX model at {path} ({stage})")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    
    if stage == Stage.PREFILL:
        print(f"Model config: {transformer_config} (PREFILL)")
        pytorch_model = LanguageModel(transformer_config)
    else:
        print(f"Model config: {transformer_config} (DECODE)")
        pytorch_model = LanguageModelDecode(transformer_config)

    match stage:
        case Stage.PREFILL:
            dummy_input = torch.randint(
                low=0, high=255, size=(transformer_config.batch_size, transformer_config.prefill_size)
            )
        case Stage.DECODE:
            dummy_input = torch.randint(low=0, high=255, size=(transformer_config.batch_size, 1))  # Single token

    for name, param in pytorch_model.named_parameters():
        param.data = torch.zeros_like(param.data)

    for name, buffer in pytorch_model.named_buffers():
        setattr(pytorch_model, name, torch.zeros_like(buffer))

    assert isinstance(path, str)
    torch.onnx.export(  # type: ignore
        pytorch_model,
        dummy_input,
        path,
        opset_version=16,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
        do_constant_folding=True,
        export_params=False,
    )
    # Remove Identity layers from the exported ONNX model
    remove_identity_layers(path)
def add_attribute_to_onnx_node(node: NodeProto, key: str, val: Any):
    attr = onnx.helper.make_attribute(key, val)
    node.attribute.extend([attr])


def remove_identity_layers(onnx_model_path: str):
    """
    Removes Identity layers from the ONNX model.
    """
    model = onnx.load(onnx_model_path)
    graph = model.graph

    # Collect nodes to remove and update the graph
    nodes_to_remove = []
    identity_outputs_map = {}

    for node in graph.node:
        if node.op_type == "Identity":
            nodes_to_remove.append(node)
            # Map the Identity output to its input
            identity_outputs_map[node.output[0]] = node.input[0]

    # Remove Identity nodes and update references
    for node in nodes_to_remove:
        graph.node.remove(node)

    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in identity_outputs_map:
                node.input[i] = identity_outputs_map[input_name]

    # Update graph outputs if they reference Identity nodes
    for i, output in enumerate(graph.output):
        if output.name in identity_outputs_map:
            graph.output[i].name = identity_outputs_map[output.name]

    # Save the optimized model
    onnx.save(model, onnx_model_path)


if __name__ == "__main__":
    config = OPT_2_7B
    config.batch_size = 1
    config.prefill_size = 2048
    quant_config = W32A32
    stage = Stage.DECODE
    # config = config.to_single_layer_config()

    path = get_onnx_path("outputs", config, stage, quant_config)
    export_model_to_onnx(
        config,
        quant_config,
        stage=stage,
        output_path=path,
    )


