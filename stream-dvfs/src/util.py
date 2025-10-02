import sys
import os
from pathlib import Path
import pickle
from enum import StrEnum
from typing import Any, TypeAlias

import numpy as np
from zigzag.cost_model.cost_model import CostModelEvaluation
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
sys.path.append(str(STREAM_DVFS_DIR))
from src.config import ModelConfig, QuantConfig, TransformerConfig, TransformerConfigSingleLayer

LAYERS_TO_PLOT = ["key_proj", "mul_qk_t", "mul_logits", "up_proj", "down_proj"]
GROUPS = ["Linear proj.", "Attention", "FFN"]

CME_T: TypeAlias = CostModelEvaluation
ARRAY_T: TypeAlias = np.ndarray[Any, Any]


class Stage(StrEnum):
    PREFILL = "Prefill"
    DECODE = "Decode"


BIT_T: TypeAlias = int | float
BYTE_T: TypeAlias = int | float


def generalize_layer_name(layer: str):
    """Give the layer name a prettier format, and generalize single layers to full LLM. e.g. key projection -> all
    linear projections"""
    if "key_proj" in layer:
        return "linear projection"
    elif "mul_qk_t" in layer:
        return "mul K*Q^T"
    elif "mul_logits" in layer:
        return "mul attn*V"
    elif "up_proj" in layer:
        return "MLP layer 1"
    elif "down_proj" in layer:
        return "MLP layer 2"
    else:
        return layer


def get_cmes_to_plot_strict_format(cmes: list[CME_T]):
    """Return CMEs in order of `LAYERS_TO_PLOT"""
    result: list[CME_T] = []
    for name in LAYERS_TO_PLOT:
        cme = next(filter(lambda x: name in x.layer.name, cmes), None)
        if cme is not None:
            result.append(cme)
    if len(result) != len(LAYERS_TO_PLOT):
        raise ValueError("Some layers are missing")
    return result


def get_cmes_full_model(
    cmes_trivial: list[CME_T],
    model: TransformerConfig,
    stage: Stage = Stage.PREFILL,
) -> list[CME_T]:
    """Generalize the zigzag results (from a `minimal` configuration, i.e. single layer and head) to a full LLM"""

    def get_multiplier(name: str):
        return model.to_single_layer_config().get_post_simulation_multiplier(name)

    number_of_runs = 1 if stage == Stage.PREFILL else model.decode_size
    return [cme * get_multiplier(cme.layer.name) * number_of_runs for cme in cmes_trivial]  # type: ignore


def get_cmes_full_model_from_pickle(
    pickle_file: str,
    model: TransformerConfigSingleLayer,
    stage: Stage,
) -> list[CME_T]:
    with open(pickle_file, "rb") as fp:
        cmes: list[CostModelEvaluation] = pickle.load(fp)

    cmes_filtered = get_cmes_to_plot_strict_format(cmes)
    cmes_generalized = get_cmes_full_model(cmes_filtered, model, stage)
    return cmes_generalized


def get_experiment_id(model: ModelConfig, stage: Stage, quant: QuantConfig, accelerator_name: str):
    """Generate the name of the experiment"""
    assert "yaml" not in accelerator_name and "/" not in accelerator_name
    return f"{model.parameterized_name}_prefill={model.prefill_size}_decode={model.decode_size}_{quant.name}_{stage}_{accelerator_name}"


def get_onnx_path(output_dir: str, model: ModelConfig, stage: Stage, quant: QuantConfig):
    name = f"{model.parameterized_name}_PREFILL_SIZE={model.prefill_size}_DECODE_SIZE={model.decode_size}_{quant.name}_{stage}.onnx"
    return f"{output_dir}/{name}"


def get_accelerator_path(accelerator_name: str):
    DEFAULT_DIR = "inputs/single_core_system"
    assert not os.path.splitext(accelerator_name)[1]  # Gives the file extension or False if no extension
    assert not os.path.dirname(os.path.normpath(accelerator_name))
    return f"{DEFAULT_DIR}/{accelerator_name}.yaml"


def get_accelerator_name_and_path(accelerator_name_or_path: str):
    """Given either the full path of an accelerator yaml file, or just the name of an accelerator saved in the default
    path, return both the name (without path or extension) and the full path."""
    path_and_name, extension = os.path.splitext(accelerator_name_or_path)
    # This is a path
    if extension == ".yaml":
        name = os.path.basename(path_and_name)
        return name, accelerator_name_or_path
    # This is just a name
    elif not extension:
        full_path = get_accelerator_path(accelerator_name_or_path)
        return accelerator_name_or_path, full_path
    else:
        raise ValueError("Argument is not a name or a yaml file")
    
    
