import sys
import os
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
sys.path.append(str(STREAM_WORKDIR))
import yaml
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.accelerator_factory import AcceleratorFactory

from stream.parser.mapping_parser import MappingParser
from stream.parser.onnx.model import ONNXModelParser
from zigzag.utils import open_yaml

from stream.workload.onnx_workload import ONNXWorkload
import logging as _logging
_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


def dump_workload_to_yaml(workload: ONNXWorkload, workload_path:str):
    nodes_data = []
    for node in workload.node_list:
        node_data = {
            "id": getattr(node, 'id', None),  
            "name": getattr(node, 'name', None),
            "operator_type": getattr(node, 'type', None),
            "equation": getattr(getattr(node, 'equation', None),'data',None),
            "layer_dim_sizes": str(getattr(node, 'layer_dim_sizes', {})),
            "inter_core_tiling": str(getattr(node, 'inter_core_tiling', {})),
            "intra_core_tiling": str(getattr(node, 'intra_core_tiling', {})),
            "input_operand_source": str(getattr(node, 'input_operand_source', {}))
        }
        nodes_data.append(node_data)
    yaml_data = {"nodes": nodes_data}
    os.makedirs(os.path.dirname(workload_path), exist_ok=True)
    with open(workload_path, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            default_flow_style=False,  
            sort_keys=False,
            indent=2,
            allow_unicode=True
        )


accelerator_yaml = "stream-dvfs/inputs/multicore_system/attention_head.yaml"
mapping_yaml = "stream-dvfs/inputs/multicore_mapping/attention_head.yaml"
workload_path = "stream-dvfs/inputs/workloads/AttentionHeadTest_B=1_FULL_PREFILL_SIZE=1_DECODE_SIZE=1_W8A8_Decode.onnx"

# Parse accelerator
accelerator_data = open_yaml(accelerator_yaml)
validator = AcceleratorValidator(accelerator_data, accelerator_yaml)
accelerator_data = validator.normalized_data
validate_success = validator.validate()
if not validate_success:
    raise ValueError("Failed to validate user provided accelerator.")
factory = AcceleratorFactory(accelerator_data)

accelerator = factory.create()

# Parse mapping
mapping_parser = MappingParser(mapping_yaml)
all_mappings = mapping_parser.run()

# Parse ONNX model
onnx_model_parser = ONNXModelParser(workload_path, all_mappings, accelerator)
onnx_model_parser.run()
onnx_model = onnx_model_parser.onnx_model
workload = onnx_model_parser.workload
dump_workload_to_yaml(workload, "stream-dvfs/inputs/workloads/attention_head.yaml")