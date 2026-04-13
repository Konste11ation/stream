from __future__ import annotations

import argparse
from pathlib import Path

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


def dump_workload_to_yaml(workload: ONNXWorkload, workload_path: str | Path):
    nodes_data = []
    for node in workload.node_list:
        node_data = {
            "id": getattr(node, 'id', None),
            "name": getattr(node, 'name', None),
            "operator_type": getattr(node, 'type', None),
            "equation": getattr(getattr(node, 'equation', None), 'data', None),
            "layer_dim_sizes": str(getattr(node, 'layer_dim_sizes', {})),
            "inter_core_tiling": str(getattr(node, 'inter_core_tiling', {})),
            "intra_core_tiling": str(getattr(node, 'intra_core_tiling', {})),
            "input_operand_source": str(getattr(node, 'input_operand_source', {}))
        }
        nodes_data.append(node_data)
    yaml_data = {"nodes": nodes_data}
    output_path = Path(workload_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            yaml_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            allow_unicode=True
        )


def main():
    parser = argparse.ArgumentParser(description="Parse LLM workload and generate YAML.")
    parser.add_argument("-w", "--workload-path", required=True, help="Path to the ONNX workload file.")
    parser.add_argument("-a", "--accelerator-yaml", required=True, help="Path to the accelerator YAML file.")
    parser.add_argument("-m", "--mapping-yaml", required=True, help="Path to the mapping YAML file.")
    parser.add_argument("-o", "--output-yaml", required=True, help="Path to save the output YAML file.")
    args = parser.parse_args()

    accelerator_data = open_yaml(args.accelerator_yaml)
    validator = AcceleratorValidator(accelerator_data, args.accelerator_yaml)
    accelerator_data = validator.normalized_data
    if not validator.validate():
        raise ValueError("Failed to validate user provided accelerator.")
    factory = AcceleratorFactory(accelerator_data)
    accelerator = factory.create()

    mapping_parser = MappingParser(args.mapping_yaml)
    all_mappings = mapping_parser.run()

    onnx_model_parser = ONNXModelParser(args.workload_path, all_mappings, accelerator)
    onnx_model_parser.run()
    dump_workload_to_yaml(onnx_model_parser.workload, args.output_yaml)


if __name__ == "__main__":
    main()
