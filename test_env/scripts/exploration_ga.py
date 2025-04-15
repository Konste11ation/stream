import sys
import os
import logging as _logging
import re
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.getcwd())
from stream.api import optimize_allocation_ga
from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json
from stream.visualization.schedule import (
    visualize_timeline_plotly,
)
from test_env.scripts.util import get_layer_stack
_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

##########
# INPUTS #
##########
output_dir = "test_env/outputs"
workload_dir = "test_env/inputs/workload"
mapping_dir = "test_env/inputs/mapping"
hardware_dir = "test_env/inputs/hardware"
hardware_list = ["meta_prototype_quad_core"]
mapping_list = ["quad_core", "quad_core_finer"]
workload_list = ["resnet18",
                 "mobilenetv2",
                 "fsrcnn",
                 "squeezenet",
                 "inception_v2",
                 "attention_head"]
# Common Settings
mode = "fused"
nb_ga_generations = 16
nb_ga_individuals = 16
section_start_percent = (0,)
percent_shown = (100,)

for hardware in hardware_list:
    for mapping in mapping_list:
        for workload in workload_list:
            hardware_path = os.path.join(hardware_dir, f"{hardware}.onnx")
            mapping_path = os.path.join(mapping_dir, f"{mapping}.yaml")
            workload_path = os.path.join(workload_dir, f"{workload}.onnx")
            experiment_id = f"{hardware}-{workload}-{mapping}-genetic_algorithm"
            json_path = os.path.join(output_dir, f"{experiment_id}/scme.json")
            layer_stacks = get_layer_stack(workload)
            print(f"Processing Experiment {experiment_id}")
            scme = optimize_allocation_ga(
                hardware=hardware_path,
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
            # Load in the CostModelEvaluationLUT from the run
            cost_lut_path = os.path.join(output_dir, f"{experiment_id}/cost_lut.pickle")
            cost_lut = CostModelEvaluationLUT(cost_lut_path)
            convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)