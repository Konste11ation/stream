import logging as _logging
import re
import os
from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json
from stream.visualization.schedule import (
    visualize_timeline_plotly,
)

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

############################################INPUTS############################################
hardware = "tpu_like_quad_core"
mapping = "tpu_like_quad_core"
workload = "resnet50"
dvfs = "fine_dvfs"
hardware_dir = "stream/inputs/examples/hardware"
mapping_dir = "stream/inputs/examples/mapping"
workload_dir = "stream/inputs/examples/workload"
mode = "fused"
output_dir = "outputs"

experiment_id = f"{hardware}-{workload}-{mode}-{dvfs}-genetic_algorithm"
dvfs_output_dir = os.path.join(output_dir, f"{experiment_id}/dvfs")
scme_json_path = os.path.join(output_dir, f"{experiment_id}/scme.json")
scme_dvfs_json_path = os.path.join(output_dir, f"{experiment_id}/scme_dvfs.json")

hardware_path = os.path.join(hardware_dir, f"{hardware}.yaml")
mapping_path = os.path.join(mapping_dir, f"{mapping}.yaml")
workload_path = os.path.join(workload_dir, f"{workload}.onnx")
dvfs_cfg_path = os.path.join(hardware_dir, f"{dvfs}.yaml")

layer_stacks = None
nb_ga_generations = 8
nb_ga_individuals = 8
nb_ga_generations_dvfs = 8
nb_ga_individuals_dvfs = 8
dvfs_opt = True
skip_if_exist = True

#####################################################################

(scme, scme_dvfs) = optimize_allocation_ga(
            hardware=hardware_path,
            workload=workload_path,
            mapping=mapping_path,
            mode=mode,
            layer_stacks=layer_stacks,
            nb_ga_generations=nb_ga_generations,
            nb_ga_individuals=nb_ga_individuals,
            experiment_id=experiment_id,
            output_path=output_dir,
            dvfs_output_dir=dvfs_output_dir,
            dvfs_cfg_path=dvfs_cfg_path,
            nb_ga_generations_dvfs=nb_ga_generations_dvfs,
            nb_ga_individuals_dvfs=nb_ga_individuals_dvfs,
            dvfs_opt=dvfs_opt,
            skip_if_exists=skip_if_exist
        )

# Load in the CostModelEvaluationLUT from the run
cost_lut_path = f"outputs/{experiment_id}/cost_lut.pickle"
cost_lut = CostModelEvaluationLUT(cost_lut_path)

# Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
convert_scme_to_perfetto_json(scme, cost_lut, json_path=scme_json_path)
convert_scme_to_perfetto_json(scme_dvfs, cost_lut, json_path=scme_dvfs_json_path)
