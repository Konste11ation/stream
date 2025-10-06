import sys
import os
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
STREAM_DEV_DIR = STREAM_WORKDIR.parent
sys.path.append(str(STREAM_WORKDIR))

os.environ['GRB_LICENSE_FILE'] = f'{STREAM_DEV_DIR}/gurobi.lic'
import logging as _logging
from stream.api import optimize_allocation_co, optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.perfetto import convert_scme_to_perfetto_json
import re
_logging_level = _logging.DEBUG
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

############################################INPUTS############################################
output_dir = "stream_dvfs/outputs"

workload_path = "stream_dvfs/inputs/workloads/AttentionHeadTest_B=1_FULL_PREFILL_SIZE=1_DECODE_SIZE=1_W8A8_Decode.onnx"
accelerator = "stream_dvfs/inputs/multicore_system/attention_head.yaml"
mapping_path = "stream_dvfs/inputs/multicore_mapping/attention_head.yaml"

# mode = "lbl"
# layer_stacks = []

mode = "fused"
layer_stacks = [tuple(range(0, 11))]

hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
    
experiment_id = f"{hw_name}-{wl_name}-{mode}-co"
scme = optimize_allocation_co(
    hardware=accelerator,
    workload=workload_path,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=layer_stacks,
    experiment_id=experiment_id,
    output_path=output_dir,
    skip_if_exists=False,
)
# experiment_id = f"{hw_name}-{wl_name}-{mode}-ga"
# nb_ga_generations = 8
# nb_ga_individuals = 8
# scme = optimize_allocation_ga(
#     hardware=accelerator,
#     workload=workload_path,
#     mapping=mapping_path,
#     mode=mode,
#     layer_stacks=layer_stacks,
#     nb_ga_generations=nb_ga_generations,
#     nb_ga_individuals=nb_ga_individuals,
#     experiment_id=experiment_id,
#     output_path=output_dir,
#     skip_if_exists=False,
# )

# Load in the CostModelEvaluationLUT from the run
cost_lut_path = f"{output_dir}/{experiment_id}/cost_lut.pickle"
cost_lut = CostModelEvaluationLUT(cost_lut_path)
json_path = f"{output_dir}/{experiment_id}/scme.json"
# Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)