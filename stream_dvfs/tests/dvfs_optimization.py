import sys
import os
from pathlib import Path
import pickle
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
STREAM_DEV_DIR = STREAM_WORKDIR.parent
sys.path.append(str(STREAM_WORKDIR))

from stream_dvfs.src.dvfs_optimization import DvfsOptimizationStage
from stream.utils import CostModelEvaluationLUT
from stream.visualization.perfetto import convert_scme_to_perfetto_json
output_dir = 'stream_dvfs/outputs/exp_sweep/Llama2-7B-W4A8-Decode'
base_scme_path = f'{output_dir}/scme.pickle'
print("Base scme Path:", base_scme_path)
with open(base_scme_path, "rb") as file:
    scme_original = pickle.load(file)

dvfs_cfg_path = 'stream_dvfs/inputs/dvfs/fine_dvfs.yaml'
workload = scme_original.workload
accelerator = scme_original.accelerator
scheduling_order = scme_original.scheduling_order
operands_to_prefetch = scme_original.operands_to_prefetch
dvfs_output_path = output_dir
ga_nb_generations = 16
ga_nb_individuals = 16
dvfs_opt_stage = DvfsOptimizationStage(list_of_callables=[],
                                       workload=workload,
                                       accelerator=accelerator,
                                       scheduling_order=scheduling_order,
                                       operands_to_prefetch=operands_to_prefetch,
                                       dvfs_output_path=dvfs_output_path,
                                       ga_nb_generations=ga_nb_generations,
                                       ga_nb_individuals=ga_nb_individuals,
                                       dvfs_cfg_path=dvfs_cfg_path
                                       )
dvfs_opt_scme=dvfs_opt_stage.run()

# Load in the CostModelEvaluationLUT from the run
cost_lut_path = f"{output_dir}/cost_lut.pickle"
cost_lut = CostModelEvaluationLUT(cost_lut_path)
json_path = f"{output_dir}/dvfs_scme.json"
# Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
convert_scme_to_perfetto_json(dvfs_opt_scme, cost_lut, json_path=json_path)