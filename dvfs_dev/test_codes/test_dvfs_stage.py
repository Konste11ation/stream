import sys
import os
import pickle
import copy
import logging as _logging
sys.path.append(os.getcwd())
from stream.stages.optimization.dvfs_optimization import DvfsOptimizationStage
from stream.visualization.perfetto import convert_scme_to_perfetto_json
from stream.utils import CostModelEvaluationLUT
from stream.stages.stage import MainStage

stream_experiment = "tpu_like_quad_core-resnet50-lbl-fine_dvfs-genetic_algorithm"
cost_lut_path = f"outputs/{stream_experiment}/cost_lut.pickle"


saved_vars_path = f'outputs/{stream_experiment}/scme.pickle'
print("Saved Variables Path:", saved_vars_path)
with open(saved_vars_path, "rb") as file:
    scme_original = pickle.load(file)

dvfs_cfg_path = "dvfs_dev/inputs/fine_dvfs_1ms.yaml"
dvfs_output_dir = "dvfs_dev/outputs"
dvfs_json_path = "dvfs_dev/outputs/dvfs_scme.json"
nb_ga_generations_dvfs = 4
nb_ga_individuals_dvfs = 16



dvfs_opt_stage = MainStage(
    [DvfsOptimizationStage],
    scme=scme_original,
    dvfs_output_dir=dvfs_output_dir,
    dvfs_cfg_path=dvfs_cfg_path,
    nb_ga_generations_dvfs=nb_ga_generations_dvfs,
    nb_ga_individuals_dvfs=nb_ga_individuals_dvfs,
)

answers = dvfs_opt_stage.run()
scme = answers[0][0]
scmes_dvfs_opt = answers[0][1]

cost_lut = CostModelEvaluationLUT(cost_lut_path)
for i, scme_dvfs_opt in enumerate(scmes_dvfs_opt):
    dvfs_scme_json_path = f"dvfs_dev/outputs/dvfs_scme_{i}.json"
    convert_scme_to_perfetto_json(scme_dvfs_opt, cost_lut, json_path=dvfs_scme_json_path)