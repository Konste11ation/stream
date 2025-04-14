import sys
import os
import pickle
import copy
import logging as _logging
sys.path.append(os.getcwd())
from stream.stages.optimization.dvfs_optimization import DvfsOptimizationStage
from stream.visualization.perfetto import convert_scme_to_perfetto_json
from stream.utils import CostModelEvaluationLUT
saved_vars_path = 'outputs/tpu_like_quad_core-resnet18-fine_dvfs-fused-constraint_optimization-test/scme.pickle'
print("Saved Variables Path:", saved_vars_path)
with open(saved_vars_path, "rb") as file:
    scme_original = pickle.load(file)

workload = scme_original.workload
accelerator = scme_original.accelerator
scheduling_order = scme_original.scheduling_order
dvfs_cfg_path = "stream/inputs/examples/hardware/fine_dvfs.yaml"
dvfs_output_path = "dvfs_dev/outputs/pareto.png"
dvfs_json_path = "dvfs_dev/outputs/dvfs_scme.json"
ga_nb_generations = 5
ga_nb_individuals = 10
dvfs_opt_stage = DvfsOptimizationStage(list_of_callables=[],
                                       workload=workload,
                                       accelerator=accelerator,
                                       scheduling_order=scheduling_order,
                                       dvfs_cfg_path=dvfs_cfg_path,
                                       dvfs_output_path=dvfs_output_path,
                                       ga_nb_generations=ga_nb_generations,
                                       ga_nb_individuals=ga_nb_individuals
                                       )
dvfs_opt_scme=dvfs_opt_stage.run()

cost_lut_path = "outputs/tpu_like_quad_core-resnet18-fine_dvfs-fused-constraint_optimization-test/cost_lut_post_co.pickle"
cost_lut = CostModelEvaluationLUT(cost_lut_path)
convert_scme_to_perfetto_json(dvfs_opt_scme, cost_lut, json_path=dvfs_json_path)