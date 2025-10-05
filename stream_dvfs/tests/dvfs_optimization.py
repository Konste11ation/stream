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

saved_vars_path = 'stream_dvfs/outputs/attention_head-AttentionHeadTest_B=1_FULL_PREFILL_SIZE=1_DECODE_SIZE=1_W8A8_Decode-fused-ga/scme.pickle'
saved_var_dir = os.path.dirname(saved_vars_path)
print("Saved Variables Path:", saved_vars_path)
with open(saved_vars_path, "rb") as file:
    scme_original = pickle.load(file)
    
workload = scme_original.workload
accelerator = scme_original.accelerator
scheduling_order = scme_original.scheduling_order
dvfs_output_path = f"{saved_var_dir}/pareto.png"
ga_nb_generations = 5
ga_nb_individuals = 10
dvfs_opt_stage = DvfsOptimizationStage(list_of_callables=[],
                                       workload=workload,
                                       accelerator=accelerator,
                                       scheduling_order=scheduling_order,
                                       dvfs_output_path=dvfs_output_path,
                                       ga_nb_generations=ga_nb_generations,
                                       ga_nb_individuals=ga_nb_individuals
                                       )
dvfs_opt_scme=dvfs_opt_stage.run()