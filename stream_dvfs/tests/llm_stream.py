import sys
import os
import argparse
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
sys.path.append(str(STREAM_WORKDIR))

import logging as _logging
from stream.api import optimize_allocation_co, optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.perfetto import convert_scme_to_perfetto_json
import re
_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

############################################INPUTS############################################
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run Stream LLM workload.")
    parser.add_argument("-w", "--workload_path", required=True, help="Path to the ONNX workload file.")
    parser.add_argument("-a", "--accelerator_yaml", required=True, help="Path to the accelerator YAML file.")
    parser.add_argument("-m", "--mapping_yaml", required=True, help="Path to the mapping YAML file.")
    parser.add_argument("-o", "--output_dir", required=True, help="Dir to save outputs.")
    args = parser.parse_args()  
    
    workload_path = args.workload_path
    accelerator = args.accelerator_yaml
    mapping_path = args.mapping_yaml
    output_dir = args.output_dir
    # mode = "lbl"
    # layer_stacks = []

    mode = "fused"
    layer_stacks = [tuple(range(0, 67))]

    hw_name = accelerator.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
        
    experiment_id = f"{hw_name}-{wl_name}-{mode}-ga"
    nb_ga_generations = 8
    nb_ga_individuals = 8
    scme = optimize_allocation_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=nb_ga_generations,
        nb_ga_individuals=nb_ga_individuals,
        experiment_id=experiment_id,
        output_path=output_dir,
        skip_if_exists=False,
    )

    # Load in the CostModelEvaluationLUT from the run
    cost_lut_path = f"{output_dir}/{experiment_id}/cost_lut.pickle"
    cost_lut = CostModelEvaluationLUT(cost_lut_path)
    json_path = f"{output_dir}/{experiment_id}/scme.json"
    # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)

if __name__ == "__main__":
    main()