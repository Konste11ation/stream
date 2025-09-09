import sys
import os
import logging as _logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
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


def process_experiment_ga(args):
    """
    Function to handle single thread experiment
    """
    (hardware, mapping, workload, hardware_dir, mapping_dir, workload_dir,
     output_dir, mode, nb_ga_generations_alloc, nb_ga_individuals_alloc, skip_if_exist,
    dvfs_cfg_path, nb_ga_generations_dvfs, nb_ga_individuals_dvfs, dvfs_opt) = args

    experiment_id = f"{hardware}-{workload}-{mapping}-genetic_algorithm"
    scme_json_path = os.path.join(output_dir, f"{experiment_id}/scme.json")
    scme_dvfs_json_path = os.path.join(output_dir, f"{experiment_id}/scme_dvfs.json")
    hardware_path = os.path.join(hardware_dir, f"{hardware}.yaml")
    mapping_path = os.path.join(mapping_dir, f"{mapping}.yaml")
    workload_path = os.path.join(workload_dir, f"{workload}.onnx")
    dvfs_output_dir = os.path.join(output_dir, f"{experiment_id}/dvfs")
    print(f"Processing: {experiment_id}")
    print(f"Hardware: {hardware_path}")
    print(f"Workload: {workload_path}")
    print(f"Mapping: {mapping_path}")
    try:
        # Main function
        layer_stacks = get_layer_stack(workload)  
        (scme, scme_dvfs) = optimize_allocation_ga(
            hardware=hardware_path,
            workload=workload_path,
            mapping=mapping_path,
            mode=mode,
            layer_stacks=layer_stacks,
            nb_ga_generations=nb_ga_generations_alloc,
            nb_ga_individuals=nb_ga_individuals_alloc,
            experiment_id=experiment_id,
            output_path=output_dir,
            dvfs_output_dir=dvfs_output_dir,
            dvfs_cfg_path=dvfs_cfg_path,
            nb_ga_generations_dvfs=nb_ga_generations_dvfs,
            nb_ga_individuals_dvfs=nb_ga_individuals_dvfs,
            dvfs_opt=dvfs_opt,
            skip_if_exists=skip_if_exist
        )

        cost_lut_path = os.path.join(output_dir, f"{experiment_id}/cost_lut.pickle")
        cost_lut = CostModelEvaluationLUT(cost_lut_path)
        convert_scme_to_perfetto_json(scme, cost_lut, json_path=scme_json_path)
        convert_scme_to_perfetto_json(scme_dvfs, cost_lut, json_path=scme_dvfs_json_path)
    except Exception as e:
        print(f"Error processing {experiment_id}: {str(e)}")


def main():
    num_threads = 4
    skip_if_exist = True
    dvfs_opt = True
    output_dir = "test_env/outputs"
    dvfs_cfg_path = "test_env/inputs/hardware/dvfs/fine_dvfs.yaml"
    workload_dir = "test_env/inputs/workload"
    mapping_dir = "test_env/inputs/mapping"
    hardware_dir = "test_env/inputs/hardware"
    
    # hardware_list = ["tpu_like_quad_core"]

    # hardware_list = ["tpu_like_quad_core", "mixed_ascend_edgetpu_tpu_metaproto"]
    hardware_list = ["ascend_quad_core", "edge_tpu_quad_core", "meta_prototype_quad_core", "tpu_like_quad_core", "mixed_ascend_edgetpu_tpu_metaproto"]
    mapping_list = ["quad_core_OY1",
                    "quad_core_OY4",
                    "quad_core_OY16"]
    # mapping_list = ["quad_core_OY1",
    #                 "quad_core_OY4",
    #                 "quad_core_OY16",
    #                 "quad_core_OY64"]
    # mapping_list = ["quad_core_finer"]
    # mapping_list = ["quad_core", "quad_core_finer"]
    workload_list = ["mobilebert", "tinyyolov2", "xception"]

    # workload_list = ["resnet18",
    #                  "mobilenetv2",
    #                  "fsrcnn",
    #                  "squeezenet",
    #                  "inception_v2",
    #                  "mobilebert",
    #                  "tinyyolov2",
    #                  "xception"] 
    # Generate all the args
    mode = "fused"
    nb_ga_generations_alloc = 4
    nb_ga_individuals_alloc = 4
    nb_ga_generations_dvfs = 8
    nb_ga_individuals_dvfs = 16
    experiment_args = [
        (h, m, w, hardware_dir, mapping_dir, workload_dir, 
         output_dir, mode, nb_ga_generations_alloc, nb_ga_individuals_alloc, skip_if_exist,
         dvfs_cfg_path, nb_ga_generations_dvfs, nb_ga_individuals_dvfs, dvfs_opt)
        for h, m, w in product(hardware_list, mapping_list, workload_list)
    ]
    # # for exp_arg in experiment_args:
    #     # process_experiment_ga(exp_arg)
    # process_experiment_ga(experiment_args[0])


    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_experiment_ga, args) for args in experiment_args]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Thread error: {str(e)}")


if __name__ == "__main__":
    main()