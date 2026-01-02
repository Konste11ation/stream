import os
import sys
# Resolve paths early
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
STREAM_DEV_DIR = STREAM_WORKDIR.parent
sys.path.append(str(STREAM_WORKDIR))
# onnx generation imports
from src.util import Stage, get_onnx_path  # noqa: E402 
from src.export_onnx import export_model_to_onnx  # noqa: E402 

# sanity check imports
import yaml
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.accelerator_factory import AcceleratorFactory

from stream.parser.mapping_parser import MappingParser
from stream.parser.onnx.model import ONNXModelParser
from zigzag.utils import open_yaml

# stream simulation imports
from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.perfetto import convert_scme_to_perfetto_json

# dvfs optimiztion imports
import pickle
from stream_dvfs.src.dvfs_optimization import DvfsOptimizationStage

def sanity_check(workload_path, accelerator_path, mapping_path, output_yaml_path):
    # Sanity Check here
    # Running the parsing of the workload, acc, mapping and dump the workload info to yaml output
    
    # 1. Parse accelerator
    accelerator_data = open_yaml(accelerator_path)
    validator = AcceleratorValidator(accelerator_data, accelerator_path)
    accelerator_data = validator.normalized_data
    validate_success = validator.validate()
    if not validate_success:
        raise ValueError("Failed to validate user provided accelerator.")
    factory = AcceleratorFactory(accelerator_data)
    accelerator = factory.create()
    
    # 2. Parse mapping
    mapping_parser = MappingParser(mapping_path)
    all_mappings = mapping_parser.run()

    # 3. Parse ONNX model
    onnx_model_parser = ONNXModelParser(workload_path, all_mappings, accelerator)
    onnx_model_parser.run()
    workload = onnx_model_parser.workload
    
    # 4. Dump workload to yaml
    nodes_data = []
    for node in workload.node_list:
        node_data = {
            "id": getattr(node, 'id', None),  
            "name": getattr(node, 'name', None),
            "operator_type": getattr(node, 'type', None),
            "equation": getattr(getattr(node, 'equation', None),'data',None),
            "layer_dim_sizes": str(getattr(node, 'layer_dim_sizes', {})),
            "inter_core_tiling": str(getattr(node, 'inter_core_tiling', {})),
            "intra_core_tiling": str(getattr(node, 'intra_core_tiling', {})),
            "input_operand_source": str(getattr(node, 'input_operand_source', {}))
        }
        nodes_data.append(node_data)
    yaml_data = {"nodes": nodes_data}

    with open(output_yaml_path, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            default_flow_style=False,  
            sort_keys=False,
            indent=2,
            allow_unicode=True
        )    

def run_stream(model, quant, stage, accelerator_path, mapping_path, output_dir, skip_if_exists=True):
    experiment_id=f"{model}-{quant.name}-{stage}"
    exp_output_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(exp_output_dir, exist_ok=True)
    if os.path.exists(os.path.join(exp_output_dir, "scme.json")) and skip_if_exists:
        print(f"Experiment {experiment_id} already exists. Skipping...")
        return
    print(f"Running STREAM DVFS simulation for model: {model}, quant: {quant.name}, stage: {stage}")
    print("Steps: 1. Generate ONNX model")
    # 1. Generate the ONNX model
    onnx_path = get_onnx_path(
        output_dir=exp_output_dir,
        model=model,
        stage=stage,
        quant=quant,
    )  
    export_model_to_onnx(
        model_config=model,
        quant_config=quant,
        output_path=onnx_path,
        stage=stage,
    )
    print(f"ONNX model generated at: {onnx_path}")
    print("Steps: 2. Sanity check ONNX model and mappings")
    # 2. Sanity check onnx models and mappings
    sanity_check(
        workload_path=onnx_path,
        accelerator_path=accelerator_path,
        mapping_path=mapping_path,
        output_yaml_path=os.path.join(exp_output_dir, "workload_mapping.yaml")
    )
    print("Sanity check completed successfully.")
    print("Steps: 3. Run STREAM DVFS simulation")
    # 3. Run the Stream Simulation

    cost_lut_path = os.path.join(exp_output_dir,"cost_lut.pickle")
    output_json_path = os.path.join(exp_output_dir,"scme.json")

    scme = optimize_allocation_ga(
        hardware=accelerator_path,
        workload=onnx_path,
        mapping=mapping_path,
        mode="fused",
        layer_stacks=[tuple(range(0, 67))],
        nb_ga_generations=8,
        nb_ga_individuals=8,
        experiment_id=experiment_id,
        output_path=output_dir,
        skip_if_exists=True,
    )
    # Load in the CostModelEvaluationLUT from the run
    cost_lut = CostModelEvaluationLUT(cost_lut_path)
    # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=output_json_path)

def run_dvfs_optimization(model, quant, stage, dvfs_cfg, output_dir, skip_if_exists=True):
    experiment_id=f"{model}-{quant.name}-{stage}"
    exp_output_dir = os.path.join(output_dir, experiment_id)
    if os.path.exists(os.path.join(exp_output_dir, "dvfs_scme.json")) and skip_if_exists:
        print(f"Experiment {experiment_id} already exists. Skipping...")
        return
    print("Steps: 1. Load base SCME from STREAM simulation")
    base_scme_path = os.path.join(output_dir, experiment_id, "scme.pickle")
    print(f"Base scme Path:", base_scme_path)
    with open(base_scme_path, "rb") as file:
        scme_original = pickle.load(file)
    print("Steps: 2. Run DVFS Optimization")
    ga_nb_generations = 64
    ga_nb_individuals = 64
    dvfs_opt_stage = DvfsOptimizationStage(list_of_callables=[],
                                        workload=scme_original.workload,
                                        accelerator=scme_original.accelerator,
                                        scheduling_order=scme_original.scheduling_order,
                                        operands_to_prefetch=scme_original.operands_to_prefetch,
                                        dvfs_output_path=exp_output_dir,
                                        ga_nb_generations=ga_nb_generations,
                                        ga_nb_individuals=ga_nb_individuals,
                                        dvfs_cfg_path=dvfs_cfg
                                        )
    dvfs_opt_scme=dvfs_opt_stage.run()
    # Load in the CostModelEvaluationLUT from the run
    cost_lut_path = os.path.join(output_dir, experiment_id,"cost_lut.pickle")
    cost_lut = CostModelEvaluationLUT(cost_lut_path)
    json_path = os.path.join(output_dir, experiment_id,"dvfs_scme.json")
    convert_scme_to_perfetto_json(dvfs_opt_scme, cost_lut, json_path=json_path)