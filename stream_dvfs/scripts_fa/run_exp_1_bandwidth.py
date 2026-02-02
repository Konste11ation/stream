
import os
import sys
import yaml
import csv
import logging
from pathlib import Path

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent.parent))

from stream_dvfs.scripts_fa.stream_dvfs_fa import run_stream_fa, gen_flash_attention_onnx, gen_flash_attention_mapping_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_noc_bandwidth(bandwidth: int, template_path: str, output_path: str):
    """Updates the NoC bandwidth in the hardware config file."""
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update all links bandwidth
    for link in config['core_connectivity']:
        link['bandwidth'] = bandwidth
        
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Updated Hardware Config with Bandwidth = {bandwidth}")

def run_exp_bandwidth():
    # Define Experiment Root Directory
    exp_root_dir = str(CURRENT_DIR / "outputs_exp1_bandwidth")
    if not os.path.exists(exp_root_dir):
        os.makedirs(exp_root_dir)
        
    output_csv = os.path.join(exp_root_dir, "results_exp1_bandwidth.csv")
    
    # Initialize CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "SeqLen", "TileSize", "NoC_BW", "Mode", 
                         "Baseline_Energy", "Baseline_Latency", "Baseline_EDP",
                         "Best_Global_Energy_Norm", "Best_Global_Latency_Norm", "Best_Global_EDP",
                         "GA_Energy", "GA_Latency", "GA_EDP", 
                         "Improvement_Baseline_%", "Improvement_Global_%"])

    # Global Hardware Settings
    num_cores = 2
    embedding_dim = 1024
    
    # Experiment Settings
    seq_len = 1024
    tile_size = 128   
    bandwidths = [32, 64, 512] # bits/cycle
    
    hw_template = str(CURRENT_DIR / "inputs" / "multicores" / f"FA_{num_cores}gemm.yaml")
    
    print("\n--- Starting Experiment 1: Bandwidth Wall ---")
    for bw in bandwidths:
        # Update Hardware Config
        update_noc_bandwidth(bw, hw_template, hw_template) 
        
        # Prepare Inputs
        gen_flash_attention_onnx(seq_len, embedding_dim, tile_size, output_dir=str(CURRENT_DIR / "inputs" / "workloads"), include_linear_layers=False)
        gen_flash_attention_mapping_config(num_qkv_tiles=seq_len//tile_size, num_cores=num_cores)
        

        experiment_out_dir = os.path.join(exp_root_dir, f"bw_{bw}")
        if not os.path.exists(experiment_out_dir):
            os.makedirs(experiment_out_dir)
        
        scme = run_stream_fa(seq_len, embedding_dim, tile_size, num_cores, experiment_out_dir)
        
        # Extract Metrics
        base_E = scme.baseline_energy
        base_L = scme.baseline_latency
        base_EDP = 1.0 # Normalized
        
        ga_E_norm = scme.energy / base_E
        ga_L_norm = scme.latency / base_L
        ga_EDP = ga_E_norm * ga_L_norm
        
        global_E_norm = scme.best_global_energy_norm
        global_L_norm = scme.best_global_latency_norm
        global_EDP = global_E_norm * global_L_norm
        
        imp_base = (1.0 - ga_EDP) * 100
        imp_global = (global_EDP - ga_EDP) / global_EDP * 100
        
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Bandwidth_Sweep", seq_len, tile_size, bw, "GA",
                                base_E, base_L, base_EDP,
                                global_E_norm, global_L_norm, global_EDP,
                                scme.energy, scme.latency, ga_EDP,
                                imp_base, imp_global])
                                 


    # Restore BW (Default 64)
    update_noc_bandwidth(64, hw_template, hw_template)
    logger.info("Experiment 1 Complete. Results saved to results_exp1_bandwidth.csv")

if __name__ == "__main__":
    run_exp_bandwidth()
