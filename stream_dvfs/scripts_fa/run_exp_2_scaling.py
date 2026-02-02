
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

def run_exp_scaling():
    experiment_out_dir = str(CURRENT_DIR / "outputs_exp2_scaling_unaligned")
    if not os.path.exists(experiment_out_dir):
        os.makedirs(experiment_out_dir)

    output_csv = os.path.join(experiment_out_dir, "results_exp2_scaling_unaligned.csv")
    
    # Initialize CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "NumCores", "SeqLen", "TileSize", "NoC_BW", "Mode", 
                         "Baseline_Energy", "Baseline_Latency", "Baseline_EDP",
                         "Best_Global_Energy_Norm", "Best_Global_Latency_Norm", "Best_Global_EDP", "Global_AUC",
                         "GA_Energy", "GA_Latency", "GA_EDP", "GA_AUC",
                         "Improvement_Baseline_EDP_%", "Improvement_Global_EDP_%", "Improvement_Global_AUC_%",
                         "Avg_Latency_Reduction", "Global_Lat_50_Energy", "GA_Lat_50_Energy"])

    # Global Hardware Settings
    core_configs = [8,4] # Sweep Cores
    embedding_dim = 512
    tile_size = 128  # Fixed Tile Size
    
    # Ensure Bandwidth is Default
    # NOTE: Since we change num_cores, we need to handle the yaml path dynamically inside the loop
    seq_lens = range(128, 1024 + 64, 64)
    
    print("\n--- Starting Experiment 2: Unaligned Workload Scaling ---")
    
    for num_cores in core_configs:
        # Update/Reset HW Config for this core count
        hw_template = str(CURRENT_DIR / "inputs" / "multicores" / f"FA_{num_cores}gemm.yaml")
        # Ensure the file exists or is generated - assuming standard files exist for 2/4 cores
        if os.path.exists(hw_template):
            update_noc_bandwidth(64, hw_template, hw_template)
        else:
            logger.error(f"Hardware config {hw_template} not found.")
            continue

        for seq in seq_lens:
            print(f"Running: Cores={num_cores}, Seq={seq}, Tile={tile_size}")
            
            # Prepare Inputs
            # Note: num_qkv_tiles used to be just integer division. 
            # With unaligned support, mapping generation logic might need update if it relies strictly on this integer.
            # But standard mapping generation usually just defines 'spatial mapping' which is independent of total loop size.
            # We pass seq//tile for now as a naming convention or approximation if needed.
            gen_flash_attention_onnx(seq, embedding_dim, tile_size, output_dir=str(CURRENT_DIR / "inputs" / "workloads"))
            gen_flash_attention_mapping_config(num_qkv_tiles=seq//tile_size, num_cores=num_cores)
            
            try:
                # experiment_out_dir is already defined at the top
                scme = run_stream_fa(seq, embedding_dim, tile_size, num_cores, experiment_out_dir)
                
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
                
                # Retrieve AUC metrics attached to SCME (requires GA stage to attach them, assumed from previous edit)
                # If these attributes are missing, default to 0.0
                ga_auc = getattr(scme, 'ga_auc', 0.0)
                global_auc = getattr(scme, 'global_auc', 0.0)
                
                # Retrieve New Metrics
                avg_lat_red = getattr(scme, 'avg_latency_reduction', 0.0)
                lat_50_ga = getattr(scme, 'lat_50_ga', 0.0)
                lat_50_global = getattr(scme, 'lat_50_global', 0.0)
                
                imp_base_edp = (1.0 - ga_EDP) * 100
                imp_global_edp = (global_EDP - ga_EDP) / global_EDP * 100 if global_EDP > 0 else 0
                imp_global_auc = (global_auc - ga_auc) / global_auc * 100 if global_auc > 0 else 0
                
                with open(output_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Scaling_Unaligned", num_cores, seq, tile_size, 64, "GA",
                                     base_E, base_L, base_EDP,
                                     global_E_norm, global_L_norm, global_EDP, global_auc,
                                     scme.energy, scme.latency, ga_EDP, ga_auc,
                                     imp_base_edp, imp_global_edp, imp_global_auc,
                                     avg_lat_red, lat_50_global, lat_50_ga])
            except Exception as e:
                logger.error(f"Run Failed for Cores={num_cores}, Seq={seq}, Tile={tile_size}: {e}")

    logger.info(f"Experiment 2 Complete. Results saved to {output_csv}")

if __name__ == "__main__":
    run_exp_scaling()
