import sys
import os
from pathlib import Path
import pickle
import re
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.visualization import bar_plot_cost_model_evaluations_breakdown
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
sys.path.append(str(STREAM_WORKDIR))

zigzag_cme_path = "stream_dvfs/outputs/attention_head-AttentionHeadTest_B=1_FULL_PREFILL_SIZE=1_DECODE_SIZE=1_W8A8_Decode-fused-ga/cost_lut.pickle"
zigzag_cme_dir = os.path.dirname(zigzag_cme_path)

# Create breakdown directory if it doesn't exist
breakdown_dir = os.path.join(zigzag_cme_dir, "breakdown")
os.makedirs(breakdown_dir, exist_ok=True)

with open(zigzag_cme_path, "rb") as fp:
    cmes = pickle.load(fp)


def sanitize_filename(filename):
    """Remove or replace characters that are not valid in filenames"""
    # Replace problematic characters with underscores
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Replace spaces and other characters
    filename = re.sub(r'[\s(),]', '_', filename)
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    return filename
    
total_stall = 0
for node, node_cme in cmes.items():
    node_name = str(node)
    for cme in node_cme.values():
        mem_names = [ml.memory_instance.name for ml in cme.mem_level_list]
        stall_slacks =  cme.stall_slack_comb_collect
        node_stall = cme.latency_total0 - cme.ideal_temporal_cycle
        total_stall += node_stall
        for mem_name, ports_ss in zip(mem_names, stall_slacks):
            print(f"  {mem_name}: {ports_ss}")
        print(
            f"Latency: {cme.latency_total2:.3e} (bd: ideal -> {cme.ideal_temporal_cycle}, spatial_stall -> {cme.ideal_temporal_cycle - cme.ideal_cycle}, temporal_stall -> {cme.latency_total0 - cme.ideal_temporal_cycle}, total_stall -> {cme.latency_total0 - cme.ideal_temporal_cycle}, onload -> {cme.latency_total1 - cme.latency_total0}, offload -> {cme.latency_total2 - cme.latency_total1})"
        )
    
    # Sanitize the node name for use in filename
    safe_node_name = sanitize_filename(node_name)
    save_path = os.path.join(breakdown_dir, f"breakdown_{safe_node_name}.png")
    
    bar_plot_cost_model_evaluations_breakdown([cme], save_path=save_path)
    
print(f"Total temporal stall across all nodes: {total_stall}")