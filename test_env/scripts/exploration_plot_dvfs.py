import sys
import os
import re
sys.path.append(os.getcwd())

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def extract_oy_number(mapping_name):
    match = re.search(r'OY(\d+)$', mapping_name)
    return int(match.group(1)) if match else 0
def mapping_sort_key(item):
    return extract_oy_number(item["mapping"])
# Configuration
output_base_dir = "test_env/outputs/"
save_dir = "test_env/outputs/visualization_results/"
workload_list = ["resnet18", "mobilenetv2", "fsrcnn", "squeezenet", "inception_v2"]
# workload_list = ["resnet18", "mobilenetv2", "fsrcnn", "squeezenet", "inception_v2","mobilebert", "tinyyolov2", "xception"]

# Create save directory
os.makedirs(save_dir, exist_ok=True)

# plt.style.use('seaborn')
colors = plt.cm.tab20.colors  
markers = ['o', 's', '^', 'D', 'v', 'p', '*', '<', '>', 'H', 'X', 'd']

for workload in workload_list:
    plt.figure(figsize=(14, 9))
    ax = plt.gca()
    experiments = []
    
    # 收集实验数据
    for exp_dir in os.listdir(output_base_dir):
        exp_path = os.path.join(output_base_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        parts = exp_dir.split("-")
        if len(parts) < 4:
            continue
            
        current_workload = parts[1]
        if current_workload != workload:
            continue
            
        pickle_path = os.path.join(exp_path, "dvfs", "dvfs_meta.pickle")
        if not os.path.exists(pickle_path):
            continue
            
        with open(pickle_path, 'rb') as f:
            dvfs_meta = pickle.load(f)
            
        experiments.append({
            "hardware": parts[0],  # 硬件类型
            "mapping": parts[2],   # mapping策略
            "pf_energy": np.array(dvfs_meta["pf_energy"]),
            "pf_latency": np.array(dvfs_meta["pf_latency"]),
            "base_energy": dvfs_meta["base_energy"],
            "base_latency": dvfs_meta["base_latency"]
        })

    experiments.sort(key=mapping_sort_key)
    

    unique_mappings = sorted(
        list(OrderedDict.fromkeys([e["mapping"] for e in experiments])),
        key=extract_oy_number
    )
    unique_hardwares = sorted(list(OrderedDict.fromkeys([e["hardware"] for e in experiments])))
    

    color_map = {m: colors[i % len(colors)] for i, m in enumerate(unique_mappings)}
    marker_map = {h: markers[i % len(markers)] for i, h in enumerate(unique_hardwares)}    
    

    for exp in experiments:

        sort_idx = np.argsort(exp["pf_energy"])
        sorted_energy = exp["pf_energy"][sort_idx]
        sorted_latency = exp["pf_latency"][sort_idx]
        

        ax.plot(sorted_energy, sorted_latency,
               marker=marker_map[exp["hardware"]],
               markersize=8,
               linestyle='--',
               linewidth=1.5,
               color=color_map[exp["mapping"]],
               alpha=0.8,
               label=f'{exp["mapping"]} ({exp["hardware"]})')
        

        ax.scatter(exp["base_energy"], exp["base_latency"],
                   color=color_map[exp["mapping"]],
                   marker=marker_map[exp["hardware"]],
                   s=180,
                   edgecolors='black',
                   linewidth=1,
                   zorder=5)


    if experiments:
        ax.axhline(y=exp["base_latency"], color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=exp["base_energy"], color='gray', linestyle=':', alpha=0.5)
    

    ax.set_xlabel('Energy Consumption (pJ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (CC)', fontsize=12, fontweight='bold')
    ax.set_title(f'Pareto Front Analysis: {workload.upper()}', 
                fontsize=14, fontweight='bold', pad=20)
    

    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10)
    

    legend_handles = []
    

    for mapping, color in color_map.items():
        legend_handles.append(plt.Line2D([0], [0],
                                        color=color,
                                        marker='s',
                                        linestyle='None',
                                        markersize=8,
                                        label=mapping))
        

    for hardware, marker in marker_map.items():
        legend_handles.append(plt.Line2D([0], [0],
                                        color='gray',
                                        marker=marker,
                                        linestyle='None',
                                        markersize=8,
                                        label=hardware))
        

    legend_handles.append(plt.Line2D([0], [0],
                                   marker='X',
                                   color='w',
                                   markerfacecolor='gray',
                                   markeredgecolor='black',
                                   markersize=10,
                                   label='Base Configuration'))
    

    ax.legend(handles=legend_handles,
             loc='upper right',
             ncol=2,
             fontsize=12,
             framealpha=0.9,
             title='Hardware Configuration',
             title_fontsize=12,
             bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'pareto_{workload}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

print(f"Visualization completed. Results saved to: {save_dir}")