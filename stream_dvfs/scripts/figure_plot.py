import os
import sys
# Resolve paths early
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
STREAM_DEV_DIR = STREAM_WORKDIR.parent
sys.path.append(str(STREAM_WORKDIR))

import pickle
import matplotlib.pyplot as plt
import numpy as np

def quantization_comparison_plot(models, quants, stages, input_dir, output_dir):
    """
    Create a 2x2 subplot figure comparing energy savings across models, quantizations, and stages.
    
    Args:
        models: List of model names (should have 4 models)
        quants: List of quantization schemes (e.g., ['W4A8', 'W8A8'])
        stages: List of stages (e.g., ['decode', 'prefill'])
        input_dir: Directory containing the dvfs_meta.pickle files
        output_dir: Directory to save the output figure
    """
    # 1. Gather data from the output directory
    data = {}  # Structure: data[model][stage][quant] = {'ideal': ..., 'pf': ...}
    
    for model in models:
        data[model] = {}
        for stage in stages:
            data[model][stage] = {}
            for quant in quants:
                experiment_id = f"{model}-{quant}-{stage}"
                dvfs_meta_path = os.path.join(input_dir, experiment_id, "dvfs_meta.pickle")
                
                try:
                    with open(dvfs_meta_path, "rb") as file:
                        dvfs_meta = pickle.load(file)
                    
                    ideal_energy_saving = dvfs_meta.get('ideal_energy_saving_at_target', 0)
                    pf_energy_saving = dvfs_meta.get('pf_energy_saving_at_target', 0)
                    
                    # Convert to percentage if needed
                    if ideal_energy_saving is not None and ideal_energy_saving < 1:
                        ideal_energy_saving *= 100
                    if pf_energy_saving is not None and pf_energy_saving < 1:
                        pf_energy_saving *= 100
                    
                    data[model][stage][quant] = {
                        'ideal': ideal_energy_saving if ideal_energy_saving is not None else 0,
                        'pf': pf_energy_saving if pf_energy_saving is not None else 0
                    }
                except FileNotFoundError:
                    print(f"Warning: File not found: {dvfs_meta_path}")
                    data[model][stage][quant] = {'ideal': 0, 'pf': 0}
    
    # 2. Create the 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    axes = axes.flatten()
    
    # Define colors for the bars - same color per method, different hatches for quantization
    color_ideal = '#4A90E2'    # Blue for Ideal DVFS
    color_ga = '#E57373'       # Red for GA DVFS
    
    # Hatch patterns to distinguish W4A8 vs W8A8
    hatch_w4a8 = '\\\\'   # Backward diagonal lines
    hatch_w8a8 = '///'   # Forward diagonal lines

    # Bar settings
    bar_width = 0.2
    group_gap = 0.3
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        # Prepare data for this model
        x_pos = []
        current_x = 0
        
        for stage_idx, stage in enumerate(stages):
            # Position for the 4 bars in this group
            positions = [
                current_x,
                current_x + bar_width,
                current_x + 2 * bar_width,
                current_x + 3 * bar_width
            ]
            
            # Extract values
            ideal_w4a8 = data[model][stage].get(quants[0], {}).get('ideal', 0)
            ideal_w8a8 = data[model][stage].get(quants[1], {}).get('ideal', 0) if len(quants) > 1 else 0
            pf_w4a8 = data[model][stage].get(quants[0], {}).get('pf', 0)
            pf_w8a8 = data[model][stage].get(quants[1], {}).get('pf', 0) if len(quants) > 1 else 0
            
            # Plot bars
            ax.bar(positions[0], ideal_w4a8, bar_width, 
                   color=color_ideal, edgecolor='black', linewidth=0.8,
                   hatch=hatch_w4a8,
                   label='Naive DVFS (W4A8)' if stage_idx == 0 else '')
            ax.bar(positions[1], ideal_w8a8, bar_width, 
                   color=color_ideal, edgecolor='black', linewidth=0.8,
                   hatch=hatch_w8a8,
                   label='Naive DVFS (W8A8)' if stage_idx == 0 else '')
            ax.bar(positions[2], pf_w4a8, bar_width, 
                   color=color_ga, edgecolor='black', linewidth=0.8,
                   hatch=hatch_w4a8,
                   label='GA DVFS (W4A8)' if stage_idx == 0 else '')
            ax.bar(positions[3], pf_w8a8, bar_width, 
                   color=color_ga, edgecolor='black', linewidth=0.8,
                   hatch=hatch_w8a8,
                   label='GA DVFS (W8A8)' if stage_idx == 0 else '')
            
            # Store the center position for x-tick
            x_pos.append(current_x + 1.5 * bar_width)
            
            # Move to next group
            current_x += 4 * bar_width + group_gap
        
        # Customize subplot
        ax.set_ylabel('Energy Saving (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model}', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([stage.capitalize() for stage in stages], fontsize=10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add legend only to the first subplot
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        
        # Set y-axis limits
        ax.set_ylim(0, 40)
    
    # Overall title
    fig.suptitle('Energy Savings @ 10% latency overhead Comparison\nQuantization Schemes across Models and Stages', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'quantization_energy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    models = ['Llama-3.2-1B', 'Llama-3.2-3B', 'Qwen2.5-1.5B', 'Qwen2.5-3B']
    quants = ['W4A8', 'W8A8']
    stages = ['decode', 'prefill']
    
    # Update these paths according to your directory structure
    input_dir = os.path.join(STREAM_DVFS_DIR, 'outputs')
    output_dir = os.path.join(STREAM_DVFS_DIR, 'outputs', 'figures')
    
    quantization_comparison_plot(models, quants, stages, input_dir, output_dir)