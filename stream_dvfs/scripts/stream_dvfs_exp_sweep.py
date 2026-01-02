# The main sweeping script for Stream-DVFS experiments
import itertools
import os
import sys
# Resolve paths early
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
STREAM_DEV_DIR = STREAM_WORKDIR.parent
sys.path.append(str(STREAM_DVFS_DIR))
# Import models
from src.config_library import LLAMA1_7B, LLAMA2_7B, LLAMA3_8B, OPT_6_7B
from src.config_library import W4A8, W8A8
from src.util import Stage
# Import simulation function
from scripts.simulation import run_stream, run_dvfs_optimization
# Import plotting function
from scripts.figure_plot import quantization_comparison_plot
# Set output directory
output_dir = STREAM_DVFS_DIR / "outputs" / "exp_sweep"
output_dir.mkdir(exist_ok=True)
# Set output figure directory
output_figure_dir = output_dir / "figures"
output_figure_dir.mkdir(exist_ok=True)
# Set accelerator and mapping paths
accelerator_path = "stream_dvfs/inputs/multicore_system/3core.yaml"
mapping_path = "stream_dvfs/inputs/multicore_mapping/3core_llama_hand_mapping.yaml"
#########################################
# Define the models and quantizations to sweep over
models = [LLAMA1_7B, LLAMA2_7B, LLAMA3_8B, OPT_6_7B]
quants = [W4A8, W8A8]
stages = [Stage.PREFILL, Stage.DECODE]
dvfs_cfg = "stream_dvfs/inputs/dvfs/fine_dvfs.yaml"
def run_stream_experiment():
    for model, quant, stage in itertools.product(models, quants, stages):
        run_stream(model, quant, stage, accelerator_path, mapping_path, output_dir)

def run_dvfs_optimization_experiment():
    for model, quant, stage in itertools.product(models, quants, stages):
        run_dvfs_optimization(model, quant, stage, dvfs_cfg, output_dir)
def plot_quantization_comparison():
    quantization_comparison_plot(models, quants, stages, output_dir, output_figure_dir)

if __name__ == "__main__":
    run_stream_experiment()
    run_dvfs_optimization_experiment()
    plot_quantization_comparison()