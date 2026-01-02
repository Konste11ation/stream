import sys
import os
from pathlib import Path
import argparse

# Resolve paths early
CURRENT_DIR = Path(__file__).resolve().parent  # noqa: F821 
STREAM_DVFS_DIR = CURRENT_DIR.parent  # noqa: F821 
sys.path.append(str(STREAM_DVFS_DIR))  # noqa: F821 

from src.config_library import W4A8, W8A8, W4A16, W16A16, W32A32  # noqa: E402 
from src.config_library import LLAMA1_7B, LLAMA2_7B, LLAMA3_8B, OPT_6_7B, FlashAttentionTestConfig  # noqa: E402 
from src.util import Stage, get_onnx_path  # noqa: E402 
from src.export_onnx import export_model_to_onnx  # noqa: E402 

# Maps for argparse choices -> actual objects
MODEL_CHOICES = {
    "llama1_7b": LLAMA1_7B,
    "llama2_7b": LLAMA2_7B,
    "llama3_8b": LLAMA3_8B,
    "opt_6_7b": OPT_6_7B,
    "fa_test": FlashAttentionTestConfig,
}  

QUANT_CHOICES = {
    "w4a8": W4A8,
    "w8a8": W8A8,
    "w4a16": W4A16,
    "w16a16": W16A16,
    "w32a32": W32A32,
}  

STAGE_CHOICES = {
    "prefill": Stage.PREFILL,
    "decode": Stage.DECODE,
}  

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a model to ONNX with selectable model, quant, and stage.",  
    )  

    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES.keys(),
        default="llama2_7b",
        help="Model config to export: {%(choices)s} (default: %(default)s)",  
    )  

    parser.add_argument(
        "--quant",
        choices=QUANT_CHOICES.keys(),
        default="w8a8",
        help="Quantization config: {%(choices)s} (default: %(default)s)",  
    )  

    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES.keys(),
        default="prefill",
        help="Pipeline stage: {%(choices)s} (default: %(default)s)",  
    )  

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STREAM_DVFS_DIR / "inputs" / "workloads",
        help="Output directory for ONNX file (default: STREAM_DVFS_DIR/inputs/workloads)",
    )  

    return parser.parse_args()  

def main():
    args = parse_args()  

    model = MODEL_CHOICES[args.model]
    quant = QUANT_CHOICES[args.quant]
    stage = STAGE_CHOICES[args.stage]  
    print(f"Exporting model: {args.model}, quant: {args.quant}, stage: {args.stage}")
    onnx_path = get_onnx_path(
        output_dir=args.output_dir,
        model=model,
        stage=stage,
        quant=quant,
    )  

    export_model_to_onnx(model, quant, output_path=onnx_path, stage=stage)  

if __name__ == "__main__":
    main()  
