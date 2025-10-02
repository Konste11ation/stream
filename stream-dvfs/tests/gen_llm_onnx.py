import sys
import os
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
sys.path.append(str(STREAM_DVFS_DIR))

from src.config_library import W8A8, LLAMA1_7B
from src.util import Stage, get_onnx_path
from src.export_onnx import export_model_to_onnx
model = LLAMA1_7B
quant = W8A8    
model.batch_size = 1
stage = Stage.DECODE
onnx_path = get_onnx_path(output_dir=STREAM_DVFS_DIR / "inputs" / "workloads",
                          model=model,
                          stage=stage,
                          quant=quant)
export_model_to_onnx(model, quant, path=onnx_path, stage=stage)