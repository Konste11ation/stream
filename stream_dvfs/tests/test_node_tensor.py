# The testing function for the NodeTensor class
import sys

from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
sys.path.append(str(STREAM_WORKDIR))

from stream.node_tensor import NodeTensor
import numpy as np

# Create a NodeTensor with shape (3, 4, 5)
data = np.zeros((1,32,32))
print("Data shape before NodeTensor initialization:", data.shape)
node_tensor = NodeTensor(data, pre_allocation_size=32)
print("NodeTensor shape after initialization:", node_tensor.full_shape)

# Slice along axis 1 (rows) from index 1 to 3
sliced_tensor = node_tensor.slice(starts=0, ends=16, axis=1)

print("Sliced NodeTensor (axis=1, starts=0, ends=16):")
print(sliced_tensor.full_shape)