import torch
import torch.nn as nn
from torch import Tensor


class NamedEinsum(nn.Module):
    """Wrap the einsum operator to allow for custom names in the exported ONNX model"""

    def forward(self, operation: str, a: Tensor, b: Tensor):
        return torch.einsum(operation, a, b)


class NamedEinsumSingleOp(nn.Module):
    """Wrap the einsum operator to allow for custom names in the exported ONNX model"""

    def forward(self, operation: str, a: Tensor):
        return torch.einsum(operation, a)


class NamedMul(nn.Module):
    """Wrap the einsum operator to allow for custom names in the exported ONNX model"""

    def forward(self, a: Tensor, b: Tensor):
        return a * b


class NamedMatmul(nn.Module):
    """Wrap Torch Matmul operator so that the operation can be given a custom name that is exported to ONNX"""

    def forward(self, a: Tensor, b: Tensor):
        return a @ b


class NamedAdd(nn.Module):
    """Wrap Torch addition operator (+) so that the operation can be given a custom name that is exported to ONNX"""

    def forward(self, a: Tensor, b: Tensor):
        return a + b


class DummyEinsum(nn.Module):
    """Wrap the einsum operator to allow for custom names in the exported ONNX model.
    Replace operation with dummy so it won't be parsed by Stream"""

    def forward(self, operation: str, a: Tensor, b: Tensor):
        return a - b  # just use some operation that uses both tensors and is not implemented in stream
