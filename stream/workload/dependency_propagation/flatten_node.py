from math import prod
import numpy as np
from zigzag.datatypes import LayerOperand, Constants

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node
from stream.workload.computation.computation_node import ComputationNode


class FlattenNode(PropagationNode):
    """Class that represents an onnx Flatten node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int | None,
        axis: int | None,
        input_names: list[str],
    ) -> None:
        """Initialize the FlattenNode

        Args:
            shape: The output tensor's shape.
        """
        op_type = "flatten"
        super().__init__(node_id, node_name, op_type, input_names)

        self.axis = axis
        if predecessor is not None:
            self.input_operand_source = {LayerOperand("I"): predecessor}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Reshape an input tensor"""
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        shape = tensor.tensor_shape
        # taken from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-51
        new_shape = (1, -1) if self.axis == 0 else (np.prod(shape[0 : self.axis]).astype(int), -1)
        # All axes will be relevant in case of a flatten operation
        relevant_axes = [True] * len(new_shape)
        return tensor.reshape(new_shape), relevant_axes

    def propagate_ranges(
        self,
        input_ranges: dict,
        previous_node: Node | None = None,
        next_node: Node | None = None,
    ) -> dict | None:
        """
        Propagate ranges through Flatten.
        Input Dims [0..axis-1] -> Output Dim 0
        Input Dims [axis..N-1] -> Output Dim 1
        """
        # 1. Determine Input Shape
        input_shape = None
        if isinstance(previous_node, ComputationNode):
             if Constants.OUTPUT_LAYER_OP in previous_node.operand_dimensionality_order:
                dims = previous_node.operand_dimensionality_order[Constants.OUTPUT_LAYER_OP]
                input_shape = []
                for d in dims:
                    r = previous_node.loop_ranges.get(d)
                    if r:
                        input_shape.append(r[1] - r[0])
                    else:
                        input_shape.append(1)
                input_shape = tuple(input_shape)

        if not input_shape:
            return {}
            
        # 2. Split dimensions based on axis
        rank = len(input_shape)
        # Handle axis formatting
        axis = self.axis if self.axis is not None else 1
        if axis < 0:
            axis += rank
            
        groups = [
            range(0, axis),          # Group 0 -> Output 0
            range(axis, rank)        # Group 1 -> Output 1
        ]
        
        output_ranges = {}
        
        for out_dim, group in enumerate(groups):
            if not group:
                continue
                
            # Extract sub-shape for this group
            sub_shape = [input_shape[i] for i in group]
            
            # Linearize Min/Max
            # Min = sum(start_i * stride_i)
            # Max = sum((end_i - 1) * stride_i)
            
            group_min = 0
            group_max = 0
            
            current_stride = 1
            # Iterate backwards (least significant dim first)
            for k in reversed(range(len(group))):
                in_dim = group[k]
                dim_size = sub_shape[k]
                
                if in_dim in input_ranges:
                    s, e = input_ranges[in_dim]
                    # Constraints
                    group_min += s * current_stride
                    group_max += (e - 1) * current_stride
                else:
                    # Full range
                    # min += 0
                    group_max += (dim_size - 1) * current_stride
                
                current_stride *= dim_size
                
            output_ranges[out_dim] = (group_min, group_max + 1)
            
        return output_ranges
