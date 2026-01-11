from zigzag.datatypes import LayerOperand

from stream.node_tensor import NodeTensor
from stream.workload.computation.computation_node import ComputationNode, GeneratedComputationNode
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class ConcatConstantNode(PropagationNode):
    """Class that represents an onnx Concat node with one constant input."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        axis: int,
        constant_shape: tuple[int, ...],
        variable_input_first: bool,
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the ConcatConstantNode

        Args:
            predecessors: The id of this node's parent.
            axis: axis in which the input/constants are concatenated
            constant_shape: the shape of the constant tensor
            variable_input_first: Wether the result is `concat(input, constant_tensor)` or
                `concat(constant_tensor, input)`
        """
        if input_names is None:
            input_names = []
        op_type = "concat"
        super().__init__(node_id, node_name, op_type, input_names)

        self.axis = axis
        self.constant_shape = constant_shape
        self.variable_input_first = variable_input_first

        match len(predecessors):
            case 0:
                self.input_operand_source = {}
            case 1:
                self.input_operand_source = {LayerOperand("I"): predecessors[0]}
            case 2:
                # `indices` (the second input) are considered as inputs
                self.input_operand_source = {LayerOperand("W"): predecessors[0], LayerOperand("I"): predecessors[1]}
            case _:
                raise ValueError("More than two inputs for ConcatConstantNode")

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Perform gather operation on the tensor."""
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        relevant_axes[self.axis] = True
        extended_tensor = tensor.concat_with_empty(
            shape=self.constant_shape, axis=self.axis, variable_input_first=self.variable_input_first
        )
        return extended_tensor, relevant_axes


class ConcatNode(PropagationNode):
    """Class that represents an onnx Concat node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        axis: int,
        output_shape: tuple[int, ...],
        input_names: list[str] | None = None,
        axis_exists_in_input: bool = False,
    ) -> None:
        """Initialize the ConcatConstantNode

        Args:
            predecessors: The id of this node's parent.
            axis: axis in which the inputs are concatenated
            output_shape: the shape of the output
            axis_exists_in_input: whether the input already has the axis over which the concationation happens

        """
        if input_names is None:
            input_names = []
        op_type = "concat"
        super().__init__(node_id, node_name, op_type, input_names)
        self.axis = axis
        self.output_shape = output_shape
        self.axis_exists_in_input = axis_exists_in_input

        self.input_operand_source = {LayerOperand(f"I{i}"): node_id for i, node_id in enumerate(predecessors)}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """The input slice is only one of many inputs of this node, but the output tensor should have the shape of the
        concat node output. Return a tensor of all zeros except the input tensor at the correct index"""
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        assert isinstance(previous_node, GeneratedComputationNode), (
            "Concat only supported for procedurally generated nodes for now"
        )
        assert not self.axis_exists_in_input or (
            len(tensor.tensor_shape) == len(self.output_shape) and tensor.tensor_shape[self.axis] == 1
        ), """Input tensor does not have size-1 dimension to concatenate on"""

        slice_idx = previous_node.gen_id
        extended_tensor = tensor.concat_with_empty_both_sides(
            output_shape=self.output_shape,
            axis=self.axis,
            slice_idx=slice_idx,
            axis_exists_in_input=self.axis_exists_in_input,
        )

        # Log this axis as relevant
        relevant_axes[self.axis] = True

        return extended_tensor, relevant_axes


class BlockConcatNode(PropagationNode):
    """Class that represents an onnx Concat node where input blocks (size > 1) are concatenated."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        axis: int,
        output_shape: tuple[int, ...],
        input_names: list[str] | None = None,
        axis_exists_in_input: bool = True,
    ) -> None:
        """Initialize the BlockConcatNode

        Args:
            predecessors: The id of this node's parent.
            axis: axis in which the inputs are concatenated
            output_shape: the shape of the output
            axis_exists_in_input: whether the input already has the axis over which the concationation happens.
                                  Defaults to True as this is primarily for block concatenation.

        """
        if input_names is None:
            input_names = []
        op_type = "concat"
        super().__init__(node_id, node_name, op_type, input_names)
        self.axis = axis
        self.output_shape = output_shape
        self.axis_exists_in_input = axis_exists_in_input

        self.input_operand_source = {LayerOperand(f"I{i}"): node_id for i, node_id in enumerate(predecessors)}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """The input slice is only one of many inputs of this node, but the output tensor should have the shape of the
        concat node output. Return a tensor of all zeros except the input tensor at the correct index"""
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        assert isinstance(previous_node, GeneratedComputationNode), (
            "BlockConcatNode only supported for procedurally generated nodes for now"
        )
        
        # We expect input dimensions to match output dimensions if axis exists
        if self.axis_exists_in_input:
            assert len(tensor.tensor_shape) == len(self.output_shape), "Tensor shape mismatch"
            # We explicitly DO NOT check checks on size 1 dimensions here, unlike ConcatNode

        slice_idx = previous_node.gen_id
        extended_tensor = tensor.concat_with_empty_both_sides_chunk(
            output_shape=self.output_shape,
            axis=self.axis,
            slice_idx=slice_idx,
            axis_exists_in_input=self.axis_exists_in_input,
        )

        # Log this axis as relevant
        relevant_axes[self.axis] = True

        return extended_tensor, relevant_axes

    def propagate_ranges(
        self,
        input_ranges: dict,
        previous_node: Node | None = None,
        next_node: Node | None = None,
    ) -> dict | None:
        """
        Propagate ranges through BlockConcatNode.
        Inputs are blocks (tiles).
        Usually they map to specific offsets.
        """
        output_ranges = input_ranges.copy()
        
        # Determine the slice index (same logic as in propagate)
        slice_idx = -1
        for operand, pred_id in self.input_operand_source.items():
            if pred_id == previous_node.id:
                slice_idx = int(str(operand)[1:])
                break
        
        if slice_idx == -1:
             raise ValueError(f"Predecessor {previous_node} not found in BlockConcatNode predecessors: {self.input_operand_source}")

        # Calculate block size and offset
        output_dim_size = self.output_shape[self.axis]
        num_preds = len(self.input_operand_source)
        if num_preds == 0:
            return output_ranges
        block_size = output_dim_size // num_preds
        offset = slice_idx * block_size

        # Strategy: Use next_node (consumer) to find the target dimension name.
        # The consumer knows which input dimension corresponds to the axis 
        # because the Concat output feeds into strict input operand slots.
        if next_node and isinstance(next_node, ComputationNode) and hasattr(next_node, 'input_operand_source'):
            relevant_operand = None
            # Find which operand of next_node connects to this BlockConcatNode (self)
            for op, source_id in next_node.input_operand_source.items():
                if source_id == self.id:
                    relevant_operand = op
                    break
            
            # Use dimensionality order to find the LayerDim
            if relevant_operand and hasattr(next_node, 'operand_dimensionality_order'):
                dims = next_node.operand_dimensionality_order.get(relevant_operand)
                # self.axis is the index in the tensor. Matches the index in dims list.
                if dims and self.axis < len(dims):
                    target_dim = dims[self.axis]
                    # We override the range for this dimension to be the specific block range
                    output_ranges[target_dim] = (offset, offset + block_size)
                    return output_ranges

        # Fallback
        # If we can't determine the target dimension from consumer, remove axis from ranges
        if self.axis in output_ranges:
            del output_ranges[self.axis]

        return output_ranges

