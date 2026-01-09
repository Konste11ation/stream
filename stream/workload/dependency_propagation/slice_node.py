from zigzag.datatypes import Constants

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class SliceNode(PropagationNode):
    """Class that represents an onnx Slice node."""

    def __init__(  # noqa: PLR0913
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        starts: list[int],
        ends: list[int],
        axes: list[int],
        steps: list[int],
        output_names: list[str],
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the SliceNode
        Slice the tensor at axis `axis`. The sizes are given by `Slices`. `len(Slices)` is the number of output nodes.

        Args:
            predecessors: The id of this node's parent.
            axis: axis in which to Slice
            Slices: sizes of the output Slices in the given axis
            output_names: the node names that correspond to the Slices
        """
        if input_names is None:
            input_names = []
        op_type = "Slice"
        super().__init__(node_id, node_name, op_type, input_names)

        self.starts = starts
        self.ends = ends
        self.axes = axes
        self.steps = steps
        self.input_operand_source = {Constants.LAYER_OP_I: predecessor}
        self.output_names = output_names

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Slice the tensor.
        Currently assumes only one slice is created."""
        sliced_tensor = tensor.slice(starts=self.starts[0], ends=self.ends[0], axis=self.axes[0], steps=self.steps[0])
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        relevant_axes[self.axes[0]] = True
        return sliced_tensor, relevant_axes

    def propagate_ranges(
        self,
        input_ranges: dict,
        previous_node: Node | None = None,
        next_node: Node | None = None,
    ) -> dict | None:
        """
        Propagate the input ranges through the Slice operation.
        We intersect the input range with the slice range and shift the coordinates.
        If intersection is empty, return None.
        """
        output_ranges = input_ranges.copy()
        
        # Iterate over the sliced axes
        for i, axis in enumerate(self.axes):
            start = self.starts[i]
            end = self.ends[i]
            step = self.steps[i]
            
            # If this axis is being tracked (it's in the loop ranges)
            if axis in output_ranges:
                in_start, in_end = output_ranges[axis]
                
                # 1. Intersection of Input Range [in_start, in_end) and Slice Range [start, end)
                # Note: Handling negative steps or complex slicing would be harder. Assuming positive step.
                inter_start = max(in_start, start)
                inter_end = min(in_end, end)
                
                if inter_start >= inter_end:
                    # No overlap -> No dependency
                    return None
                    
                # 2. Shift to output coordinates (relative to slice start)
                # O_val = (I_val - start) / step
                # We need to map the start/end carefully.
                # The first valid index in intersection is inter_start.
                # However, it must also align with the step if step > 1.
                # The slice generates indices start, start+step, start+2*step...
                # We need smallest k such that start + k*step >= inter_start
                
                # Simplified for step=1 (common case)
                if step == 1:
                    out_start = inter_start - start
                    out_end = inter_end - start
                else:
                    # Generic case (assuming step > 0)
                    # Align inter_start to the grid
                    offset = (inter_start - start) % step
                    if offset != 0:
                        inter_start += (step - offset)
                    
                    if inter_start >= inter_end:
                        return None
                        
                    out_start = (inter_start - start) // step
                    # For end, we just need the count of elements
                    # Number of elements = ceil((inter_end - inter_start) / step)
                    
                    # More simply: The last valid element is the largest start + k*step < inter_end
                    # Let's trust simple integer arithmetic
                    # Output valid range is from out_start to ...
                    out_end = (inter_end - start + step - 1) // step
                    
                output_ranges[axis] = (out_start, out_end)
                
        return output_ranges
