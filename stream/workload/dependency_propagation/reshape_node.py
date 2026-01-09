from yaml import Node
from zigzag.datatypes import Constants
from math import prod

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.computation.computation_node import ComputationNode


class ReshapeNode(PropagationNode):
    """Class that represents an onnx Reshape node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        shape: tuple[int, ...],
        allow_zero: bool = False,
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the ReshapeNode

        Args:
            predecessors: The id of this node's parent.
            shape: The output tensor's shape.
            allow_zero: wether the output shape can be 0 at some dimensions. Iff True, shape `[2,0,3]` becomes `[2,3]`
        """
        if input_names is None:
            input_names = []
        op_type = "reshape"
        super().__init__(node_id, node_name, op_type, input_names)

        self.allow_zero = allow_zero
        self.shape = shape
        self.input_operand_source = {Constants.LAYER_OP_I: predecessor}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Reshape the tensor back to the representation needed for producer/consumer."""
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        new_shape = self.shape
        if not new_shape:
            return tensor, relevant_axes

        if not self.allow_zero:
            new_shape = tuple(x for x in new_shape if x != 0)

        relevant_axes = self.update_relevant_axes(relevant_axes, tensor.tensor_shape, new_shape)

        return tensor.reshape(new_shape), relevant_axes

    def update_relevant_axes(self, relevant_axes: list[bool], old_shape: tuple[int, ...], new_shape: tuple[int, ...]):
        if len(new_shape) < len(old_shape):
            # We need to cut an axis
            try:
                axis_to_cut = next(i for i in range(len(new_shape)) if old_shape[i] != new_shape[i])
            except StopIteration:
                axis_to_cut = len(old_shape) - 1

            new_shape_list = list(new_shape)
            del relevant_axes[axis_to_cut]
            del new_shape_list[axis_to_cut]

            for idx, (old_dim, new_dim) in enumerate(zip(old_shape, new_shape_list, strict=False)):
                if old_dim != new_dim:
                    relevant_axes[idx] = True

            return relevant_axes

        if len(new_shape) > len(old_shape):
            # We need to add an axes
            relevant_axes.append(new_shape[-1] != old_shape[-1])

        for idx, (old_dim, new_dim) in enumerate(zip(old_shape, new_shape, strict=False)):
            if old_dim != new_dim:
                relevant_axes[idx] = True

        return relevant_axes

    def propagate_ranges(
        self,
        input_ranges: dict,
        previous_node: Node | None = None,
        next_node: Node | None = None,
    ) -> dict | None:
        """
        Propagate range through Reshape.
        Detailed logic:
        1. Identify input shape from previous_node.
        2. Identify output shape (self.shape).
        3. Find mapping between Input Dimensions and Output Dimensions.
        4. Convert Input Ranges to Output Ranges.
        """
        # 1. Determine Input Shape
        input_shape = None
        if isinstance(previous_node, ComputationNode):
            # Try Output Operand
            if Constants.OUTPUT_LAYER_OP in previous_node.operand_dimensionality_order:
                dims = previous_node.operand_dimensionality_order[Constants.OUTPUT_LAYER_OP]
                # Assuming simple loop ranges cover the tensor size for now
                # Usually loop_ranges[d] = (0, size)
                input_shape = []
                for d in dims:
                    r = previous_node.loop_ranges.get(d)
                    if r:
                        input_shape.append(r[1] - r[0])
                    else:
                        input_shape.append(1) # Unknown?
                input_shape = tuple(input_shape)
        
        # Fallback if unknown or if shape mismatch
        output_shape = self.shape
        if not self.allow_zero and output_shape:
             output_shape = tuple(x for x in output_shape if x != 0)

        if not input_shape:
            # Cannot calculate stride logic without input shape.
            # Conservative: Return empty (Full Dependency)
            return {}
            
        current_prod = prod(input_shape)
        target_prod = prod(output_shape)
        import math
        # Handle -1 in reshape
        if -1 in output_shape:
            # infer
            known_prod = -1 * prod(output_shape)
            missing = current_prod // known_prod
            output_shape = tuple(x if x != -1 else missing for x in output_shape)
            
        if prod(input_shape) != prod(output_shape):
            # Shape mismatch, cannot propagate safely
            return {}

        # 2. Block Alignment Algorithm
        output_ranges = {}
        
        in_idx = 0
        out_idx = 0
        curr_in_prod = input_shape[0] if input_shape else 1
        curr_out_prod = output_shape[0] if output_shape else 1
        
        in_block_start = 0
        out_block_start = 0
        
        while in_idx < len(input_shape) and out_idx < len(output_shape):
            if curr_in_prod == curr_out_prod:
                # Found a boundary match.
                # Input Dims [in_block_start ... in_idx]  <-> Output Dims [out_block_start ... out_idx]
                
                # Check mapping type
                # 1-to-1
                if in_idx == in_block_start and out_idx == out_block_start:
                    if in_idx in input_ranges:
                        output_ranges[out_idx] = input_ranges[in_idx]
                        
                # 1-to-Many (Split)
                elif in_idx == in_block_start:
                    if in_idx in input_ranges:
                        # Split Logic
                        start, end = input_ranges[in_idx]
                        # Compute strides for the output block
                        sub_out_shape = output_shape[out_block_start : out_idx + 1]
                        
                        # We map [start, end) to the sub_out_shape
                        # Since we want a bounding box, we look at the start and end-1 points
                        # Convert flat index to sub-indices
                        
                        # Helper for Multi-Index
                        def get_indices(flat_val, shape):
                            indices = []
                            for dim_size in reversed(shape):
                                indices.append(flat_val % dim_size)
                                flat_val //= dim_size
                            return list(reversed(indices))

                        # Range is [start, end) -> valid [start, end-1]
                        start_indices = get_indices(start, sub_out_shape)
                        end_indices_inclusive = get_indices(end - 1, sub_out_shape)
                        
                        # Bounding Box Logic with Wrapping Check
                        for k in range(len(sub_out_shape)):
                            dim_abs = out_block_start + k
                            s_val = start_indices[k]
                            e_val = end_indices_inclusive[k]
                            
                            # Check if higher dimensions changed between start and end
                            higher_changed = False
                            if k > 0:
                                if start_indices[:k] != end_indices_inclusive[:k]:
                                    higher_changed = True
                            
                            if not higher_changed:
                                # Safe range in this dimension
                                # Range [s_val, e_val + 1)
                                if e_val >= s_val:
                                    output_ranges[dim_abs] = (s_val, e_val + 1)
                                else:
                                    # Wrapped around within this block?
                                    # If higher dims same, e_val >= s_val must hold for monotonic increase?
                                    # No, if purely 1-to-Many split of a contiguous range, it is monotonic.
                                    pass 
                            else:
                                # Higher dimension changed, meaning we wrapped.
                                # This dimension likely covers full range [0, dim_size)
                                # Unless total range is very small? 
                                # For Bounding Box, if we wrap, usually it is safer to assume Full.
                                # Exception: We cover 'tail' of first row and 'head' of last row.
                                # The Union is [min(head, tail), max(...)] -> usually full 0..N
                                # Safe fallback: Do not set range (implies full)
                                pass
                                
                # Many-to-1 (Merge)
                elif out_idx == out_block_start:
                    # Check if all component inputs are constrained
                    # Construct flat range for this block
                    # If any component input is NOT in input_ranges, assume FULL => Output is FULL.
                    
                    all_constrained = True
                    # Calculate cumulative strides for input block
                    sub_in_shape = input_shape[in_block_start : in_idx + 1]
                    total_block_size = curr_out_prod # same as curr_in_prod
                    
                    # We need to linearize the input box.
                    # BBox linearize: Min = sum(start_i * stride_i), Max = sum((end_i-1)*stride_i) + 1
                    # This is valid ONLY if the inputs form a contiguous chunk?
                    # No. Inputs form a Grid. A grid linearized is NOT a single interval.
                    # It is a set of intervals.
                    # The output is a single dimension.
                    # So the set of intervals becomes a Bounding Box on that single dimension.
                    # The Min is the first element, Max is the last element.
                    # So yes, we can just calculate Start and End points.
                    
                    batch_min = 0
                    batch_max = 0
                    current_stride = 1
                    
                    for k in reversed(range(len(sub_in_shape))):
                        dim_abs = in_block_start + k
                        if dim_abs in input_ranges:
                            s, e = input_ranges[dim_abs]
                            batch_min += s * current_stride
                            batch_max += (e - 1) * current_stride
                        else:
                            # Unconstrained dimension.
                            # min adds 0.
                            # max adds (size - 1) * stride
                            batch_max += (sub_in_shape[k] - 1) * current_stride
                        current_stride *= sub_in_shape[k]
                        
                    output_ranges[out_idx] = (batch_min, batch_max + 1)
                
                # Many-to-Many
                else:
                    # Too complex, fallback to Full
                    pass
                
                # Advance to next block
                in_idx += 1
                out_idx += 1
                in_block_start = in_idx
                out_block_start = out_idx
                if in_idx < len(input_shape):
                    curr_in_prod = input_shape[in_idx]
                if out_idx < len(output_shape):
                    curr_out_prod = output_shape[out_idx]
                    
            elif curr_in_prod < curr_out_prod:
                in_idx += 1
                if in_idx < len(input_shape):
                    curr_in_prod *= input_shape[in_idx]
                else:
                     break # Should not happen if sizes match
            else:
                out_idx += 1
                if out_idx < len(output_shape):
                    curr_out_prod *= output_shape[out_idx]
                else:
                    break

        return output_ranges
