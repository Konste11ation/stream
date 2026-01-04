from zigzag.datatypes import LayerOperand

from stream.node_tensor import NodeTensor
from stream.workload.computation.computation_node import GeneratedComputationNode
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node

class DiagNode(PropagationNode):
    """Class that represents a diagonal matrix creation.
    
    Transforms a vector of shape (..., N) into diagonal matrices of shape (..., N, N).
    For example:
    - Input shape: (Batch, Br) where each element is a scaling factor
    - Output shape: (Batch, Br, Br) where each (batch, :) is a Br x Br diagonal matrix
    """

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the DiagNode

        Args:
            node_id: Unique identifier for this node
            node_name: Name of the node
            predecessors: The id of this node's parent (should have exactly 1 predecessor)
            input_names: Names of the input tensors
        """
        if input_names is None:
            input_names = []
        op_type = "diag"
        super().__init__(node_id, node_name, op_type, input_names)

        match len(predecessors):
            case 0:
                self.input_operand_source = {}
            case 1:
                self.input_operand_source = {LayerOperand("I"): predecessors[0]}
            case _:
                raise ValueError("More than one input for DiagNode")

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Perform diag operation on the tensor.
        
        Converts a vector tensor of shape (..., N) to diagonal matrices of shape (..., N, N).
        
        Args:
            tensor: Input NodeTensor with shape (..., N)
            previous_node: The predecessor node (unused)
            next_node: The successor node (unused)
            relevant_axes: Boolean list indicating which axes are relevant for computation
        
        Returns:
            Tuple of (output_tensor, updated_relevant_axes)
            - output_tensor: NodeTensor with shape (..., N, N)
            - updated_relevant_axes: Updated boolean list for the output shape
        """
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        
        # Apply the diag operation to create diagonal matrices
        output_tensor = tensor.diag()
        
        # Update relevant_axes for the output shape
        # The output has one more dimension than input: (..., N, N)
        # All axes remain relevant, and the new diagonal dimension is also relevant
        updated_relevant_axes = relevant_axes + [True]
        
        return output_tensor, updated_relevant_axes

    