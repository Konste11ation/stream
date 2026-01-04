# The flash attention parser
import logging
from enum import StrEnum
from typing import Any
from zigzag.datatypes import Constants
from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from onnx import ModelProto, NodeProto
from zigzag.parser.workload_factory import LayerNodeFactory
from stream.workload.mapping import InterCoreMappingAttributes
from stream.hardware.architecture.accelerator import Accelerator
from stream.onnx_utils import get_onnx_input_shapes, get_onnx_output_shapes
from stream.workload.node import Node
from stream.workload.computation.computation_node import ComputationNode
# Dependency propagation nodes
from stream.workload.dependency_propagation.reshape_node import ReshapeNode
from stream.workload.dependency_propagation.slice_node import SliceNode
from stream.workload.dependency_propagation.concat_node import ConcatNode
from stream.workload.dependency_propagation.diag_node import DiagNode
# Dummy node for diagonal
from stream.workload.dependency_propagation.dummy_node import DummyNode
logger = logging.getLogger(__name__)

class FlashAttentionParser(OnnxComputeOperatorParser):
    """Parses the FlashAttention operator into a single computation node.
    FlashAttention is a memory-efficient attention mechanism that computes attention in a tiled manner.
    This parser creates a computation node that represents the entire FlashAttention operation.
    """

    NODE_TYPE = "FlashAttention"
    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        onnx_model: ModelProto,
        all_mappings: dict[str, InterCoreMappingAttributes],
        accelerator: Accelerator,
    ):
        super().__init__(
            node_id=node_id,
            node=node,
            nodes_outputs=nodes_outputs,
            onnx_model=onnx_model,
            all_mappings=all_mappings,
            accelerator=accelerator,
        )
        self.__node_id_tracker = node_id
        self.node_name_to_id: dict[str, int] = {}
        self.tensor_shape: tuple[int, ...] = ()
        self.batch: int = 0
        self.seq_len: int = 0
        self.hidden_dim: int = 0
        self.tile_Br: int = 0
        self.tile_Bc: int = 0
        self.Tr: int = 0
        self.Tc: int = 0
        self.init_set_shape_info()
        self.init_set_tile_info()
        # For now just fixed numbers
        self.operand_precision = {"W": 8,
                                  "I": 8,
                                  "O_final": 8,
                                  "O": 16}
    # Top level run function
    # Call the get_nodes funtion
    def run(self):
        yield from self.get_nodes()
    # Set shape info
    def init_set_shape_info(self):
        """" Set the FlashAttention shape information from the ONNX"""
        input_shapes = get_onnx_input_shapes(self.node, self.onnx_model)
        output_shapes = get_onnx_output_shapes(self.node, self.onnx_model)
        # Sanity check on the inputs and outputs
        EXPECTED_NUMBER_OF_INPUTS = 3
        EXPECTED_NUMBER_OF_OUTPUTS = 1
        assert len(input_shapes) == EXPECTED_NUMBER_OF_INPUTS, "FlashAttention node must have 3 (QKV) inputs"
        assert len(output_shapes) == EXPECTED_NUMBER_OF_OUTPUTS, "FlashAttention node must have 1(O) output"
        assert input_shapes[0] == output_shapes[0], "FlashAttention input and output shapes must match"
        # For now the input shapes are assumend to be in the format [Batch, Seq_Len, Hidden_Dim]
        Batch, Seq_Len, Hidden_Dim = input_shapes[0]
        self.tensor_shape = (Batch, Seq_Len, Hidden_Dim)
        self.batch = Batch
        self.seq_len = Seq_Len
        self.hidden_dim = Hidden_Dim
        
    def init_set_tile_info(self):
        """ Set the FlashAttention tiling information """

        # For now we set to a fix value
        DEFAULT_TILE_Br = 16
        DEFAULT_TILE_Bc = 16
        self.tile_Br = DEFAULT_TILE_Br
        self.tile_Bc = DEFAULT_TILE_Bc
        self.Tr = self.seq_len // self.tile_Br # Number of row tiles, for Q and O
        self.Tc = self.seq_len // self.tile_Bc # Number of column tiles, for K and V
        # For now we assume seq_len is divisible by tile sizes
        assert self.seq_len % self.tile_Br == 0, "Sequence length must be divisible by tile size Br"
        assert self.seq_len % self.tile_Bc == 0, "Sequence length must be divisible by tile size Bc"
        # TODO 1: Compute the Br and Bc from the memory capacity
        # TODO 2: Handle the case where seq_len is not divisible by tile sizes
    def get_layer_node_user_format(
        self, input_shape: list[int], output_shape: list[int], mapping: Any | None
    ) -> dict[str, Any]:
        """Not used for this class, but abstract base class requires instantiation anyway"""
        ...
    # Top functions
    def get_nodes(self):
        # Parse the FlashAttention CN
        self.parse_into_subnodes()
        # Format the node names
        self.format_node_names()
        return self.nodes
    def parse_into_subnodes(self):
        """Parse the base ONNX node into several FlashAttention nodes."""
        # 1. Preprocessing nodes
        preprocessing_nodes = self.get_preprocessing_nodes()
        # 2. Compute nodes
        compute_nodes = []
        for idx in range(self.Tr):
            for jdx in range(self.Tc):
                compute_nodes.extend(self.get_compute_qkv_tile_nodes(idx, jdx))
            # 3. Output nodes
        compute_nodes.extend(self.get_output_o_nodes())
        
        self.nodes = tuple(preprocessing_nodes + compute_nodes)
    def format_node_names(self):
        # Add the prefix to all node names
        fa_node_prefix = self.node.name
        for node in self.nodes:
            node.name = f"{fa_node_prefix}/{node.name}"
            
            
    # To create a computation node, the function needs to create a dictionary
    # node_data: dict[str, Any] = {}
    # Include the following fields:
    # 1. id
    # 2. name
    # 3. operator_type
    # 4. dimension_relations: null for now
    # 5. operand_source: list of input sources
    # 6. operand_precision: list of precisions for each operand
    # 7. loop_dims: list of names for each loop dimension
    # 8. loop_sizes: list of sizes for each loop dimension
    # 9. equation: the computation equation in string format
    # And then mimic the generate_node function in stream/stream/parser/onnx/operator_parser.py
    
    # To create a dependency propagation node, just needs to create the class instance
    # Below are some helper functions to create the nodes needed for FlashAttention
    def _helper_create_reshape_k_node(self, node_id, pred_id):
        # This function create the reshape k node
        # Kj_T = reshape(Kj)
        # Kj: [Batch, Seq_Len, Hidden_Dim] -> Kj_T: [Batch, Hidden_Dim, Seq_Len]
        # The shape is the new shape after reshape
        return ReshapeNode(
            node_id=node_id,
            node_name=f"reshape_k",
            predecessor=pred_id,
            shape=(self.batch, self.hidden_dim, self.seq_len),
            input_names=list(self.node.input[1]),
        )
    def _helper_create_slice_qkv_node(self, input_name, idx, node_id, pred_id):
        # This function create the slice qkv node
        # input names should be only 1 from ["Q", "K", "V"]
        if input_name == "Q":
            node_name = f"slice_q_{idx}"
            start = idx * self.tile_Br
            end = start + self.tile_Br
            axe = 1
            input_name = self.node.input[0]
            output_names = [f"Q_tile_{idx}"]
        if input_name == "K":
            node_name = f"slice_k_{idx}"
            start = idx * self.tile_Bc
            end = start + self.tile_Bc
            axe = 1
            input_name = "reshape_k"
            output_names = [f"K_tile_{idx}" ]
        if input_name == "V":
            node_name = f"slice_v_{idx}"
            start = idx * self.tile_Bc
            end = start + self.tile_Bc
            axe = 2
            input_name = self.node.input[2]
            output_names = [f"V_tile_{idx}"]
        return SliceNode(
            node_id=node_id,
            node_name=node_name,
            predecessor=pred_id,
            starts=[start],
            ends=[end],
            axes=[axe],
            steps=[1],
            input_names=[input_name],
            output_names=output_names,
        )
    def _helper_create_gemm_qk_node(self, id, pred_id_input_Qi, pred_id_input_Kj, idx, jdx):
        # This function create the gemm qk node
        # 1. x = gemm(Qi, Kj.T) [Br x Bc]
        # Qi: [Batch, Br, Hidden_Dim]
        # Kj.T: [Batch, Hidden_Dim, Bc]
        # x: [Batch, Br, Bc]
        # idx: index for Q
        # jdx: index for K
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"gemm_qk_i{idx}_j{jdx}"
        node_data["operator_type"] = "FA_Gemm"
        node_data["operand_source"] = {"I": pred_id_input_Qi, "W": pred_id_input_Kj}
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "HIDDEN", "BC"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.hidden_dim, self.tile_Bc]
        node_data["equation"] = "O[batch][br][bc]+=I[batch][br][hidden]*W[batch][hidden][bc]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"Q_tile_{idx}", f"K_tile_{jdx}"]
        return ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
    
    def _helper_create_simd_scale_node(self, idx, jdx, id, pred_id):
        # This is the second step for the FA
        # 2. s = x/scale [Br x Bc]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"scale_i{idx}_j{jdx}"
        node_data["operator_type"] = "FA_Simd"
        node_data["operand_source"] = {"I": pred_id}
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "BC"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.tile_Bc]
        node_data["equation"] = "O[batch][br][bc]+=I[batch][br][bc]*W[]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"gemm_qk_i{idx}_j{jdx}"]
        return ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
    
    def _helper_create_compute_m_node(self, idx, jdx, id, pred_id):
        # This is the third step for the FA
        # 3. m = max_row(s) [Br x 1]
        # Pretty much a reduce 1d operator
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"compute_m_i{idx}_j{jdx}"
        node_data["operator_type"] = "FA_Simd"
        node_data["operand_source"] = {"I":pred_id}
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "BC"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.tile_Bc]
        node_data["equation"] = "O[batch][br]+=I[batch][br][bc]*W[]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"scale_i{idx}_j{jdx}"]
        return ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
    
    def _helper_create_compute_p_node(self, idx, jdx, id, pred_id_s, pred_id_m):
        # This is the fourth step for the FA
        # 4. p = exp(s - m) [Br x Bc]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"compute_p_i{idx}_j{jdx}"
        node_data["operator_type"] = "FA_Simd"
        node_data["operand_source"] = {
            "I": pred_id_s,
            "W": pred_id_m,
        }
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "BC"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.tile_Bc]
        node_data["equation"] = "O[batch][br][bc]+=I[batch][br][bc]*W[batch][br]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"scale_i{idx}_j{jdx}", f"compute_m_i{idx}_j{jdx}"]
        return ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
    
    def _helper_create_compute_l_node(self, idx, jdx, id, pred_id_input):
        # This is the fifth step for the FA
        # 5. l = sum_row(p) [Br x 1]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"compute_l_i{idx}_j{jdx}"
        node_data["operator_type"] = "FA_Simd"
        node_data["operand_source"] = {"I":pred_id_input}
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "BC"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.tile_Bc]
        node_data["equation"] = "O[batch][br]+=I[batch][br][bc]*W[]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"compute_p_i{idx}_j{jdx}"]
        return ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
    
    def _helper_create_gemm_pv_node(self, idx, jdx, id, pred_id_input, pred_id_weight):
        # This function create the gemm pv node
        # 6. o_partial = gemm(p, Vj) [Br x Hidden_Dim]
        # p: [Batch, Br, Bc]
        # Vj: [Batch, Bc, Hidden_Dim]
        # o_partial: [Batch, Br, Hidden_Dim]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"gemm_pv_i{idx}_j{jdx}"
        node_data["operator_type"] = "FA_Gemm"
        node_data["operand_source"] = {"I":pred_id_input, "W":pred_id_weight}
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "BC", "HIDDEN"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.tile_Bc, self.hidden_dim]
        node_data["equation"] = "O[batch][br][hidden]+=I[batch][br][bc]*W[batch][bc][hidden]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"compute_p_i{idx}_j{jdx}", f"V_tile_{jdx}"]
        return ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
    
    def _helper_create_simd_scale_factor_node(self, idx, jdx, id, pred_id_input):
        # This is the seventh step for the FA
        # 7. scale_factor = exp(-m) [Br x 1]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"scaling_factor_i{idx}_j{jdx}"
        node_data["operator_type"] = "FA_Simd"
        node_data["operand_source"] = {"I": pred_id_input}
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br]
        node_data["equation"] = "O[batch][br]+=I[batch][br]*W[]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"compute_m_i{idx}_j{jdx}"]
        return ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,
        )
    def _helper_create_diag_sf_node(self, idx, jdx, id, pred_id):
        # This is the diag node
        # Basically it will create a diagonal matrix from the input vector
        # for example the input with size Br*1 will create a Br*Br matrix
        # [1,2] -> [[1,0],[0,2]]
        # We need to use a concat node to pad the
        
        return DiagNode(
            node_id=id,
            node_name=f"diag_sf_i{idx}_j{jdx}",
            predecessors=[pred_id],
            input_names=[f"scaling_factor_i{idx}_j{jdx}"],
        )
    def _helper_create_update_og_node(self, idx, jdx, id, pred_id_scale_factor, pred_id_og_partial):
        # This is the eigth step for the FA
        # 8. o_updated = scale_factor [Br x Br] * o_partial [Br x Hidden_Dim]
        # A Gemm operation
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"update_og_i{idx}_j{jdx}"
        node_data["operator_type"] = "FA_Gemm"
        node_data["operand_source"] = {
            "I": pred_id_scale_factor,
            "W": pred_id_og_partial,
        }
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "D", "HIDDEN"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.tile_Br, self.hidden_dim]
        node_data["equation"] = "O[batch][br][hidden]+=I[batch][br][d]*W[batch][br][hidden]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"scaling_factor_i{idx}_j{jdx}", f"gemm_pv_i{idx}_j{jdx}"]
        return ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,
        )
    def _helper_create_update_lg_node(self, idx, jdx, id, pred_id_scale_factor, pred_id_lg_partial):
        # This is the ninth step for the FA
        # 9. lg_updated = scale_factor [Br x Br] * l_g [Br x 1]
        # A Gemm
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"update_lg_i{idx}_j{jdx}"
        node_data["operator_type"] = "FA_Gemm"
        node_data["operand_source"] = {
            "I": pred_id_scale_factor,
            "W": pred_id_lg_partial,
        }
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "BR"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.tile_Br]
        node_data["equation"] = "O[batch][br]+=I[batch][br][br]*W[batch][br]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"scaling_factor_i{idx}_j{jdx}", f"compute_l_i{idx}_j{jdx}"]
        return ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,
        )

    def _helper_create_update_mg_node(self, idx, jdx, id, pred_id):
        # This is the tenth step for the FA
        # m_g = m
        # since it is just a copy operation, we use a dummy node
        return DummyNode(
            node_id=id,
            node_name=f"update_mg_i{idx}_j{jdx}",
            predecessors=[pred_id],
        )
    def _helper_create_diag_lg_node(self, idx, id, pred_id):
        # This is the diag node for lg
        return DiagNode(
            node_id=id,
            node_name=f"diag_lg_i{idx}_j{self.Tc -1}",
            predecessors=[pred_id],
            input_names=[f"update_lg_i{idx}_j{self.Tc -1}"],
        )
    def _helper_create_rescale_o_node(self, idx, id, pred_id_lg_updated, pred_id_og_updated):
        # This is the eleventh step for the FA
        # 11. Oi = o_updated / lg_updated
        # Again we use a GeMM operation for this
        # [Br x Br] * [Br x Hidden_Dim]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"rescale_o_i{idx}"
        node_data["operator_type"] = "FA_Gemm"
        node_data["operand_source"] = {
            "I": pred_id_lg_updated,
            "W": pred_id_og_updated,
        }
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "BR","HIDDEN"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.tile_Br, self.hidden_dim]
        node_data["equation"] = "O[batch][br][hidden]+=I[batch][br][br]*W[batch][br][hidden]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"update_lg_i{idx}_j{self.Tc-1}", f"update_og_i{idx}_j{self.Tc-1}"]
        return ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,
        )

    def _helper_create_concat_o_node(self, id, pred_id_partial_o_nodes):
        # This is the final step to concatenate all the output tiles
        return ConcatNode(
            node_id=id,
            node_name=f"concat_o",
            predecessors=pred_id_partial_o_nodes,
            axis=1, # Concatenate along the sequence length axis
            input_names=[f"O_tile_{i}" for i in range(self.Tr)],
            output_shape = (self.batch, self.seq_len, self.hidden_dim),
        )

    def _util_get_and_increment_id(self):
        """Keeps track of how many nodes have been created. Returns a new id that has not been used before"""
        curr_id = self.__node_id_tracker
        self.__node_id_tracker += 1
        return curr_id

    def _util_add_node(self, node):
        """Add a node to the node_name_to_id dictionary"""
        name = node.name
        id = node.id
        self.node_name_to_id[name] = id

    def _util_get_id_from_node_name(self, name: str) -> int:
        """Get the node id from the node_name_to_id dictionary using the node name"""
        if name in self.node_name_to_id:
            return self.node_name_to_id[name]
        raise ValueError(f"Node with name {name} not found in node_name_to_id={self.node_name_to_id}")
    
    def _util_get_mapping_this_node(self, node_data: dict[str, Any]):
        default_mapping = self.all_mappings["default"]
        if node_data["name"] in self.all_mappings:
            mapping = self.all_mappings[node_data["name"]]
        elif node_data["operator_type"] in self.all_mappings:
            mapping = self.all_mappings[node_data["operator_type"]]
        else:
            mapping = default_mapping
        # If no inter/intra mapping is given: use default one
        if not mapping.intra_core_tiling:
            mapping.intra_core_tiling = default_mapping.intra_core_tiling
        if not mapping.inter_core_tiling:
            mapping.inter_core_tiling = default_mapping.inter_core_tiling
        return mapping
    # Main get_nodes function
    def get_preprocessing_nodes(self):
        """Get the preprocessing nodes for FlashAttention"""
        # Mainly the slice QKV nodes and reshape K node
        nodes = []
        for idx in range(self.Tr):
            current_id = self._util_get_and_increment_id()
            # Slice Q node
            slice_q_node = self._helper_create_slice_qkv_node(
                input_name="Q",
                idx=idx,
                node_id=current_id,
                pred_id=self.get_node_predecessors()[0], # Q input
            )
            nodes.append(slice_q_node)
            self._util_add_node(slice_q_node)
        # Reshape K node
        current_id = self._util_get_and_increment_id()
        reshape_k_node = self._helper_create_reshape_k_node(
            node_id=current_id,
            pred_id=self.get_node_predecessors()[1], # K input
        )
        nodes.append(reshape_k_node)
        self._util_add_node(reshape_k_node)
        # Slice K node
        for idx in range(self.Tc):
            current_id = self._util_get_and_increment_id()
            slice_k_node = self._helper_create_slice_qkv_node(
                input_name="K",
                idx=idx,
                node_id=current_id,
                pred_id=self._util_get_id_from_node_name("reshape_k"), # reshape K node id
            )
            nodes.append(slice_k_node)
            self._util_add_node(slice_k_node)
        for idx in range(self.Tc):
            # Slice V node
            current_id = self._util_get_and_increment_id()
            slice_v_node = self._helper_create_slice_qkv_node(
                input_name="V",
                idx=idx,
                node_id=current_id,
                pred_id=self.get_node_predecessors()[2], # V input
            )
            nodes.append(slice_v_node)
            self._util_add_node(slice_v_node)
        return nodes
    def get_compute_qkv_tile_nodes(self, idx: int, jdx: int):
        """Get the compute nodes for one tile of FlashAttention"""
        nodes = []
        # For each qkv tile, we need to create the following nodes:
        # 1. Gemm QK
        # 2. Scale
        # 3. Compute M
        # 4. Compute P
        # 5. Compute L
        # 6. Gemm PV
        # 7. Scaling Factor
        # 8. Update Og
        # 9. Update Lg
        # 10. Update Mg
        # Each of these nodes will be created using the helper functions defined above
        # The predecessor ids will be determined based on the base_ids dictionary
        
        # 1. Gemm QK
        current_id = self._util_get_and_increment_id()
        gemm_qk_node = self._helper_create_gemm_qk_node(
            id=current_id,
            pred_id_input_Qi=self._util_get_id_from_node_name(f"slice_q_{idx}"),
            pred_id_input_Kj=self._util_get_id_from_node_name(f"slice_k_{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(gemm_qk_node)
        self._util_add_node(gemm_qk_node)
        # 2. Scale
        current_id = self._util_get_and_increment_id()
        scale_node = self._helper_create_simd_scale_node(
            id=current_id,
            pred_id=self._util_get_id_from_node_name(f"gemm_qk_i{idx}_j{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(scale_node)
        self._util_add_node(scale_node)
        # 3. Compute M
        current_id = self._util_get_and_increment_id()
        compute_m_node = self._helper_create_compute_m_node(
            id=current_id,
            pred_id=self._util_get_id_from_node_name(f"scale_i{idx}_j{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(compute_m_node)
        self._util_add_node(compute_m_node)
        # 4. Compute P
        current_id = self._util_get_and_increment_id()
        compute_p_node = self._helper_create_compute_p_node(
            id=current_id,
            pred_id_s=self._util_get_id_from_node_name(f"scale_i{idx}_j{jdx}"),
            pred_id_m=self._util_get_id_from_node_name(f"compute_m_i{idx}_j{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(compute_p_node)
        self._util_add_node(compute_p_node)
        # 5. Compute L
        current_id = self._util_get_and_increment_id()
        compute_l_node = self._helper_create_compute_l_node(
            id=current_id,
            pred_id_input=self._util_get_id_from_node_name(f"compute_p_i{idx}_j{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(compute_l_node)
        self._util_add_node(compute_l_node)
        # 6. Gemm PV
        current_id = self._util_get_and_increment_id()
        gemm_pv_node = self._helper_create_gemm_pv_node(
            id=current_id,
            pred_id_input=self._util_get_id_from_node_name(f"compute_p_i{idx}_j{jdx}"),
            pred_id_weight=self._util_get_id_from_node_name(f"slice_v_{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(gemm_pv_node)
        self._util_add_node(gemm_pv_node)
        # 7. Scaling Factor
        current_id = self._util_get_and_increment_id()
        scaling_factor_node = self._helper_create_simd_scale_factor_node(
            id=current_id,
            pred_id_input=self._util_get_id_from_node_name(f"compute_m_i{idx}_j{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(scaling_factor_node)
        self._util_add_node(scaling_factor_node)
        # The diag node
        current_id = self._util_get_and_increment_id()
        diag_sf_node = self._helper_create_diag_sf_node(
            id=current_id,
            pred_id=self._util_get_id_from_node_name(f"scaling_factor_i{idx}_j{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(diag_sf_node)
        self._util_add_node(diag_sf_node)
        # 8. Update Og
        current_id = self._util_get_and_increment_id()
        update_og_node = self._helper_create_update_og_node(
            id=current_id,
            pred_id_scale_factor=self._util_get_id_from_node_name(f"diag_sf_i{idx}_j{jdx}"),
            pred_id_og_partial=self._util_get_id_from_node_name(f"gemm_pv_i{idx}_j{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(update_og_node)
        self._util_add_node(update_og_node)
        # 9. Update Lg
        current_id = self._util_get_and_increment_id()
        update_lg_node = self._helper_create_update_lg_node(
            id=current_id,
            pred_id_scale_factor=self._util_get_id_from_node_name(f"diag_sf_i{idx}_j{jdx}"),
            pred_id_lg_partial=self._util_get_id_from_node_name(f"compute_l_i{idx}_j{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(update_lg_node)
        self._util_add_node(update_lg_node)
        # 10. Update Mg
        current_id = self._util_get_and_increment_id()
        update_mg_node = self._helper_create_update_mg_node(
            id=current_id,
            pred_id=self._util_get_id_from_node_name(f"compute_m_i{idx}_j{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(update_mg_node)
        self._util_add_node(update_mg_node)
        return nodes
    def get_output_o_nodes(self):
        """Get the output nodes for one tile of FlashAttention"""
        nodes = []
        # For each output tile, we need to create the following nodes:
        # 11. Rescale O
        for idx in range(self.Tr):
            current_id = self._util_get_and_increment_id()
            diag_lg_node = self._helper_create_diag_lg_node(
                id=current_id,
                pred_id=self._util_get_id_from_node_name(f"update_lg_i{idx}_j{self.Tc -1}"),
                idx=idx,
            )
            nodes.append(diag_lg_node)
            self._util_add_node(diag_lg_node)
            current_id = self._util_get_and_increment_id()
            rescale_o_node = self._helper_create_rescale_o_node(
                id=current_id,
                pred_id_lg_updated=self._util_get_id_from_node_name(f"diag_lg_i{idx}_j{self.Tc -1}"),
                pred_id_og_updated=self._util_get_id_from_node_name(f"update_og_i{idx}_j{self.Tc -1}"),
                idx=idx,
            )
            nodes.append(rescale_o_node)
            self._util_add_node(rescale_o_node)
        # Final concat O node
        current_id = self._util_get_and_increment_id()
        concat_o_node = self._helper_create_concat_o_node(
            id=current_id,
            pred_id_partial_o_nodes=[self._util_get_id_from_node_name(f"rescale_o_i{i}") for i in range(self.Tr)],
        )
        nodes.append(concat_o_node)
        self._util_add_node(concat_o_node)
        return nodes

    def plot_dfg(self):
        """Plot the Data Flow Graph of the generated nodes"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("NetworkX or Matplotlib not found. Skipping DFG plotting.")
            return

        G = nx.DiGraph()
        id_to_name = {}

        # First pass: register all nodes
        for node in self.nodes:
            if isinstance(node, dict):
                node_id = node["id"]
                node_name = node["name"]
            else:
                node_id = node.node_id
                node_name = node.node_name
            
            id_to_name[node_id] = node_name
            G.add_node(node_name)

        # Second pass: add edges
        for node in self.nodes:
            node_name = node.node_name
            sources = node.input_operand_source.values()
            
            for src_id in sources:
                if isinstance(src_id, list): # Handle list of predecessors (e.g. Concat)
                    src_ids = src_id
                else:
                    src_ids = [src_id]
                
                for s_id in src_ids:
                    if s_id in id_to_name:
                        src_name = id_to_name[s_id]
                        G.add_edge(src_name, node_name)
                    else:
                        # External dependency
                        ext_name = f"External_{s_id}"
                        G.add_edge(ext_name, node_name)

        plt.figure(figsize=(20, 20))
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G)
            
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue", font_size=8, arrowsize=20)
        plt.title("FlashAttention DFG")
        plt.savefig("flash_attention_dfg.png")
        print("DFG plot saved to flash_attention_dfg.png")