# The flash attention parser
import logging
from enum import StrEnum
from typing import Any
from zigzag.datatypes import Constants
from zigzag.datatypes import LayerDim
from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from onnx import ModelProto, NodeProto
from zigzag.parser.onnx.utils import get_attribute_ints_with_name
from zigzag.parser.workload_factory import LayerNodeFactory
from stream.workload.mapping import InterCoreMappingAttributes
from stream.hardware.architecture.accelerator import Accelerator
from stream.onnx_utils import get_onnx_input_shapes, get_onnx_output_shapes
from stream.workload.node import Node
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.computation.computation_node import GeneratedComputationNode
# Dependency propagation nodes
from stream.workload.dependency_propagation.reshape_node import ReshapeNode
from stream.workload.dependency_propagation.slice_node import SliceNode
from stream.workload.dependency_propagation.concat_node import ConcatNode, BlockConcatNode
from stream.workload.dependency_propagation.diag_node import DiagNode
from stream.workload.dependency_propagation.transpose_node import TransposeNode
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
        self.init_set_operand_precision()
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
        self.tile_Br = get_attribute_ints_with_name("tile_Br", self.node.attribute, default=16)
        self.tile_Bc = get_attribute_ints_with_name("tile_Bc", self.node.attribute, default=16)
        self.Tr = self.seq_len // self.tile_Br # Number of row tiles, for Q and O
        self.Tc = self.seq_len // self.tile_Bc # Number of column tiles, for K and V
        # For now we assume seq_len is divisible by tile sizes
        assert self.seq_len % self.tile_Br == 0, "Sequence length must be divisible by tile size Br"
        assert self.seq_len % self.tile_Bc == 0, "Sequence length must be divisible by tile size Bc"
        # TODO 2: Handle the case where seq_len is not divisible by tile sizes
    def init_set_operand_precision(self):
        """ Set the operand precision for FlashAttention """
        act_precision: int = self.get_activation_precision()
        weight_precision: int = self.get_weight_precision()
        intermediate_output_precision: int = self.get_intermediate_output_precision()
        self.operand_precision = {
            "W": act_precision,
            "I": act_precision,
            "O_final": act_precision,
            "O": intermediate_output_precision,
        }
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
        # We use TransposeNode identity [0, 1, 2] to pass K as [Batch, Seq_Len, Hidden_Dim]
        # This aligns with Gemm QK expectation and avoids broadcasting issues
        return TransposeNode(
            node_id=node_id,
            node_name=f"reshape_k",
            predecessor=pred_id,
            permute_axes=[0, 1, 2],
            input_names=list(self.node.input[1]),
        )
    def _helper_create_slice_qkv_node(self, input_name, idx, node_id, pred_id):
        # This function create the slice qkv node
        # input names should be only 1 from ["Q", "K", "V"]
        if input_name == "Q":
            node_name = f"slice_q_i_{idx}"
            start = idx * self.tile_Br
            end = start + self.tile_Br
            axe = 1
            input_name = self.node.input[0]
            output_names = [f"Q_tile_i_{idx}"]
        if input_name == "K":
            node_name = f"slice_k_j_{idx}"
            start = idx * self.tile_Bc
            end = start + self.tile_Bc
            axe = 1
            input_name = self.node.input[1]
            output_names = [f"K_tile_{idx}" ]
        if input_name == "V":
            node_name = f"slice_v_j_{idx}"
            start = idx * self.tile_Bc
            end = start + self.tile_Bc
            axe = 1
            input_name = self.node.input[2]
            output_names = [f"V_tile_j_{idx}"]
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
        node_data["name"] = f"gemm_qk_i_{idx}_j_{jdx}"
        node_data["operator_type"] = "FA_Gemm"
        node_data["operand_source"] = {"I": pred_id_input_Qi, "W": pred_id_input_Kj}
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "HIDDEN", "BC"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.hidden_dim, self.tile_Bc]
        node_data["equation"] = "O[batch][br][bc]+=I[batch][br][hidden]*W[batch][bc][hidden]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"Q_tile_{idx}", f"K_tile_{jdx}"]
        
        node = ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
        self._util_apply_loop_offsets(node, idx, jdx)
        return node
    
    def _helper_create_simd_scale_node(self, idx, jdx, id, pred_id):
        # This is the second step for the FA
        # 2. s = x/scale [Br x Bc]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"scale_i_{idx}_j_{jdx}"
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
        input_names = [f"gemm_qk_i_{idx}_j_{jdx}"]
        node = ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
        self._util_apply_loop_offsets(node, idx, jdx)
        return node
    
    def _helper_create_compute_m_node(self, idx, jdx, id, pred_id_s, pred_id_mg):
        # This is the third step for the FA
        # 3. m = max_row(s, m_g) [Br x 1]
        # Pretty much a reduce 1d operator
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"compute_m_i_{idx}_j_{jdx}"
        node_data["operator_type"] = "FA_Simd"
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "BC"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.tile_Bc]
        node_data["dimension_relations"] = []
        if jdx == 0:
            node_data["operand_source"] = {"I": pred_id_s}
            node_data["equation"] = "O[batch][br]+=I[batch][br][bc]*W[]"
        else:
            node_data["operand_source"] = {"I": pred_id_s, "W": pred_id_mg}
            node_data["equation"] = "O[batch][br]+=I[batch][br][bc]*W[batch][br]"
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"scale_i_{idx}_j_{jdx}"]
        node = ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
        self._util_apply_loop_offsets(node, idx, jdx)
        return node
    
    def _helper_create_compute_p_node(self, idx, jdx, id, pred_id_s, pred_id_m):
        # This is the fourth step for the FA
        # 4. p = exp(s - m) [Br x Bc]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"compute_p_i_{idx}_j_{jdx}"
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
        input_names = [f"scale_i_{idx}_j_{jdx}", f"compute_m_i_{idx}_j_{jdx}"]
        node = ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
        self._util_apply_loop_offsets(node, idx, jdx)
        return node
    
    def _helper_create_compute_l_node(self, idx, jdx, id, pred_id_input):
        # This is the fifth step for the FA
        # 5. l = sum_row(p) [Br x 1]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"compute_l_i_{idx}_j_{jdx}"
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
        input_names = [f"compute_p_i_{idx}_j_{jdx}"]
        node = ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
        self._util_apply_loop_offsets(node, idx, jdx)
        return node
    
    def _helper_create_gemm_pv_node(self, idx, jdx, id, pred_id_input, pred_id_weight):
        # This function create the gemm pv node
        # 6. o_partial = gemm(p, Vj) [Br x Hidden_Dim]
        # p: [Batch, Br, Bc]
        # Vj: [Batch, Bc, Hidden_Dim]
        # o_partial: [Batch, Br, Hidden_Dim]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"gemm_pv_i_{idx}_j_{jdx}"
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
        input_names = [f"compute_p_i_{idx}_j_{jdx}", f"V_tile_{jdx}"]
        node = ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,           
        )
        self._util_apply_loop_offsets(node, idx, jdx)
        return node
    
    def _helper_create_simd_scale_factor_node(self, idx, jdx, id, pred_id_m, pred_id_mg):
        # This is the seventh step for the FA
        # 7. scale_factor = exp(m_g-m) [Br x 1]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"scaling_factor_i_{idx}_j_{jdx}"
        node_data["operator_type"] = "FA_Simd"
        node_data["loop_dims"] = ["BATCH", "BR"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br]
        node_data["operand_precision"] = self.operand_precision
        node_data["dimension_relations"] = []
        if jdx == 0:
            node_data["operand_source"] = {"I": pred_id_m}
            node_data["equation"] = "O[batch][br]+=I[batch][br]*W[]"
        else:
            node_data["operand_source"] = {
                "I": pred_id_m,
                "W": pred_id_mg,
            }
            node_data["equation"] = "O[batch][br]+=I[batch][br]*W[batch][br]"
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"compute_m_i_{idx}_j_{jdx}"]
        node = ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,
        )
        self._util_apply_loop_offsets(node, idx, jdx)
        return node
    def _helper_create_diag_sf_node(self, idx, jdx, id, pred_id):
        # This is the diag node
        # Basically it will create a diagonal matrix from the input vector
        # for example the input with size Br*1 will create a Br*Br matrix
        # [1,2] -> [[1,0],[0,2]]
        # We need to use a concat node to pad the
        
        return DiagNode(
            node_id=id,
            node_name=f"diag_sf_i_{idx}_j_{jdx}",
            predecessors=[pred_id],
            input_names=[f"scaling_factor_i_{idx}_j_{jdx}"],
        )
    def _helper_create_update_partial_og_node(self, idx, jdx, id, pred_id_scale_factor, pred_id_og):
        # This is the eigth step for the FA
        # 8. og_partial = scale_factor [Br x Br] * og [Br x Hidden_Dim]
        # A Gemm operation
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"update_partial_og_i_{idx}_j_{jdx}"
        node_data["operator_type"] = "FA_Gemm"
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "D", "HIDDEN"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.tile_Br, self.hidden_dim]
        node_data["dimension_relations"] = []
        node_data["operand_source"] = {
            "I": pred_id_scale_factor,
            "W": pred_id_og,
        }
        node_data["equation"] = "O[batch][br][hidden]+=I[batch][br][d]*W[batch][br][hidden]"
        
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"scaling_factor_i_{idx}_j_{jdx}", f"og_partial_i_{idx}_j_{jdx}"]
        node = ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,
        )
        self._util_apply_loop_offsets(node, idx, jdx)
        return node
        
    def _helper_create_update_og_node(self, idx, jdx, id, pred_id_partial_og, pred_id_o):
        # This is the eigth step for the FA
        # 8. og = og_partial[Br x Hidden_Dim] + o [Br x Hidden_Dim]
        # A Gemm operation
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"update_og_i_{idx}_j_{jdx}"
        node_data["operator_type"] = "FA_Gemm"
        node_data["operand_source"] = {
            "I": pred_id_partial_og,
            "W": pred_id_o,
        }
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "HIDDEN"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.hidden_dim]
        node_data["equation"] = "O[batch][br][hidden]+=I[batch][br][hidden]*W[batch][br][hidden]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"partial_og_i_{idx}_j_{jdx}", f"gemm_pv_i_{idx}_j_{jdx}"]
        node = ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,
        )
        self._util_apply_loop_offsets(node, idx, jdx)
        return node
        
    def _helper_create_dummy_update_og_node(self, idx, jdx, id, pred_id_o):
        # This is the dummy node for og update when j=0 such that o_g = o
        # The pred_id_o is the o input
        return DummyNode(
            node_id=id,
            node_name=f"update_og_i_{idx}_j_{jdx}",
            predecessors=[pred_id_o],
        )

    def _helper_create_update_partial_lg_node(self, idx, jdx, id, pred_id_scale_factor, pred_id_lg):
        # This is the ninth step for the FA
        # 9. lg_partial = scale_factor [Br x Br] * lg [Br x 1]
        # A Gemm
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"update_partial_lg_i_{idx}_j_{jdx}"
        node_data["operator_type"] = "FA_Gemm"
        node_data["operand_source"] = {
            "I": pred_id_scale_factor,
            "W": pred_id_lg,
        }
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR", "BR"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br, self.tile_Br]
        node_data["equation"] = "O[batch][br]+=I[batch][br][br]*W[batch][br]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"scaling_factor_i_{idx}_j_{jdx}", f"compute_l_i_{idx}_j_{jdx}"]
        node = ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,
        )
        self._util_apply_loop_offsets(node, idx, jdx)
        return node
    
    def _helper_create_update_lg_node(self, idx, jdx, id, pred_id_partial_lg, pred_id_l):
        # This is the ninth step for the FA
        # 9. lg = lg_partial[Br x 1] + l [Br x 1]
        # A Gemm
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"update_lg_i_{idx}_j_{jdx}"
        node_data["operator_type"] = "FA_Gemm"
        node_data["operand_source"] = {
            "I": pred_id_partial_lg,
            "W": pred_id_l,
        }
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "BR"]
        node_data["loop_sizes"] = [self.batch, self.tile_Br]
        node_data["equation"] = "O[batch][br]+=I[batch][br]*W[batch][br]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = [f"partial_lg_i_{idx}_j_{jdx}", f"compute_l_i_{idx}_j_{jdx}"]
        node = ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,
        )
        self._util_apply_loop_offsets(node, idx, jdx)
        return node
    
    def _helper_create_dummy_update_lg_node(self, idx, jdx, id, pred_id_l):
        # This is the dummy node for lg update when j=0 such that lg = l
        # The pred_id_l is the l input
        return DummyNode(
            node_id=id,
            node_name=f"update_lg_i_{idx}_j_{jdx}",
            predecessors=[pred_id_l],
        )
        
        
    def _helper_create_update_mg_node(self, idx, jdx, id, pred_id):
        # This is the tenth step for the FA
        # m_g = m
        # since it is just a copy operation, we use a dummy node
        return DummyNode(
            node_id=id,
            node_name=f"update_mg_i_{idx}_j_{jdx}",
            predecessors=[pred_id],
        )
    def _helper_create_diag_lg_node(self, idx, id, pred_id):
        # This is the diag node for lg
        return DiagNode(
            node_id=id,
            node_name=f"diag_lg_i_{idx}_j_{self.Tc -1}",
            predecessors=[pred_id],
            input_names=[f"update_lg_i_{idx}_j_{self.Tc -1}"],
        )
    def _helper_create_rescale_o_node(self, idx, id, pred_id_lg_updated, pred_id_og_updated):
        # This is the eleventh step for the FA
        # 11. Oi = o_updated / lg_updated
        # Again we use a GeMM operation for this
        # [Br x Br] * [Br x Hidden_Dim]
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = f"rescale_o_i"
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
        # We need to create a generated computation node here to indicate that this node is generated
        # for the concat node later
        if idx == 0:
            base_id = id
        else:
            base_id = self._util_get_id_from_node_name(f"rescale_o_i_0")
        node = GeneratedComputationNode(
            node_id=node_data["id"],
            gen_id = idx,
            gen_split_layer_dim=LayerDim("BR"),
            base_id=base_id,
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,
        )
        self._util_apply_loop_offsets(node, idx, 0)
        return node

    def _helper_create_concat_o_node(self, id, pred_id_partial_o_nodes):
        # This is the final step to concatenate all the output tiles
        return BlockConcatNode(
            node_id=id,
            node_name=f"concat_o",
            predecessors=pred_id_partial_o_nodes,
            axis=1, # Concatenate along the sequence length axis
            input_names=[f"O_tile_{i}" for i in range(self.Tr)],
            output_shape = (self.batch, self.seq_len, self.hidden_dim),
            axis_exists_in_input=True
        )

    def _helper_create_dummy_final_o_node(self, id, pred_id):
        # This is a dummy computation node to represent the final output O for the CO
        node_data: dict[str, Any] = {}
        node_data["id"] = id
        node_data["name"] = "final_o"
        node_data["operator_type"] = "FA_Gemm"
        node_data["operand_source"] = {"I": pred_id}
        node_data["operand_precision"] = self.operand_precision
        node_data["loop_dims"] = ["BATCH", "L", "HIDDEN"]
        node_data["loop_sizes"] = [self.batch, self.seq_len, self.hidden_dim]
        node_data["equation"] = "O[batch][l][hidden]+=I[batch][l][hidden]*W[]"
        node_data["dimension_relations"] = []
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        mapping_attr = self._util_get_mapping_this_node(node_data)
        input_names = ["concat_o"]
        return ComputationNode(
            node_id=node_data["id"],
            node_name=node_data["name"],
            op_type=node_data["operator_type"],
            node_attr=node_attrs,
            mapping_attr=mapping_attr,
            input_names=input_names,
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

    def _util_apply_loop_offsets(self, node: ComputationNode, idx: int, jdx: int):
        """
        Apply offsets to loop ranges of a ComputationNode based on tile indices.
        idx corresponds to BR (Row/Q sequence dim).
        jdx corresponds to BC (Col/K/V sequence dim).
        """
        br_offset = idx * self.tile_Br
        bc_offset = jdx * self.tile_Bc
        
        # We need to update loop_ranges
        # loop_ranges is a dict {loop_name: (start, end)}
        new_loop_ranges = {}
        for loop_name, (start, end) in node.loop_ranges.items():
            if str(loop_name) == "BR":
                new_loop_ranges[loop_name] = (start + br_offset, end + br_offset)
            elif str(loop_name) == "BC":
                new_loop_ranges[loop_name] = (start + bc_offset, end + bc_offset)
            else:
                new_loop_ranges[loop_name] = (start, end)
        
        node.loop_ranges = new_loop_ranges
        
        # Refresh operand tensors to reflect new loop ranges
        node.set_operand_tensors()
    # Main get_nodes function
    def get_preprocessing_nodes(self):
        """Get the preprocessing nodes for FlashAttention"""
        # Mainly the slice QKV nodes and reshape K node
        nodes = []
        # Slice Q node - SKIPPED
        
        # Reshape K node
        current_id = self._util_get_and_increment_id()
        reshape_k_node = self._helper_create_reshape_k_node(
            node_id=current_id,
            pred_id=self.get_node_predecessors()[1], # K input
        )
        nodes.append(reshape_k_node)
        self._util_add_node(reshape_k_node)
        # Slice K node - SKIPPED
        
        # Slice V node - SKIPPED

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
            pred_id_input_Qi=self.get_node_predecessors()[0], # Q Input directly
            pred_id_input_Kj=self._util_get_id_from_node_name("reshape_k"), # K Reshaped
            idx=idx,
            jdx=jdx,
        )
        nodes.append(gemm_qk_node)
        self._util_add_node(gemm_qk_node)
        # 2. Scale
        current_id = self._util_get_and_increment_id()
        scale_node = self._helper_create_simd_scale_node(
            id=current_id,
            pred_id=self._util_get_id_from_node_name(f"gemm_qk_i_{idx}_j_{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(scale_node)
        self._util_add_node(scale_node)
        # 3. Compute M
        current_id = self._util_get_and_increment_id()
        compute_m_node = self._helper_create_compute_m_node(
            id=current_id,
            pred_id_s=self._util_get_id_from_node_name(f"scale_i_{idx}_j_{jdx}"),
            pred_id_mg= 0 if jdx == 0 else self._util_get_id_from_node_name(f"update_mg_i_{idx}_j_{jdx-1}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(compute_m_node)
        self._util_add_node(compute_m_node)
        # 4. Compute P
        current_id = self._util_get_and_increment_id()
        compute_p_node = self._helper_create_compute_p_node(
            id=current_id,
            pred_id_s=self._util_get_id_from_node_name(f"scale_i_{idx}_j_{jdx}"),
            pred_id_m=self._util_get_id_from_node_name(f"compute_m_i_{idx}_j_{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(compute_p_node)
        self._util_add_node(compute_p_node)
        # 5. Compute L
        current_id = self._util_get_and_increment_id()
        compute_l_node = self._helper_create_compute_l_node(
            id=current_id,
            pred_id_input=self._util_get_id_from_node_name(f"compute_p_i_{idx}_j_{jdx}"),
            idx=idx,
            jdx=jdx,
        )
        nodes.append(compute_l_node)
        self._util_add_node(compute_l_node)
        # 6. Gemm PV
        current_id = self._util_get_and_increment_id()
        gemm_pv_node = self._helper_create_gemm_pv_node(
            id=current_id,
            pred_id_input=self._util_get_id_from_node_name(f"compute_p_i_{idx}_j_{jdx}"),
            pred_id_weight=self.get_node_predecessors()[2], # V Input directly
            idx=idx,
            jdx=jdx,
        )
        nodes.append(gemm_pv_node)
        self._util_add_node(gemm_pv_node)

        if jdx==0:
            # For j=0, only simple update and do not need to compute scaling factor
            # og = o
            current_id = self._util_get_and_increment_id()
            dummy_og_node = self._helper_create_dummy_update_og_node(
                id=current_id,
                pred_id_o=self._util_get_id_from_node_name(f"gemm_pv_i_{idx}_j_{jdx}"),
                idx=idx,
                jdx=jdx,
            )
            nodes.append(dummy_og_node)
            self._util_add_node(dummy_og_node)
            # lg = l
            current_id = self._util_get_and_increment_id()
            dummy_lg_node = self._helper_create_dummy_update_lg_node(
                id=current_id,
                pred_id_l=self._util_get_id_from_node_name(f"compute_l_i_{idx}_j_{jdx}"),
                idx=idx,
                jdx=jdx,
            )
            nodes.append(dummy_lg_node)
            self._util_add_node(dummy_lg_node)
        else:
            # we need to create the scaling factor to update og and lg
            # 7. Scaling Factor
            current_id = self._util_get_and_increment_id()
            scaling_factor_node = self._helper_create_simd_scale_factor_node(
                id=current_id,
                pred_id_m=self._util_get_id_from_node_name(f"compute_m_i_{idx}_j_{jdx}"),
                pred_id_mg=self._util_get_id_from_node_name(f"update_mg_i_{idx}_j_{jdx-1}"),
                idx=idx,
                jdx=jdx,
            )
            nodes.append(scaling_factor_node)
            self._util_add_node(scaling_factor_node)
            # The diag node for SF
            current_id = self._util_get_and_increment_id()
            diag_sf_node = self._helper_create_diag_sf_node(
                id=current_id,
                pred_id=self._util_get_id_from_node_name(f"scaling_factor_i_{idx}_j_{jdx}"),
                idx=idx,
                jdx=jdx,
            )
            nodes.append(diag_sf_node)
            self._util_add_node(diag_sf_node)
            # 8. Update Og
            # we first create the update partial og node to compute the sf*og_partial
            current_id = self._util_get_and_increment_id()
            update_og_node = self._helper_create_update_partial_og_node(
                id=current_id,
                pred_id_scale_factor=self._util_get_id_from_node_name(f"diag_sf_i_{idx}_j_{jdx}"),
                pred_id_og=self._util_get_id_from_node_name(f"update_og_i_{idx}_j_{jdx-1}"),
                idx=idx,
                jdx=jdx,
            )
            nodes.append(update_og_node)
            self._util_add_node(update_og_node)
            # then the update og node
            current_id = self._util_get_and_increment_id()
            update_og_node = self._helper_create_update_og_node(
                id=current_id,
                pred_id_partial_og=self._util_get_id_from_node_name(f"update_partial_og_i_{idx}_j_{jdx}"),
                pred_id_o=self._util_get_id_from_node_name(f"gemm_pv_i_{idx}_j_{jdx}"),
                idx=idx,
                jdx=jdx,
            )
            nodes.append(update_og_node)
            self._util_add_node(update_og_node)
            # 9. Update Lg
            current_id = self._util_get_and_increment_id()
            update_lg_node = self._helper_create_update_partial_lg_node(
                id=current_id,
                pred_id_scale_factor=self._util_get_id_from_node_name(f"diag_sf_i_{idx}_j_{jdx}"),
                pred_id_lg=self._util_get_id_from_node_name(f"update_lg_i_{idx}_j_{jdx-1}"),
                idx=idx,
                jdx=jdx,
            )
            nodes.append(update_lg_node)
            self._util_add_node(update_lg_node)
            # then the update lg node
            current_id = self._util_get_and_increment_id()
            update_lg_node = self._helper_create_update_lg_node(
                id=current_id,
                pred_id_partial_lg=self._util_get_id_from_node_name(f"update_partial_lg_i_{idx}_j_{jdx}"),
                pred_id_l=self._util_get_id_from_node_name(f"compute_l_i_{idx}_j_{jdx}"),
                idx=idx,
                jdx=jdx,
            )
            nodes.append(update_lg_node)
            self._util_add_node(update_lg_node)
        # 10. Update Mg
        # always mg=m
        current_id = self._util_get_and_increment_id()
        update_mg_node = self._helper_create_update_mg_node(
            id=current_id,
            pred_id=self._util_get_id_from_node_name(f"compute_m_i_{idx}_j_{jdx}"),
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
                pred_id=self._util_get_id_from_node_name(f"update_lg_i_{idx}_j_{self.Tc -1}"),
                idx=idx,
            )
            nodes.append(diag_lg_node)
            self._util_add_node(diag_lg_node)
            current_id = self._util_get_and_increment_id()
            rescale_o_node = self._helper_create_rescale_o_node(
                id=current_id,
                pred_id_lg_updated=self._util_get_id_from_node_name(f"diag_lg_i_{idx}_j_{self.Tc -1}"),
                pred_id_og_updated=self._util_get_id_from_node_name(f"update_og_i_{idx}_j_{self.Tc -1}"),
                idx=idx,
            )
            nodes.append(rescale_o_node)
            self._util_add_node(rescale_o_node)
        # Concat O node
        current_id = self._util_get_and_increment_id()
        concat_o_node = self._helper_create_concat_o_node(
            id=current_id,
            pred_id_partial_o_nodes=[self._util_get_id_from_node_name(f"rescale_o_i_{idx}") for idx in range(self.Tr)],
        )
        nodes.append(concat_o_node)
        self._util_add_node(concat_o_node)
        # # Final dummy O computation node
        # current_id = self._util_get_and_increment_id()
        # final_o_node = self._helper_create_dummy_final_o_node(
        #     id=current_id,
        #     pred_id=self._util_get_id_from_node_name("concat_o"),
        # )
        # nodes.append(final_o_node)
        # self._util_add_node(final_o_node)
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
        