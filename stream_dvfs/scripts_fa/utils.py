import os
import sys
# Resolve paths early
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_WORKDIR = STREAM_DVFS_DIR.parent
sys.path.append(str(STREAM_WORKDIR))
import yaml
from zigzag.utils import open_yaml
from zigzag.datatypes import Constants
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.mapping_parser import MappingParser
from stream.parser.onnx.model import ONNXModelParser
def sanity_check(workload_path, accelerator_path, mapping_path, output_yaml_path):
    # Sanity Check here
    # Running the parsing of the workload, acc, mapping and dump the workload info to yaml output
    
    # 1. Parse accelerator
    accelerator_data = open_yaml(accelerator_path)
    validator = AcceleratorValidator(accelerator_data, accelerator_path)
    accelerator_data = validator.normalized_data
    validate_success = validator.validate()
    if not validate_success:
        raise ValueError("Failed to validate user provided accelerator.")
    factory = AcceleratorFactory(accelerator_data)
    accelerator = factory.create()
    
    # 2. Parse mapping
    mapping_parser = MappingParser(mapping_path)
    all_mappings = mapping_parser.run()

    # 3. Parse ONNX model
    onnx_model_parser = ONNXModelParser(workload_path, all_mappings, accelerator)
    onnx_model_parser.run()
    workload = onnx_model_parser.workload
    
    # 4. Dump workload to yaml
    nodes_data = []
    for node in workload.node_list:
        node_data = {
            "id": getattr(node, 'id', None),  
            "name": getattr(node, 'name', None),
            "operator_type": getattr(node, 'type', None),
            "equation": getattr(getattr(node, 'equation', None),'data',None),
            "layer_dim_sizes": str(getattr(node, 'layer_dim_sizes', {})),
            "inter_core_tiling": str(getattr(node, 'inter_core_tiling', {})),
            "intra_core_tiling": str(getattr(node, 'intra_core_tiling', {})),
            "input_operand_source": str(getattr(node, 'input_operand_source', {}))
        }
        nodes_data.append(node_data)
    yaml_data = {"nodes": nodes_data}
    if not os.path.exists(os.path.dirname(output_yaml_path)):
        os.makedirs(os.path.dirname(output_yaml_path))
    with open(output_yaml_path, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            default_flow_style=False,  
            sort_keys=False,
            indent=2,
            allow_unicode=True
        )

def get_node_communication_energy(scme, target_nodes):
    print(f"DEBUG: Calculating communication energy for {len(target_nodes)} target nodes.")
    total_energy = 0
    target_nodes_set = set(target_nodes)
    
    # Map: Tensor ID -> List of (Consumer Node, Consumer Core)
    tensor_consumers = {}
    
    # helper to get core
    def get_core(n):
        alloc = n.chosen_core_allocation
        if isinstance(alloc, int):
            return scme.accelerator.get_core(alloc)
        return alloc

    # Build Consumer Map for inputs to target nodes
    for node in target_nodes:
        consumer_core = get_core(node)
        # Find predecessors
        preds = list(scme.workload.predecessors(node))
        for pred in preds:
            # Check for Output operand of pred
            if hasattr(pred, 'operand_tensors') and Constants.OUTPUT_LAYER_OP in pred.operand_tensors:
                target_tensor = pred.operand_tensors[Constants.OUTPUT_LAYER_OP]
                t_id = target_tensor.id
                if t_id not in tensor_consumers:
                    tensor_consumers[t_id] = []
                tensor_consumers[t_id].append((node, consumer_core))

    # Iterate over all unique links
    active_links = set()
    if hasattr(scme.accelerator, 'communication_manager'):
        cm = scme.accelerator.communication_manager
        # Use get_all_links if available
        if hasattr(cm, "get_all_links"):
            all_links = cm.get_all_links()
            print(f"DEBUG: CommunicationManager has {len(all_links)} total links.")
            for link in all_links:
                if link.events:
                    active_links.add(link)
        else:
            for all_links_pairs in cm.all_pair_links.values():
                for link_pair in all_links_pairs:
                    for link in link_pair:
                        if link.events:
                            active_links.add(link)
    
    print(f"DEBUG: Found {len(active_links)} active links.")
    
    for link in active_links:
        for event in link.events:
            # Case 1: Output from Target Node
            if event.tensor.origin in target_nodes_set:
                print(f"DEBUG: Output Transfer - {event.tensor} Energy: {event.energy:.4f} pJ")
                total_energy += event.energy
            # Case 2: Input to Target Node
            # We check if this tensor is one of the inputs we care about
            # AND if it is being delivered to the consumer core
            elif event.tensor.id in tensor_consumers:
                receiver_core = event.receiver
                # Check if receiver matches any consumer core
                consumers_info = tensor_consumers[event.tensor.id]
                matched_consumer = None
                for (cons_node, cons_core) in consumers_info:
                    if cons_core == receiver_core:
                        matched_consumer = cons_node
                        break
                
                if matched_consumer:
                    # Log inputs to target node
                    total_energy += event.energy
                
    return total_energy

def compare_energy(scme_fa, scme_ah):
    print("="*60)
    print("Energy Comparison: FlashAttention (Fused) vs AttentionHead (Unfused)")
    print("="*60)
    
    # 1. Flash Attention Nodes
    # Filter nodes that belong to the Flash Attention mechanism
    fa_nodes = [n for n in scme_fa.workload.node_list if "FlashAttention" in n.name]
    
    fa_onchip = sum(n.onchip_energy for n in fa_nodes)
    # Gather offchip energy from data transfers since node attribute might be 0
    fa_offchip = get_node_communication_energy(scme_fa, fa_nodes)
    
    print(f"Flash Attention (Fused Kernel):")
    print(f"  Nodes found: {len(fa_nodes)}")
    print(f"  On-Chip Energy:  {fa_onchip:,.2f} pJ")
    print(f"  Off-Chip Energy: {fa_offchip:,.2f} pJ")
    print(f"  Total Energy:    {fa_onchip + fa_offchip:,.2f} pJ")
    print("-" * 60)

    # 2. Attention Head Nodes
    # Filter nodes for the standard attention mechanism: QK^T -> Softmax -> PV
    # Excludes Projections (q/MatMul, k/MatMul, v/MatMul, o/MatMul)
    ah_target_prefixes = ["/MatMul", "/Softmax-max/", "/Softmax-exp/", "/Softmax-sum/", "/Softmax-div/", "/MatMul_1"]
    
    # We use startswith to catch the exact node names (avoiding /q/MatMul etc since they start with /q)
    # Note: "/MatMul" starts with "/" and is distinct from "/q/MatMul"
    ah_nodes = [n for n in scme_ah.workload.node_list if any(n.name.startswith(pre) for pre in ah_target_prefixes)]
    
    ah_onchip = sum(n.onchip_energy for n in ah_nodes)
    # For the attention head, since we use the lbl offchip energy attribute, we sum it directly
    ah_offchip = sum(n.offchip_energy for n in ah_nodes)

    print(f"Attention Head (Standard):")
    print(f"  Nodes found: {len(ah_nodes)}")
    print(f"  On-Chip Energy:  {ah_onchip:,.2f} pJ")
    print(f"  Off-Chip Energy: {ah_offchip:,.2f} pJ")
    print(f"  Total Energy:    {ah_onchip + ah_offchip:,.2f} pJ")
    print("=" * 60)
    
