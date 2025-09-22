import logging
from operator import itemgetter
from typing import TYPE_CHECKING

from zigzag.datatypes import Constants, LayerOperand, MemoryOperand

from stream.hardware.architecture.core import Core
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.workload.tensor import Tensor

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator

logger = logging.getLogger(__name__)


def initialize_workload_accelerator(workload: ComputationNodeWorkload, accelerator: "Accelerator"):
    for n in workload.node_list:
        n.start = None
        n.end = None
    accelerator.clean_accelerator()

def initialize_priorities(workload: ComputationNodeWorkload, accelerator: "Accelerator"):
    for n in workload.node_list:
        for tensor in n.operand_tensors.values():
            tensor.initialize_instance_priorities(workload, n, accelerator)


def initialize_offchip_tensors(workload: ComputationNodeWorkload, accelerator: "Accelerator"):
    offchip_core_id = accelerator.offchip_core_id
    assert offchip_core_id is not None, "No offchip core found for this accelerator"
    offchip_core = accelerator.get_core(offchip_core_id)
    offchip_top_instances = accelerator.get_top_instances_of_core(offchip_core_id)
    for n in workload.node_list:
        for op, tensor in n.operand_tensors.items():
            # For constant operands or inputs of first node
            if op in n.constant_operands or (op != Constants.OUTPUT_LAYER_OP and len(workload.in_edges(n)) == 0):
                if not any(
                    (
                        accelerator.contains_tensor(tensor, offchip_top_instance)
                        for offchip_top_instance in offchip_top_instances
                    )
                ):
                    memory_op = n.memory_operand_links.layer_to_mem_op(op)
                    accelerator.spawn(
                        tensor=tensor,
                        core=offchip_core,
                        memory_op=memory_op,
                        initial_timestep=0,
                        available_timestep=0,
                    )


def prefetch_constant_operands(
    G: ComputationNodeWorkload, accelerator: "Accelerator", operands_to_prefetch: list[LayerOperand]
):
    total_cn_offchip_link_energy = 0
    total_cn_offchip_memory_energy = 0
    total_eviction_to_offchip_link_energy = 0
    total_eviction_to_offchip_memory_energy = 0
    for n in G.node_list:
        for op, tensor in n.operand_tensors.items():
            if op in n.constant_operands and op in operands_to_prefetch:
                core_allocation = n.chosen_core_allocation
                assert core_allocation is not None, "Core should be allocated"
                memory_op = n.memory_operand_links.layer_to_mem_op(op)
                if not accelerator.contains_tensor(tensor, core_allocation):
                    (
                        _,
                        transfer_link_energy_cost,
                        transfer_memory_energy_cost,
                        eviction_link_energy_cost,
                        eviction_memory_energy_cost,
                        came_from_offchip,
                    ) = accelerator.transfer_tensor_to_core(tensor, core_allocation, memory_op, [])
                    assert came_from_offchip
                    total_cn_offchip_link_energy += transfer_link_energy_cost
                    total_cn_offchip_memory_energy += transfer_memory_energy_cost
                    total_eviction_to_offchip_link_energy += eviction_link_energy_cost
                    total_eviction_to_offchip_memory_energy += eviction_memory_energy_cost
    return (
        total_cn_offchip_link_energy,
        total_cn_offchip_memory_energy,
        total_eviction_to_offchip_link_energy,
        total_eviction_to_offchip_memory_energy,
    )


def get_best_candidate(
    candidates: list[tuple[int, ComputationNode]], scheduling_order: list[tuple[int, int]]
) -> tuple[ComputationNode, int]:
    # If this core doesn't have any candidates, continue to the next core
    if not candidates:
        raise ValueError("There are no candidates to schedule.")
    preds_ends, cn_candidates = zip(*candidates)
    cn_candidates: list[ComputationNode]
    idxs = [scheduling_order.index((n.id, n.sub_id)) for n in cn_candidates]
    best_candidate_idx = idxs.index(min(idxs))
    best_candidate = cn_candidates[best_candidate_idx]
    preds_end = preds_ends[best_candidate_idx]
    # Remove the candidate from the list of candidates
    del candidates[best_candidate_idx]
    return best_candidate, preds_end


def get_tensors_needed_for_node(node: ComputationNode, G: ComputationNodeWorkload):
    """Determine all the tensors needed to compute a node.
    The node might need multiple outputs from previous nodes, depending on the graph.

    Args:
        node (ComputationNode): The node to be computed.
        G : The graph of all nodes.

    Returns:
        tuple: A tuple of tensors and a tuple of memory operands for the node.
    """
    tensors_this_candidate_needs: list[Tensor] = []
    tensors_operands: list[MemoryOperand] = []
    # Constant operands
    for layer_op in node.constant_operands:
        memory_op = node.memory_operand_links.layer_to_mem_op(layer_op)
        if memory_op in node.too_large_operands:
            continue
        tensors_this_candidate_needs.append(node.operand_tensors[layer_op])
        tensors_operands.append(memory_op)
    # Non-constant operands
    for pred, node, edge_data in sorted(G.in_edges(node, data=True), key=itemgetter(0)):
        if pred.id == node.id:
            continue  # Skip if predecessor was from the same layer (intra-edge)
        consumer_layer_op = edge_data["operand"]
        consumer_memory_op = node.memory_operand_links[consumer_layer_op]
        if consumer_memory_op in node.too_large_operands:
            continue  # Skip if tensor will be fetched fromm offchip throughout computation
        pred_output_tensor = pred.operand_tensors[pred.output_operand]
        tensors_this_candidate_needs.append(pred_output_tensor)
        tensors_operands.append(consumer_memory_op)
    if tensors_this_candidate_needs:
        # Sort these tensors based on their earliest possible transfer time
        tensors_this_candidate_needs, tensors_operands = zip(
            *sorted(zip(tensors_this_candidate_needs, tensors_operands))
        )
    return tensors_this_candidate_needs, tensors_operands


def clear_memories(
    accelerator: "Accelerator",
    core: Core,
    memory_operands: list[MemoryOperand],
    timestep: int,
    exceptions: list[Tensor] = [],
):
    total_eviction_to_offchip_link_energy = 0
    total_eviction_to_offchip_memory_energy = 0
    for too_large_operand in memory_operands:
        (
            timestep,
            eviction_link_energy_cost,
            eviction_memory_energy_cost,
        ) = accelerator.remove_all(core, too_large_operand, timestep, exceptions, write_back_to_offchip=True)
        total_eviction_to_offchip_link_energy += eviction_link_energy_cost
        total_eviction_to_offchip_memory_energy += eviction_memory_energy_cost
    return (
        total_eviction_to_offchip_link_energy,
        total_eviction_to_offchip_memory_energy,
        timestep,
    )


def decrease_priority(
    tensors: list[Tensor],
    tensors_operands: list[MemoryOperand],
    accelerator: "Accelerator",
    node: ComputationNode,
):
    for tensor_used_by_node, tensor_memory_operand in zip(tensors, tensors_operands):
        # TODO: tensor_memory_operand will be 'O' for activation tensors.
        # TODO: If the memory between input and output is not shared, this will give a wrong instance.
        assert node.chosen_core_allocation is not None
        top_instance = accelerator.get_top_instance_of_core(node.chosen_core_allocation, tensor_memory_operand)
        tensor_used_by_node.instance_priorities[top_instance] -= 1


def check_for_removal(
    tensors: list[Tensor],
    accelerator: "Accelerator",
    node: ComputationNode,
    G: ComputationNodeWorkload,
    timestep: int,
):
    offchip_core_id = accelerator.offchip_core_id
    for tensor_used_by_node in tensors:
        if tensor_used_by_node.get_total_priority() == 0:
            instances_storing_tensor, _ = accelerator.memory_manager.find_tensor_in_top_instances(tensor_used_by_node)
            for instance_storing_tensor in instances_storing_tensor:
                core_ids_of_instance = [
                    core.id for core in accelerator.memory_manager.cores_per_top_instance[instance_storing_tensor]
                ]
                # If this tensor is an output tensor, find all nodes that needed it
                # to get an accurate timestep at which it can be removed
                timestep_for_removal = timestep
                if tensor_used_by_node.layer_operand == tensor_used_by_node.origin.output_operand:
                    origin = tensor_used_by_node.origin
                    if offchip_core_id in core_ids_of_instance:
                        # If wanting to discard it from offchip, look at the max end time across all successors
                        nodes_that_needed_tensor = [n for n in G.successors(origin) if n.id != origin.id]
                    else:
                        # If discarding it from a regular core, look at the max end time successors that used it from
                        # that instance
                        nodes_that_needed_tensor = [
                            n
                            for n in G.successors(origin)
                            if n.chosen_core_allocation in core_ids_of_instance and n.id != origin.id
                        ]
                    end_times = [n.end for n in nodes_that_needed_tensor if n.end is not None]
                    max_end_time = max(end_times, default=timestep_for_removal)
                    # assert max_end_time != -1, "There should be at least one successor."
                    timestep_for_removal = max_end_time

                # Get a core tied to the top_instance we want to remove it on.
                core = accelerator.memory_manager.cores_per_top_instance[instance_storing_tensor][0]
                accelerator.remove(
                    tensor_used_by_node,
                    core,
                    tensor_used_by_node.memory_operand,
                    timestep_for_removal,
                )


def set_node_dvfs(
    node: "ComputationNode",
    system_clock_freq_ghz: float,
    dvfs_switching_latency_ms: float,
    dvfs_allocations: dict[int, dict[int, int]]
):
    """
    Compute post-DVFS runtime in baseline (nominal) cycles by stepping through DVFS windows.
    - We treat time on a baseline time axis (f_scale = 1.0), so 'current_time_cc' and 'dvfs_window_cc'
      are measured in baseline cycles.
    - 'nominal_remaining' is the fixed amount of productive work measured in nominal cycles.
    - In a window with frequency ratio f_scale, consuming 'x' nominal cycles requires 'x/f_scale'
      baseline cycles of wall time; conversely, a window with 'W' baseline cycles can consume at most
      'W * f_scale' nominal cycles. 
    e.g. 100 CC @ 1Ghz == 200 CC @ 0.5Ghz
         here the f_scale = 0.5 and is calculated as dvfs_freq / nominal_freq
         so in order to get the nominal cycles that can be consumed in this time window
         we need to multiply the wall-clock cycles by the f_scale
    """

    # Window size in baseline cycles (baseline time axis at f_scale=1)
    dvfs_window_cc = int(dvfs_switching_latency_ms * system_clock_freq_ghz * 1e6)  # ms * GHz * 1e6 = cycles

    core_id = node.chosen_core_allocation
    assert core_id is not None, "Core should be allocated"
    dvfs_allocations_for_core = dvfs_allocations[core_id]

    # Baseline time axis and nominal work
    current_time_cc = node.get_start()            # baseline cycles
    nominal_remaining = node.get_runtime()  # nominal cycles (productive work at f_scale=1)

    dvfs_runtime_cc = 0           # accumulated wall time in baseline cycles
    segment_levels: list[int] = [] 
    segment_durations_cc: list[int] = []  # per-segment wall time in baseline cycles

    while nominal_remaining > 0:
        time_window_id = int(current_time_cc // dvfs_window_cc)
        dvfs_lvl = dvfs_allocations_for_core[time_window_id]
        f_scale = node.freq_lut[dvfs_lvl]  # frequency ratio (0 < f_scale <= 1]

        # Remaining baseline wall time capacity in this window
        window_end_cc = (time_window_id + 1) * dvfs_window_cc
        cc_left_in_window = window_end_cc - current_time_cc

        # Max nominal work this window can consume
        max_nominal_this_window = cc_left_in_window * f_scale

        # Decide how much nominal work to consume in this window
        consumed_nominal = min(nominal_remaining, max_nominal_this_window)

        # Corresponding baseline wall time for that work
        consumed_wall_exact = consumed_nominal / f_scale

        # Control rounding: keep integer baseline cycles but avoid negative remainder
        consumed_wall_cc = int(consumed_wall_exact)
        if consumed_wall_cc == 0 and consumed_nominal > 0:
            # Ensure forward progress when small fractions appear
            consumed_wall_cc = 1

        # Record segment
        segment_levels.append(dvfs_lvl)
        segment_durations_cc.append(consumed_wall_cc)
        dvfs_runtime_cc += consumed_wall_cc

        # Update state
        nominal_remaining -= consumed_nominal
        current_time_cc += consumed_wall_cc

    node.set_dvfs_levels(segment_levels)
    node.set_dvfs_runtime(dvfs_runtime_cc)
    node.set_dvfs_window_duration(segment_durations_cc)

    # Dynamic energy: time-weighted scaling over segments using dyn_energy_lut
    total_energy = 0.0
    total_cc = sum(segment_durations_cc)
    for lvl, dur_cc in zip(segment_levels, segment_durations_cc):
        time_share = dur_cc / total_cc
        e_scale = node.get_energy_lut().get(lvl, 1.0)
        total_energy += node.get_onchip_energy() * e_scale * time_share

    node.set_onchip_dvfs_energy(total_energy)
    
def accumulate_core_leakage_energy(
    latency_cc: int,
    system_clock_freq_ghz: float,
    dvfs_switching_latency_ms: float,
    dvfs_allocations: dict[int, dict[int, int]] | None,
    sta_energy_lut: dict[int, float],
    static_power_per_core_uW: dict[int, float] | None = None,
    default_core_uW: float = 100.0,
    default_dvfs_level: int = 0,
) -> float:
    """
    Compute total leakage energy (pJ) across all cores by integrating static power over
    DVFS windows up to makespan (latency_cc).

    Fallback rules:
      - If dvfs_allocations is None (no DVFS), use default_level over the entire makespan for all known cores.
      - If a core/window level is missing, fall back to default_level.

    Returns:
      Total leakage energy in picojoules (pJ).
    """
    # Validate makespan
    if latency_cc <= 0:
        return 0.0

    # Baseline frequency and window size (cycles)
    f_nom_hz = float(system_clock_freq_ghz) * 1e9
    dvfs_window_cc = int(dvfs_switching_latency_ms * system_clock_freq_ghz * 1e6)
    if dvfs_window_cc <= 0:
        dvfs_window_cc = 1

    static_power_per_core_uW = static_power_per_core_uW or {}

    total_E_leak_J = 0.0

    # Determine cores to process
    core_ids: set[int] = set()
    if dvfs_allocations is not None:
        core_ids |= set(int(c) for c in dvfs_allocations.keys())
    # Also include cores that have a static power entry even if no dvfs_allocations
    core_ids |= set(int(c) for c in static_power_per_core_uW.keys() if isinstance(c, int) and c >= 0)

    if not core_ids:
        return 0.0

    # Last window index (inclusive) overlapping [0, latency_cc)
    last_idx_inclusive = (int(latency_cc) - 1) // dvfs_window_cc

    for core_id in sorted(core_ids):
        # Nominal per-core static power (W); allow a shared default via key -1
        P_nom_uW = static_power_per_core_uW.get(core_id, static_power_per_core_uW.get(-1, default_core_uW))
        P_nom_W = float(P_nom_uW) * 1e-6
        if P_nom_W <= 0.0:
            continue

        # Case A: no DVFS allocation provided -> use default_level across full makespan
        if dvfs_allocations is None:
            sta_scale = float(sta_energy_lut.get(default_dvfs_level, 1.0))
            dt_s = int(latency_cc) / f_nom_hz
            total_E_leak_J += (P_nom_W * sta_scale) * dt_s
            continue

        # Case B: DVFS provided -> step windows, fallback to default_level when missing
        tw_map = dvfs_allocations.get(core_id, {})
        for j in range(0, last_idx_inclusive + 1):
            lvl = int(tw_map.get(j, default_dvfs_level))
            sta_scale = float(sta_energy_lut.get(lvl, 1.0))

            # Window [t0, t1) and overlap with [0, latency_cc)
            t0 = j * dvfs_window_cc
            t1 = (j + 1) * dvfs_window_cc
            eff_start = max(0, t0)
            eff_end = min(int(latency_cc), t1)
            eff_cc = eff_end - eff_start
            if eff_cc <= 0:
                continue

            dt_s = eff_cc / f_nom_hz
            total_E_leak_J += (P_nom_W * sta_scale) * dt_s

    # Return in picojoules
    return total_E_leak_J * 1e12
def schedule_graph(
    G: ComputationNodeWorkload,
    accelerator: "Accelerator",
    cores_idle_from: dict[int, int] | None = None,
    operands_to_prefetch: list[LayerOperand] = [],
    scheduling_order: list[tuple[int, int]] | None = None,
    system_clock_freq: float = 1.0,        # in GHz
    dvfs_switching_latency: float = 1.0,   # in ms
    dvfs_allocations: dict[int, dict[int, int]] | None = None,
) -> tuple[int, float, float, float, float, float, float, float, float, float]:
    """Schedule the nodes of graph G across the cores in the system.
    Each node should have a core_allocation and runtime set.

    Args:
        G : Graph containing the nodes to be scheduled.
        accelerator (Accelerator): The accelerator to schedule the nodes on.
        cores_start_offset (dict, optional): A dict containing for each core_id its start offset. Defaults to None.
        operands_to_prefetch (list, optional): The layer operands that should be prefetched at the start of the
            schedule.
    """
    # Initialize total link energy cost and memory energy costs
    total_cn_onchip_energy = 0
    total_cn_offchip_link_energy = 0
    total_cn_offchip_memory_energy = 0
    total_eviction_to_offchip_link_energy = 0
    total_eviction_to_offchip_memory_energy = 0
    total_sink_layer_output_offchip_link_energy = 0
    total_sink_layer_output_offchip_memory_energy = 0
    total_core_to_core_link_energy = 0
    total_core_to_core_memory_energy = 0

    core_ids = set(n.chosen_core_allocation for n in G.node_list)
    assert (
        None not in core_ids
    ), "Make sure all nodes have a core allocation. Insert SetFixedAllocationPerformanceStage."
    all_core_ids: list[int] = sorted(list(core_ids))  # type: ignore

    if cores_idle_from is None:
        # Make it 0 for all cores
        cores_idle_from = {core_allocation: 0 for core_allocation in all_core_ids}

    nb_graph_nodes = G.number_of_nodes()
    nb_scheduled_nodes = 0
    scheduled_nodes: set[ComputationNode] = set()

    # List that keeps all possible candidate nodes for each core.
    candidates: list[tuple[int, ComputationNode]] = []

    # Put the very first nodes of a layer that doesn't have any incoming edges as the first candidates
    for source_node in (n for n, d in G.in_degree() if d == 0):
        core_allocation = source_node.chosen_core_allocation
        candidates.append((cores_idle_from[core_allocation], source_node))  # type: ignore

    # Get all the nodes with no successors that produce final outputs, used for off-loading final outputs
    sink_layers = sorted(set(n.id for n, d in G.out_degree() if d == 0))
    sink_layer_nodes = set((n for n in G.node_list if (n.id in sink_layers) and n.produces_final_output))

    # Get the offchip core id and core
    offchip_core_id = accelerator.offchip_core_id
    assert offchip_core_id is not None
    offchip_core = accelerator.get_core(offchip_core_id)

    # Schedule preparation:
    initialize_workload_accelerator(G,accelerator)
    # 1. Initialize the memory instance priorities for each tensor
    initialize_priorities(G, accelerator)
    # 2. Add the constant operand tensors of all nodes to the off-chip initially
    initialize_offchip_tensors(G, accelerator)
    # 3. Prefetch the constant operands that should be prefetched to their core
    (
        prefetch_cn_offchip_link_energy,
        prefetch_cn_offchip_memory_energy,
        prefetch_eviction_to_offchip_link_energy,
        prefetch_eviction_to_offchip_memory_energy,
    ) = prefetch_constant_operands(G, accelerator, operands_to_prefetch)
    total_cn_offchip_link_energy += prefetch_cn_offchip_link_energy
    total_cn_offchip_memory_energy += prefetch_cn_offchip_memory_energy
    total_eviction_to_offchip_link_energy += prefetch_eviction_to_offchip_link_energy
    total_eviction_to_offchip_memory_energy += prefetch_eviction_to_offchip_memory_energy

    done = False
    while not done:
        # Get the best candidate given the selection priority
        best_candidate, preds_end = get_best_candidate(candidates, scheduling_order)

        # Get the core this candidate will be scheduled on
        core_id = best_candidate.chosen_core_allocation
        assert core_id is not None
        core = accelerator.get_core(core_id)
        # Earliest start time is when core is available or predecessors finished
        start = max(cores_idle_from[core_id], preds_end)
        # Step 0
        tensors_this_candidate_needs, tensors_operands = get_tensors_needed_for_node(best_candidate, G)
        # Step 1
        # There could be operands that are too large to store in the highest memory on the core
        # The tensors stored in these memories should be evicted and potentially written back to off-chip
        # Clear these memories (this might delay the potential start time if things have to written to off-chip)
        timestep = start
        (
            clear_link_energy,
            clear_memory_energy,
            timestep,
        ) = clear_memories(
            accelerator,
            core,
            best_candidate.too_large_operands,
            timestep,
            exceptions=tensors_this_candidate_needs,
        )
        total_eviction_to_offchip_link_energy += clear_link_energy
        total_eviction_to_offchip_memory_energy += clear_memory_energy
        # Step 2
        # The computation might need tensors that are currently not present in the core's memories
        # We need to fetch these tensors from either off-chip or from the core where they are present
        # Transfer these tensors from wherever they are currently residing to this core
        for tensor, tensor_operand in zip(tensors_this_candidate_needs, tensors_operands):
            # Transfer the tensor
            (
                transfer_complete_timestep,
                transfer_link_energy_cost,
                transfer_memory_energy_cost,
                eviction_link_energy_cost,
                eviction_memory_energy_cost,
                came_from_offchip,
            ) = accelerator.transfer_tensor_to_core(
                tensor,
                core_id,
                tensor_operand,
                tensors_this_candidate_needs,
            )
            # Update the possible start time of this node
            timestep = max(timestep, transfer_complete_timestep)
            # Add the energy costs to their respective trackers
            if came_from_offchip:
                total_cn_offchip_link_energy += transfer_link_energy_cost
                total_cn_offchip_memory_energy += transfer_memory_energy_cost
            else:
                total_core_to_core_link_energy += transfer_link_energy_cost
                total_core_to_core_memory_energy += transfer_memory_energy_cost
            total_eviction_to_offchip_link_energy += eviction_link_energy_cost
            total_eviction_to_offchip_memory_energy += eviction_memory_energy_cost

        # Step 3
        # Check if we had any operands that were too large to store in the core's memory, block the relevant off-chip
        # link for the duration
        # This might again delay the execution if the offchip link was already blocked by another core
        
        timestep = accelerator.block_offchip_links(
            best_candidate.too_large_operands,
            core_id,
            timestep,
            best_candidate.get_runtime(),
            best_candidate,
        )

        # Step 4
        # Make space for the output tensor of this computation node and spawn it when evictions are complete
        # If the output operand is in the too large operands, add it to off-chip, otherwise add it to this core's
        # output memory
        output_layer_operand = best_candidate.output_operand
        output_memory_operand = best_candidate.memory_operand_links[output_layer_operand]
        output_tensor = best_candidate.operand_tensors[output_layer_operand]
        if output_memory_operand in best_candidate.too_large_operands:
            core_to_add_output_to = offchip_core
        else:
            core_to_add_output_to = core
        (
            evictions_complete_timestep,
            eviction_link_energy_cost,
            eviction_memory_energy_cost,
        ) = accelerator.make_space_for(
            output_tensor,
            core_to_add_output_to,
            output_memory_operand,
            timestep,
            tensors_this_candidate_needs,
        )
        total_eviction_to_offchip_link_energy += eviction_link_energy_cost
        total_eviction_to_offchip_memory_energy += eviction_memory_energy_cost
        start = evictions_complete_timestep
        best_candidate.set_start(start)
        # Here we use the dvfs_allocation for this core at the time window in which the node starts
        # to retrieve the dvfs_level for the current node
        if dvfs_allocations is not None:
            set_node_dvfs(
                best_candidate,
                system_clock_freq,
                dvfs_switching_latency,
                dvfs_allocations
            )
        # Now that we have the dvfs level, we can get the actual runtime of this node
        if dvfs_allocations is not None:
            runtime = best_candidate.get_dvfs_runtime()
        else:
            runtime = best_candidate.get_runtime()
        end = start + runtime
        accelerator.spawn(
            output_tensor,
            core_to_add_output_to,
            output_memory_operand,
            initial_timestep=start,
            available_timestep=end,
        )

        # Step 5
        # Update the start and end time of the node
        best_candidate.set_end(end)
        cores_idle_from[core_id] = end

        # Add the computation energy of running this node
        if dvfs_allocations is not None:
            on_chip_energy = best_candidate.get_onchip_dvfs_energy()
        else:
            on_chip_energy = best_candidate.get_onchip_energy()
        total_cn_onchip_energy += on_chip_energy
        total_cn_offchip_memory_energy += best_candidate.get_offchip_energy()

        # Add this node to the scheduled nodes
        scheduled_nodes.add(best_candidate)

        # Step 6
        # Memory usage: When the node ends:
        # Decrease the priority of all the tensors this node used
        decrease_priority(tensors_this_candidate_needs, tensors_operands, accelerator, best_candidate)
        # Remove the tensor if the priority is zero
        check_for_removal(
            tensors_this_candidate_needs,
            accelerator,
            best_candidate,
            G,
            end,
        )

        # Step 7
        # Memory usage: When the node ends:
        # If this node is a sink node (node that has no successors and that produces a final output), transfer final
        # outputs to offchip
        if best_candidate in sink_layer_nodes:
            # Only push back sink node outputs if they're generated and stored on the core
            if best_candidate.output_operand not in best_candidate.too_large_operands:
                (
                    _,
                    link_energy_cost,
                    memory_energy_cost,
                ) = accelerator.remove(
                    output_tensor,
                    core,
                    output_tensor.memory_operand,
                    end,
                    write_back_to_offchip=True,
                )
                total_sink_layer_output_offchip_link_energy += link_energy_cost
                total_sink_layer_output_offchip_memory_energy += memory_energy_cost

        # Step 8
        # For each successor of this node, check if all of its predecessors have been scheduled
        for successor in sorted(G.successors(best_candidate)):
            if all((pred in scheduled_nodes for pred in G.predecessors(successor))):
                preds_end = max(
                    (predecessor.end for predecessor in G.predecessors(successor)),
                    default=0,
                )
                # core_candidates[successor.core_allocation].append((preds_end, successor))
                candidates.append((preds_end, successor))

        # Increment the number of scheduled nodes
        nb_scheduled_nodes += 1
        done = nb_scheduled_nodes == nb_graph_nodes

    # Step 9
    # The total schedule latency is the max of all CN end times and the link end times
    cns_end_time = max((n.end for n in G.node_list))
    links_end_time = max([event.end for event in accelerator.communication_manager.events], default=0)
    latency = max(cns_end_time, links_end_time)
    # Add the leakage energy of all cores across the entire schedule

    if dvfs_allocations is not None:
        total_leakage_energy = accumulate_core_leakage_energy(
            latency_cc=int(latency),
            system_clock_freq_ghz=system_clock_freq,
            dvfs_switching_latency_ms=dvfs_switching_latency,
            dvfs_allocations=dvfs_allocations,
            sta_energy_lut=accelerator.get_sta_energy_lut(),
            static_power_per_core_uW=accelerator.get_sta_power_per_core_uW(),
        )
    else:
        total_leakage_energy = accumulate_core_leakage_energy(
            latency_cc=int(latency),
            system_clock_freq_ghz=system_clock_freq,
            dvfs_switching_latency_ms=dvfs_switching_latency,
            dvfs_allocations=None,
            sta_energy_lut=None,
            static_power_per_core_uW=None,
        )
    total_cn_onchip_energy += total_leakage_energy
    return (
        latency,
        total_cn_onchip_energy,
        total_cn_offchip_link_energy,
        total_cn_offchip_memory_energy,
        total_eviction_to_offchip_link_energy,
        total_eviction_to_offchip_memory_energy,
        total_sink_layer_output_offchip_link_energy,
        total_sink_layer_output_offchip_memory_energy,
        total_core_to_core_link_energy,
        total_core_to_core_memory_energy,
    )
