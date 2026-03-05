from zigzag.datatypes import LayerOperand, MemoryOperand
from zigzag.mapping.data_movement import FourWayDataMoving
from zigzag.utils import pickle_deepcopy

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.hardware.architecture.accelerator import Accelerator
from stream.utils import CostModelEvaluationLUT, get_too_large_operands, get_top_level_inst_bandwidth
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload


class FitnessEvaluator:
    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
    ) -> None:
        self.workload = workload
        self.accelerator = accelerator
        self.cost_lut = cost_lut
        # self.num_cores = len(inputs.accelerator.cores)

    def get_fitness(self):
        raise NotImplementedError


class StandardFitnessEvaluator(FitnessEvaluator):
    """The standard fitness evaluator considers latency, max buffer occupancy and energy equally."""

    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
        flexible_nodes: list[ComputationNode],
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
        latency_attr: str,
        beam_width: int = 1,
    ) -> None:
        super().__init__(workload, accelerator, cost_lut)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]

        self.flexible_nodes = flexible_nodes
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order
        self.latency_attr = latency_attr
        self.beam_width = beam_width

    def get_fitness(self, core_allocations: list[int], return_scme: bool = False, fixed_node_schedule: list | None = None):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        
        # Only deepcopy if we need to return the SCME object, to avoid huge overhead
        if return_scme:
            workload = pickle_deepcopy(self.workload)
            accelerator = pickle_deepcopy(self.accelerator)
        else:
            workload = self.workload
            accelerator = self.accelerator
            
        scme = StreamCostModelEvaluation(
            workload,
            accelerator,
            self.operands_to_prefetch,
            self.scheduling_order,
            self.beam_width,
            fixed_node_schedule=fixed_node_schedule,
        )
        scme.evaluate()
        energy = scme.energy
        latency = scme.latency
        if not return_scme:
            return energy, latency
        return energy, latency, scme

    def set_node_core_allocations(self, core_allocations: list[int]):
        """Sets the core allocation of all nodes in self.workload according to core_allocations.
        This will only set the energy, runtime and core_allocation of the nodes which are flexible in their core
        allocation.
        We assume the energy, runtime and core_allocation of the other nodes are already set.

        Args:
            core_allocations (list): list of the node-core allocations
        """
        for i, core_allocation in enumerate(core_allocations):
            core = self.accelerator.get_core(core_allocation)
            node = self.flexible_nodes[i]
            equal_unique_node = self.cost_lut.get_equal_node(node) or node
            cme = self.cost_lut.get_cme(equal_unique_node, core)
            onchip_energy = cme.energy_total  # Initialize on-chip energy as total energy
            latency = getattr(cme, self.latency_attr)
            too_large_operands = get_too_large_operands(cme, self.accelerator, core_id=core_allocation)
            # If there is a too_large_operand, we separate the off-chip energy.
            offchip_energy = 0
            for too_large_operand in too_large_operands:
                layer_operand = next(
                    k for (k, v) in cme.layer.memory_operand_links.data.items() if v == too_large_operand
                )
                layer_operand_offchip_energy = cme.mem_energy_breakdown[layer_operand][-1]
                offchip_energy += layer_operand_offchip_energy
                onchip_energy -= layer_operand_offchip_energy
            # Get the required offchip bandwidth during the execution of the node for all directions
            bandwidth_scaling = cme.ideal_temporal_cycle / latency
            offchip_bandwidth_per_op: dict[MemoryOperand, FourWayDataMoving] = {
                mem_op: get_top_level_inst_bandwidth(cme, mem_op, bandwidth_scaling)
                for mem_op in too_large_operands
            }
            node.set_onchip_energy(onchip_energy)
            node.set_offchip_energy(offchip_energy)
            node.set_runtime(int(latency))
            node.set_chosen_core_allocation(core_allocation)
            node.set_too_large_operands(too_large_operands)
            node.set_offchip_bandwidth(offchip_bandwidth_per_op)


class CoOptimizationFitnessEvaluator(StandardFitnessEvaluator):
    """Fitness evaluator for co-optimization of node-core allocation and DVFS level."""

    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
        flexible_nodes: list[ComputationNode],
        flexible_nodes_dvfs: list[ComputationNode],
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
        latency_attr: str,
        beam_width: int = 1,
        dvfs_switching_speed: int = 0,
    ) -> None:
        super().__init__(
            workload,
            accelerator,
            cost_lut,
            flexible_nodes,
            operands_to_prefetch,
            scheduling_order,
            latency_attr,
            beam_width,
        )
        self.flexible_nodes_dvfs = flexible_nodes_dvfs
        self.dvfs_switching_speed = max(0, int(dvfs_switching_speed))

    def get_fitness(self, chromosome: list[int], return_scme: bool = False, fixed_node_schedule: list | None = None):
        """Get the fitness of the given individual (chromosome).
        The chromosome contains core allocations followed by DVFS levels.
        """
        self.set_node_attributes(chromosome)
        
        # Only deepcopy if we need to return the SCME object, to avoid huge overhead
        if return_scme:
            workload = pickle_deepcopy(self.workload)
            accelerator = pickle_deepcopy(self.accelerator)
        else:
            workload = self.workload
            accelerator = self.accelerator
            
        scme = StreamCostModelEvaluation(
            workload,
            accelerator,
            self.operands_to_prefetch,
            self.scheduling_order,
            self.beam_width,
            fixed_node_schedule=fixed_node_schedule,
        )
        scme.evaluate()
        energy = scme.energy
        latency = scme.latency
        if not return_scme:
            return energy, latency
        return energy, latency, scme

    def set_node_attributes(self, chromosome: list[int]):
        """Sets both core allocation and DVFS levels for nodes."""
        # 1. Set core allocations (first part of chromosome)
        num_alloc = len(self.flexible_nodes)
        core_allocations = chromosome[:num_alloc]
        self.set_node_core_allocations(core_allocations)

        # 2. Set DVFS levels (second part of chromosome)
        dvfs_levels = chromosome[num_alloc:]
        candidate_level_per_node = {
            (node.id, node.sub_id): level
            for node, level in zip(self.flexible_nodes_dvfs, dvfs_levels)
        }

        node_lookup = {(node.id, node.sub_id): node for node in self.workload.node_list}
        ordered_nodes = []
        seen = set()
        for key in self.scheduling_order:
            node = node_lookup.get(key)
            if node is not None and key not in seen:
                ordered_nodes.append(node)
                seen.add(key)
        for node in self.workload.node_list:
            key = (node.id, node.sub_id)
            if key not in seen:
                ordered_nodes.append(node)

        prev_level_per_core: dict[int, int] = {}
        for node in ordered_nodes:
            key = (node.id, node.sub_id)
            core = node.chosen_core_allocation
            if core is None:
                continue

            candidate_level = candidate_level_per_node.get(key)
            runtime = node.runtime if node.runtime is not None else 0
            if self.dvfs_switching_speed > 0 and runtime < self.dvfs_switching_speed:
                fallback_level = prev_level_per_core.get(core, node.get_dvfs_level())
                level = int(fallback_level) if fallback_level is not None else 0
            elif candidate_level is not None:
                level = int(candidate_level)
            else:
                fallback_level = node.get_dvfs_level()
                level = int(fallback_level) if fallback_level is not None else 0

            node.set_dvfs_level(level)
            prev_level_per_core[core] = level
