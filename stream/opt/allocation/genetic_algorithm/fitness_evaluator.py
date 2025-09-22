from zigzag.datatypes import LayerOperand
from zigzag.utils import pickle_deepcopy

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.hardware.architecture.accelerator import Accelerator
from stream.utils import CostModelEvaluationLUT, get_required_offchip_bandwidth, get_too_large_operands
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
        layer_groups_flexible,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
    ) -> None:
        super().__init__(workload, accelerator, cost_lut)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]

        self.layer_groups_flexible = layer_groups_flexible
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order

    def get_fitness(self, core_allocations: list[int], return_scme: bool = False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        scme = StreamCostModelEvaluation(
            pickle_deepcopy(self.workload),
            pickle_deepcopy(self.accelerator),
            self.operands_to_prefetch,
            self.scheduling_order,
        )
        scme.run()
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
            (layer_id, group_id) = self.layer_groups_flexible[i]
            # Find all nodes of this coarse id and set their core_allocation, energy and runtime
            nodes = (
                node
                for node in self.workload.node_list
                if isinstance(node, ComputationNode) and node.id == layer_id and node.group == group_id
            )
            for node in nodes:
                equal_unique_node = self.cost_lut.get_equal_node(node)
                assert equal_unique_node is not None, "Node not found in CostModelEvaluationLUT"
                cme = self.cost_lut.get_cme(equal_unique_node, core)
                onchip_energy = cme.energy_total  # Initialize on-chip energy as total energy
                latency = cme.latency_total1
                too_large_operands = get_too_large_operands(cme, self.accelerator, core_id=core_allocation)
                # If there is a too_large_operand, we separate the off-chip energy.
                offchip_energy = 0
                for too_large_operand in too_large_operands:
                    layer_operand = next(
                        (k for (k, v) in cme.layer.memory_operand_links.data.items() if v == too_large_operand)
                    )
                    layer_operand_offchip_energy = cme.mem_energy_breakdown[layer_operand][-1]
                    offchip_energy += layer_operand_offchip_energy
                    onchip_energy -= layer_operand_offchip_energy
                # If there was offchip memory added for too_large_operands, get the offchip bandwidth
                required_offchip_bandwidth = get_required_offchip_bandwidth(cme, too_large_operands)
                node.set_onchip_energy(onchip_energy)
                node.set_offchip_energy(offchip_energy)
                node.set_runtime(int(latency))
                node.set_chosen_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)
                node.set_offchip_bandwidth(required_offchip_bandwidth)

class DvfsFitnessEvaluator(FitnessEvaluator):
    """The DVFS fitness evaluator."""
    # The DVFS fitness evaluator performs DVFS optimization using a genetic algorithm
    # It takes the core-level DVFS granulairity with the real hw constraints such as the dvfs switching latency
    # So the decision variable is the #core * #time_slots * #dvfs_level
    # The #core is from Stream inputs
    # The #dvfs_level is from the dvfs configuration file
    # The #time_slots is determined by the min_dvfs_switch_latency and the scheduling info
    # For example, if min_dvfs_switch_latency = 1ms, and the Stream scheduling output workload lantency = 20ms, and the min dvfs frequency factor is 0.5
    # then the maximum execution time is 20ms/0.5 = 40ms, so the #time_slots = 40ms/1ms = 40 
    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
        #  DVFS related parameters
        dvfs_switching_latency: float = 1.0,   # in ms
        system_clock_freq: float = 1.0,        # in GHz
        Cb_uF_per_core: dict[int, float] = {0: 20.0},  # uF, Core equivalent capacitance
        eta_up: float = 0.9,                   # converter efficiency for up-scaling (0<eta<=1)
        gamma_drop: float = 1.0,               # fraction of down-scaling stored energy treated as loss (CCM≈1, DCM≈0)
        vdd_lut: dict[int, float] = {0: 1.0}   # DVFS level to voltage mapping
    ) -> None:
        super().__init__(workload, accelerator, cost_lut)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order
        self.dvfs_switching_latency = dvfs_switching_latency
        self.system_clock_freq = system_clock_freq
        self.Cb_uF_per_core = Cb_uF_per_core
        self.eta_up = eta_up
        self.gamma_drop = gamma_drop
        self.vdd_lut = vdd_lut  # {dvfs_level: voltage}

    def get_fitness(self, dvfs_allocations: dict[int, dict[int, int]], return_scme: bool = False):
        """Get the fitness of the given core_allocations

        Args:
            dvfs_allocations: dictionary with
            {core_id: {time_window_id: dvfs_level}}
        """
        scme = StreamCostModelEvaluation(
            self.workload,
            self.accelerator,
            self.operands_to_prefetch,
            self.scheduling_order,
            self.system_clock_freq,
            self.dvfs_switching_latency,
            dvfs_allocations,
        )
        scme.run()

        energy_dyn = sum(n.get_onchip_dvfs_energy() for n in scme.workload.node_list)
        latency = scme.latency  # baseline cycles (at nominal freq)

        # Only count switching boundaries that occur before this latency
        dvfs_window_cc = int(self.dvfs_switching_latency * self.system_clock_freq * 1e6)
        if dvfs_window_cc <= 0:
            dvfs_window_cc = 1
        # The boundary between window j and j+1 occurs at time (j+1)*dvfs_window_cc
        # We include boundary j if (j+1)*dvfs_window_cc <= latency
        j_max_inclusive = max(int(latency // dvfs_window_cc) - 1, -1)

        energy_dvfs_switch = self.accumulate_switching_energy_for_allocation(
            dvfs_allocations,
            j_max_inclusive=j_max_inclusive,
        )

        energy = energy_dyn + energy_dvfs_switch
        if not return_scme:
            return energy, latency
        return energy, latency, scme

    def compute_dvfs_switch_energy(
        self,
        Vs: float,
        Ve: float,
        core_id: int = 0
    ) -> float:
        """
        Compute DVFS voltage switching energy from Vs to Ve.

        Model:
        - Upscaling: need to add energy to raise the bulk cap voltage.
            E_cap = 0.5 * Cb * (Ve^2 - Vs^2)
            E_conv = E_cap / eta_up
        - Downscaling: stored energy mostly dissipated (if no recovery).
            E_drop = 0.5 * Cb * (Vs^2 - Ve^2)
            E_conv = gamma_drop * E_drop
        Reference macro-models: Pedram et al., ISLPED/TCAD DVFS overhead modeling.
        """
        Cb = self.Cb_uF_per_core.get(core_id, 20.0) * 1e-6  # convert to Farad
        if Ve == Vs:
            return 0.0

        if Ve > Vs:
            # Up-scaling energy to charge the bulk capacitor, include converter efficiency
            E_cap = 0.5 * Cb * (Ve * Ve - Vs * Vs)
            E_conv = E_cap / max(min(self.eta_up, 1.0), 1e-3)  # clamp eta to (1e-3,1]
        else:
            # Down-scaling: treat stored energy as loss with tunable fraction gamma_drop
            E_drop = 0.5 * Cb * (Vs * Vs - Ve * Ve)
            E_conv = max(self.gamma_drop, 0.0) * E_drop
        E_conv_pJ = E_conv * 1e12  # convert to pico-Joule
        return E_conv_pJ
    
    def accumulate_switching_energy_for_allocation(
        self,
        dvfs_allocations: dict[int, dict[int, int]],
        j_max_inclusive: int,
    ) -> float:
        """
        Sum switching energy for each core across adjacent time-window boundaries (j -> j+1)
        where DVFS level changes, but only for boundaries that occur before the makespan.
        A boundary j is counted only if (j+1) * dvfs_window_cc <= latency, which is
        equivalent to j <= j_max_inclusive.
        """
        total_E_pJ = 0.0
        for core, tw_map in dvfs_allocations.items():
            if not tw_map:
                continue
            # Iterate j in sorted time windows, but stop at j_max_inclusive
            tw_ids = sorted(tw_map.keys())
            for j_idx in range(len(tw_ids) - 1):
                j = tw_ids[j_idx]
                if j > j_max_inclusive:
                    break
                jn = tw_ids[j_idx + 1]
                l1 = tw_map[j]
                l2 = tw_map[jn]
                if l1 == l2:
                    continue
                Vs = self.vdd_lut[l1]
                Ve = self.vdd_lut[l2]
                Es_pJ = self.compute_dvfs_switch_energy(Vs=Vs, Ve=Ve, core_id=core)
                total_E_pJ += Es_pJ
        return total_E_pJ