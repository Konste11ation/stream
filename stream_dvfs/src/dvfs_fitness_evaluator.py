import sys
import os
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_DIR = STREAM_DVFS_DIR.parent
from zigzag.utils import pickle_deepcopy
from stream.opt.allocation.genetic_algorithm.fitness_evaluator import FitnessEvaluator
from stream.hardware.architecture.accelerator import Accelerator
from stream.utils import CostModelEvaluationLUT
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.cost_model.cost_model import StreamCostModelEvaluation
class DvfsFitnessEvaluator(FitnessEvaluator):
    """The DVFS fitness evaluator."""

    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
        operands_to_prefetch: list,
        scheduling_order: list[tuple[int, int]],
        dvfs_node_id_list: list[int],
    ) -> None:
        super().__init__(workload, accelerator, cost_lut)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order
        self.dvfs_node_id_list = dvfs_node_id_list

    def get_fitness(self, dvfs_level_allocation: list[int], return_scme: bool = False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_dvfs_level(dvfs_level_allocation)
        scme = StreamCostModelEvaluation(
            pickle_deepcopy(self.workload),
            pickle_deepcopy(self.accelerator),
            self.operands_to_prefetch,
            self.scheduling_order,
        )
        scme.evaluate()
        energy = sum(n.get_onchip_energy() for n in scme.workload.node_list)
        latency = max(n.get_end() for n in scme.workload.node_list)
        if not return_scme:
            return energy, latency
        return energy, latency, scme
    def get_sub_nodes(self, node_id):
        sub_nodes = [n for n in self.workload.node_list if n.id == node_id]
        return sub_nodes
    def set_node_dvfs_level(self,dvfs_level_allocation: list[int]):
        """Set the DVFS level for each node in the workload based on the given allocation.
        Args:
            dvfs_level_allocation (list): A list of DVFS levels corresponding to each node in the workload.
        """
        if len(dvfs_level_allocation) != len(self.dvfs_node_id_list):
            raise ValueError("The length of dvfs_level_allocation must match the number of nodes in the workload.")
        node_id_to_dvfs_level = {node_id: dvfs_level for node_id, dvfs_level in zip(self.dvfs_node_id_list, dvfs_level_allocation)}
        for node_id, dvfs_level in node_id_to_dvfs_level.items():
            sub_nodes = self.get_sub_nodes(node_id)
            for sub_node in sub_nodes:
                sub_node.set_dvfs_level(dvfs_level)