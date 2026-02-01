import logging
import os
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
from zigzag.datatypes import LayerOperand
from zigzag.utils import pickle_deepcopy

from stream.hardware.architecture.accelerator import Accelerator
from stream.opt.allocation.genetic_algorithm.fitness_evaluator import (
    CoOptimizationFitnessEvaluator,
    StandardFitnessEvaluator,
)
from stream.opt.allocation.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from stream.stages.stage import Stage, StageCallable
from stream.utils import CostModelEvaluationLUT, get_unique_nodes
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)


class GeneticAlgorithmAllocationStage(Stage):
    """
    Class that finds the best inter-core mapping using a genetic algorithm.
    From the IntraCoreMappingStage we receive the `CostModelEvaluationLUT`, containing for each node and its valid core
      allocations the best CME.
    We then initialize the genetic algorithm.
    TODO A separate "GeneticAlgorithmStage" should be added where we parse all GA-related info and this stage then calls
    TODO that stage.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
        nb_ga_generations: int,
        nb_ga_individuals: int,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
        **kwargs: Any,
    ):
        """Initialize the InterCoreMappingStage.

        Args:
            list_of_callables (list): List of the substages to be called. This should be empty as this is a leaf stage.
            workload (DiGraph): The NetworkX DiGraph representing the workload to be scheduled
            accelerator (Accelerator): The hardware accelerator onto which we schedule the workload
            cost_lut (CostModelEvaluationLUT): A LUT of CMEs for each unique node and their valid cores
            nb_ga_generations: The number of generations considered by the genetic algorithm
            nb_ga_individuals: The number of individuals in each genetic algorithm generation
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.cost_lut = cost_lut
        self.nb_generations = nb_ga_generations
        self.nb_individuals = nb_ga_individuals
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order
        self.latency_attr = kwargs.get("latency_attr", "latency_total2")
        self.coala_beam_width = kwargs.get("coala_beam_width", 1)
        self.do_dvfs_cooptimization = kwargs.get("do_dvfs_cooptimization", False)
        self.dvfs_config_path = kwargs.get("dvfs_config_path", None)
        self.output_path = kwargs.get("output_path", "outputs")
        self.num_procs = kwargs.get("num_procs", 1)

        self.flexible_nodes: list[ComputationNode] = []
        for n in self.workload.node_list:
            if not isinstance(n.chosen_core_allocation, int):
                self.flexible_nodes.append(n)

        # For each unique node get the possible core allocations by getting the ids of the cores in cost_lut
        self.valid_allocations: list[list[int]] = []
        for flexible_node in self.flexible_nodes:
            self.valid_allocations.append(flexible_node.core_allocation)

        if self.do_dvfs_cooptimization and self.dvfs_config_path:
            from stream_dvfs.src.dvfs_parser import DvfsParser

            dvfs_parser = DvfsParser(self.dvfs_config_path)
            self.dvfs_luts = dvfs_parser.parse_dvfs_data()
            for node in self.workload.node_list:
                node.set_dvfs_level(0)  # default DVFS level
                node.set_vdd_lut(self.dvfs_luts["vdd_lut"])
                node.set_freq_lut(self.dvfs_luts["freq_lut"])
                node.set_dyn_energy_lut(self.dvfs_luts["dyn_energy_lut"])
                node.set_sta_energy_lut(self.dvfs_luts["sta_energy_lut"])
            # Apply DVFS level choices to all computation nodes
            self.flexible_nodes_dvfs = self.workload.node_list
            self.dvfs_level_choices = sorted(self.dvfs_luts["freq_lut"].keys())
            for _ in self.flexible_nodes_dvfs:
                self.valid_allocations.append(self.dvfs_level_choices)

            # Initialize the combined fitness evaluator
            self.fitness_evaluator = CoOptimizationFitnessEvaluator(
                self.workload,
                self.accelerator,
                self.cost_lut,
                self.flexible_nodes,
                self.flexible_nodes_dvfs,
                self.operands_to_prefetch,
                self.scheduling_order,
                self.latency_attr,
                self.coala_beam_width,
            )
        else:
            # Initialize the standard fitness evaluator
            self.fitness_evaluator = StandardFitnessEvaluator(
                self.workload,
                self.accelerator,
                self.cost_lut,
                self.flexible_nodes,
                self.operands_to_prefetch,
                self.scheduling_order,
                self.latency_attr,
                self.coala_beam_width,
            )

        # Extract the length of an individual.
        self.individual_length = len(self.valid_allocations)

    def run(self):
        """Run the InterCoreMappingStage by checking if we have a fixed core_allocation.
        - if yes: evaluate fixed core allocation
        - if no: initialize and run the genetic algorithm
        """

        logger.info("Start GeneticAlgorithmAllocationStage.")
        if self.individual_length == 0:
            logger.info("Evaluating fixed layer-core allocation.")
            core_allocations = []
            res = self.fitness_evaluator.get_fitness(core_allocations, return_scme=True)
            scme = res[2] if len(res) == 3 else None
            if scme:
                logger.info(f"Fixed allocation energy: {scme.energy:.2e}, latency: {scme.latency:.2e}")
                yield scme, None
        else:
            logger.info(
                f"Running Genetic Algorithm with {self.nb_generations} "
                f"generations and {self.nb_individuals} individuals."
            )
            flexible_layer_names = [f"{n.name}" for n in self.flexible_nodes]
            logger.info(
                f"Exploring allocation for {len(self.flexible_nodes)} flexible layers: {flexible_layer_names}"
            )

            # Create population seeds
            pop_seeds = []
            if self.do_dvfs_cooptimization:
                import random
                num_flex = len(self.flexible_nodes)
                num_dvfs = len(self.flexible_nodes_dvfs)
                # Seed with global DVFS levels (all nodes at same level)
                for level in self.dvfs_level_choices:
                    core_allocs = [random.choice(self.valid_allocations[i]) for i in range(num_flex)]
                    dvfs_levels = [level] * num_dvfs
                    pop_seeds.append(core_allocs + dvfs_levels)

            # Initialize the genetic algorithm
            self.genetic_algorithm = GeneticAlgorithm(
                self.fitness_evaluator,
                self.individual_length,
                self.valid_allocations,
                self.nb_generations,
                self.nb_individuals,
                pop=pop_seeds,
                num_processes=self.num_procs,
            )
            # Run the genetic algorithm and get the results
            pop, hof = self.genetic_algorithm.run()
            logger.info("Finished Genetic Algorithm.")

            if self.do_dvfs_cooptimization:
                self.plot_comparison(hof)

            # Return the SCME of the last individual in the hall of fame
            best_core_allocations = hof[-1]
            res = self.fitness_evaluator.get_fitness(best_core_allocations, return_scme=True)
            scme = res[2] if len(res) == 3 else None
            if scme:
                logger.info(f"Best allocation energy: {scme.energy:.2e}, latency: {scme.latency:.2e}")
                yield scme, None
        logger.info("Finished GeneticAlgorithmAllocationStage.")

    def is_leaf(self) -> bool:
        return True

    def plot_comparison(self, hall_of_fame):
        """Plot the comparison between the GA Pareto front and the naive DVFS results."""
        import matplotlib.pyplot as plt
        import os
        os.makedirs(self.output_path, exist_ok=True)
        fig_path = os.path.join(self.output_path, "dvfs_comparison.png")

        # 1. Get GA Pareto points
        pf_energies = [ind.fitness.values[0] for ind in hall_of_fame]
        pf_latencies = [ind.fitness.values[1] for ind in hall_of_fame]

        # 2. Get Naive DVFS point
        # Take the best latency mapping and set DVFS to 0 as baseline
        best_latency_ind = sorted(hall_of_fame, key=lambda x: x.fitness.values[1])[0]
        num_flex = len(self.flexible_nodes)
        core_allocs = best_latency_ind[:num_flex]

        # Evaluate base mapping (No DVFS)
        self.fitness_evaluator.set_node_core_allocations(core_allocations=core_allocs)
        for node in self.workload.node_list:
            node.set_dvfs_level(0)
        
        # Explicit type checking for base_result
        base_result = self.fitness_evaluator.get_fitness(core_allocs, return_scme=True)
        if len(base_result) == 3:
            base_energy = base_result[0]
            base_latency = base_result[1]
            base_scme = base_result[2]
        else:
            raise ValueError("Fitness evaluator did not return SCME even though return_scme=True was set.")

        # Apply Naive DVFS heuristic
        node_events = self.get_communication_dic(base_scme.accelerator)
        start_time_per_core = self.get_start_time_per_core()
        runtime_per_node = self.get_runtime_per_node()

        for node_id, runtime in runtime_per_node.items():
            sub_nodes = [n for n in self.workload.node_list if n.id == node_id]
            last_sub_node = max(sub_nodes, key=lambda n: n.get_end())
            # Simple slack-based DVFS
            slack = self.get_slack(last_sub_node, node_events, start_time_per_core)
            dvfs_level = self.compute_dvfs_level(runtime, slack)
            for sn in sub_nodes:
                sn.set_dvfs_level(dvfs_level)

        # Evaluate Naive Result
        naive_result = self.fitness_evaluator.get_fitness(core_allocs, return_scme=False)
        naive_energy, naive_latency = naive_result[0], naive_result[1]

        # 3. Global DVFS Scaling (All nodes at same level)
        global_energies = []
        global_latencies = []
        for level in self.dvfs_level_choices:
            for node in self.workload.node_list:
                node.set_dvfs_level(level)
            g_res = self.fitness_evaluator.get_fitness(core_allocs, return_scme=False)
            global_energies.append(g_res[0])
            global_latencies.append(g_res[1])
            if level == 0:
                base_energy, base_latency = g_res[0], g_res[1]

        # 4. Normalize
        if base_energy and base_latency:
            pf_energies = [e / base_energy for e in pf_energies]
            pf_latencies = [l / base_latency for l in pf_latencies]
            if naive_energy is not None and naive_latency is not None:
                naive_energy /= base_energy
                naive_latency /= base_latency
            global_energies = [e / base_energy for e in global_energies]
            global_latencies = [l / base_latency for l in global_latencies]
            base_energy, base_latency = 1.0, 1.0

        # 5. Plot
        plt.figure(figsize=(10, 6))
        # Plot Pareto Front from individuals
        plt.scatter(pf_energies, pf_latencies, c="red", label="Co-optimized GA (All Individuals)", alpha=0.4, s=20, zorder=3)
        
        # Plot Global Scaling Curve
        plt.plot(global_energies, global_latencies, 'g-', alpha=0.6, label="Global Scaling Curve", zorder=1)
        plt.scatter(global_energies, global_latencies, c='green', marker='o', s=50, label="Global Levels (0-10)", zorder=2)

        if naive_energy is not None and naive_latency is not None:
            plt.scatter([naive_energy], [naive_latency], c="blue", marker="D", s=120, label="Naive DVFS (Mapping-based)", zorder=4)
        if base_energy is not None and base_latency is not None:
            plt.scatter([base_energy], [base_latency], c="black", marker="x", s=120, label="Baseline (Level 0)", zorder=5)

        plt.xlabel("Normalized Energy")
        plt.ylabel("Normalized Latency")
        plt.title("DVFS Optimization Comparison (Normalized)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Comparison figure saved to {fig_path}")

    def get_communication_dic(self, accelerator):
        active_links = accelerator.communication_manager.get_all_links()
        node_events = {}
        for cl in active_links:
            for event in cl.events:
                if (event.end - event.start) == 0:
                    continue
                if not event.tensor.memory_operand.is_output():
                    continue
                node_events[(event.tensor.origin.id, event.tensor.origin.sub_id)] = {
                    "Runtime": event.end - event.start,
                }
        return node_events

    def get_start_time_per_core(self):
        start_time_per_core = defaultdict(list)
        for node in self.workload.node_list:
            start_time_per_core[node.chosen_core_allocation].append((node, node.start))
        for core in start_time_per_core:
            start_time_per_core[core].sort(key=lambda x: x[1])
        return start_time_per_core

    def get_runtime_per_node(self) -> dict[int, int]:
        nodes_by_id = defaultdict(list)
        for node in self.workload.node_list:
            nodes_by_id[node.id].append(node)
        runtime_per_node = {}
        for node_id, nodes in nodes_by_id.items():
            start = min(n.start for n in nodes)
            end = max(n.get_end() for n in nodes)
            runtime_per_node[node_id] = end - start
        return runtime_per_node

    def get_slack(self, node, node_event_dic, start_time_per_core):
        cur_end = node.get_end()
        cur_core = node.chosen_core_allocation
        successor_nodes = list(self.workload.successors(node))
        successor_starts = [n.start for n in successor_nodes]

        output_transfer_time = node_event_dic.get((node.id, node.sub_id), {}).get("Runtime", 0)
        est_successors = min(successor_starts) if successor_starts else cur_end + output_transfer_time

        # find the next start time on the same core
        est_core = float("inf")
        for next_node, next_start in start_time_per_core.get(cur_core, []):
            if next_start >= cur_end:
                est_core = next_start
                break

        deadline = min(est_successors - output_transfer_time, est_core)
        return deadline - cur_end

    def compute_dvfs_level(self, runtime, slack):
        freq_lut = self.dvfs_luts["freq_lut"]
        sorted_levels = sorted(freq_lut.keys(), reverse=True)
        for level in sorted_levels:
            if level == 0:
                continue
            required_runtime = runtime / freq_lut[level]
            if required_runtime <= (runtime + slack):
                return level
        return 0
