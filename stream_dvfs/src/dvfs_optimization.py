import sys
import os
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_DIR = STREAM_DVFS_DIR.parent
sys.path.append(str(STREAM_DIR))
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from zigzag.utils import pickle_deepcopy, pickle_save
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.hardware.architecture.accelerator import Accelerator
from stream_dvfs.src.dvfs_parser import DvfsParser
from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream_dvfs.src.dvfs_fitness_evaluator import DvfsFitnessEvaluator
from stream_dvfs.src.dvfs_ga import DvfsGeneticAlgorithm
import logging
logger = logging.getLogger(__name__)
from collections import defaultdict
class DvfsOptimizationStage(Stage):

    def __init__(self, 
                 list_of_callables: list[StageCallable],
                *,
                workload: ComputationNodeWorkload,
                accelerator: Accelerator,
                scheduling_order: list[tuple[int, int]],
                operands_to_prefetch: list,
                **kwargs
                ):
        super().__init__(list_of_callables, **kwargs)
        self.dvfs_output_path = kwargs["dvfs_output_path"]
        self.workload = workload
        self.accelerator = accelerator
        self.scheduling_order = scheduling_order
        self.operands_to_prefetch = operands_to_prefetch
        self.ga_nb_generations = kwargs["ga_nb_generations"]
        self.ga_nb_individuals = kwargs["ga_nb_individuals"]
        self.dvfs_parser = DvfsParser(kwargs["dvfs_cfg_path"])
        self.dvfs_luts = {}
        self.base_energy = sum(n.get_onchip_energy() for n in workload.node_list)
        self.base_latency = max(n.get_end() for n in workload.node_list)
        self.brute_force_energy = None
        self.brute_force_latency = None
        # Assume the system clock is 1GHz
        # Can be modified later
        self.dvfs_switching_speed = 100000 # in cycles, assume the system clock is 1GHz, so 100000 cycles = 0.1ms
    
    def run(self):
        logger.info(f"Start DVFS optimization stage")     
        self.parse_and_set_dvfs_data()
        result, hof = self.run_dvfs_ga()
        opt_energy = result[0]
        opt_latency = result[1]
        opt_scme = result[2]
        energy_reduction, latency_overhead = self.compute_metrics(opt_energy, opt_latency)
        print(f"Optimized DVFS Energy: {opt_energy}, Latency: {opt_latency}")
        print(f"Optimized DVFS Energy Reduction: {energy_reduction*100:.2f}%, Latency Overhead: {latency_overhead*100:.2f}%")
        self.plot_pareto(hof)
        return opt_scme

    def parse_and_set_dvfs_data(self):
        self.dvfs_luts = self.dvfs_parser.parse_dvfs_data()
        for node in self.workload.node_list:
            node.set_dvfs_level(0)  # default DVFS level
            node.set_vdd_lut(self.dvfs_luts['vdd_lut'])
            node.set_freq_lut(self.dvfs_luts['freq_lut'])
            node.set_dyn_energy_lut(self.dvfs_luts['dyn_energy_lut'])
            node.set_sta_energy_lut(self.dvfs_luts['sta_energy_lut'])

    def get_communication_dic(self):
        """
        Return all the output transfer event as a dic
        key:(id,sub_id)
        value:{start, end, runtime, tensors}
        """
        active_links = self.accelerator.communication_manager.get_all_links()
        node_events = {}
        for pair_link_id, cl in enumerate(active_links):
            for event in cl.events:
                start = event.start
                end = event.end
                runtime = end - start
                tensor = event.tensor
                node = event.tensor.origin
                tensor_type =  event.tensor.memory_operand
                node_id = node.id
                node_sub_id = node.sub_id

                if runtime == 0:
                    continue
                if not tensor_type.is_output():
                    continue
                key = (node_id,node_sub_id)
                event_record = {
                    "Start": start,
                    "End": end,
                    "Runtime": runtime,
                    "Tensors": tensor
                }
                node_events.setdefault(key, event_record)
        return node_events
    
    def get_runtime_per_node(self) -> dict[int, int]:
        """Aggregate total runtime for each logical node id.

        For nodes that have been split into multiple sub-nodes (same ``node.id`` but different
        ``sub_id``), the runtime is computed from the earliest start time and latest end time across
        all sub-nodes.
        """

        nodes_by_id: dict[int, list[ComputationNode]] = defaultdict(list)
        for node in self.workload.node_list:
            nodes_by_id[node.id].append(node)

        runtime_per_node: dict[int, int] = {}
        for node_id, nodes in nodes_by_id.items():
            start_times = [n.get_start() for n in nodes if n.get_start() is not None and n.get_start() >= 0]
            end_times = [n.get_end() for n in nodes if n.get_end() is not None and n.get_end() >= 0]

            if not start_times or not end_times:
                logger.warning(
                    "Missing start/end times for node %s; defaulting runtime to 0.",
                    node_id,
                )
                runtime_per_node[node_id] = 0
                continue

            first_start = min(start_times)
            last_end = max(end_times)
            total_runtime = max(0, last_end - first_start)
            runtime_per_node[node_id] = total_runtime

        return runtime_per_node
    def get_start_time_per_core(self):
        """
        Retuen a dict to store the start_time per core per node
        key: core
        value: [(node, start_time), ...]
        """
        start_time_per_core = defaultdict(list)
        
        for node in self.workload.node_list:
            core = node.chosen_core_allocation
            start_time = node.get_start()
            start_time_per_core[core].append((node, start_time))
        # sorted by the start time
        # smallest first
        for core in start_time_per_core:
            start_time_per_core[core].sort(key=lambda x: x[1])
        return start_time_per_core
    def find_next_start_time_per_core(self,start_time_per_core, core, end_time):
        # Check if the core exist
        if core not in start_time_per_core:
            return float("nan")
        
        # Get all the nodes running on the current core
        core_nodes = start_time_per_core[core]

        # find the nxt start time after the current end time
        for _,start_time in core_nodes:
            if(start_time>=end_time):
                return start_time
        return float("nan")
    def compute_dvfs_level(self, runtime, slack):
        freq_lut = self.dvfs_luts["freq_lut"]
        sorted_levels = sorted(freq_lut.keys(), reverse=True)
        for level in sorted_levels:
            freq_scaling = freq_lut[level]
            runtime_dvfs = int(runtime / freq_scaling)
            if runtime_dvfs <= runtime + slack:
                return level
        return min(freq_lut.keys())
    def get_sub_nodes(self, node_id):
        sub_nodes = [n for n in self.workload.node_list if n.id == node_id]
        return sub_nodes
    def get_slack(self, node, node_event_dic, start_time_per_core):
        cur_id = node.id
        cur_sub_id = node.sub_id
        cur_end = node.get_end()
        cur_core = node.chosen_core_allocation
        successor_nodes = set(self.workload.successors(node))
        successor_nodes_start_times = [n.start for n in successor_nodes]
        # the current node is the exit node
        if successor_nodes_start_times == []:
            return 0
        # get the output transfer time
        output_transfer_event = node_event_dic.get((cur_id, cur_sub_id), [])
        output_transfer_time = output_transfer_event["Runtime"] if output_transfer_event else 0 
        # the earlist start time of the successor
        est_successors = min(successor_nodes_start_times)
        # the earlist start time of the current core node
        est_core =  self.find_next_start_time_per_core(start_time_per_core,
                                                        cur_core,
                                                        cur_end)
        deadline = min(est_successors-output_transfer_time,est_core)
        slack = deadline - cur_end
        return slack
    def brute_dvfs_opt(self):
        node_event_dic = self.get_communication_dic()
        start_time_per_core = self.get_start_time_per_core()
        runtime_per_node = self.get_runtime_per_node()
        node_id_dvfs_dict: dict[int, int] = defaultdict(list)
        for node_id, runtime in runtime_per_node.items():
            if runtime < self.dvfs_switching_speed:
                # skip the node if the runtime is less than the dvfs switching speed
                sub_nodes = self.get_sub_nodes(node_id)
                for sub_node in sub_nodes:
                    sub_node.set_dvfs_level(0)  # set to the default level
            else:
                sub_nodes = self.get_sub_nodes(node_id)
                last_sub_node = max(sub_nodes, key=lambda n: n.get_end())
                slack = self.get_slack(last_sub_node, node_event_dic, start_time_per_core)
                dvfs_level = self.compute_dvfs_level(runtime, slack)
                for sub_node in sub_nodes:
                    sub_node.set_dvfs_level(dvfs_level)
                node_id_dvfs_dict[node_id] = dvfs_level
        return sorted(node_id_dvfs_dict.items(), key=lambda x: x[1])
    def run_coala(self):
        """
        Run the cost model evaluation with current DVFS settings.
        
        Args:
            return_scme: If True, return (scme, energy, latency), otherwise return (energy, latency)
        """
        scme_dvfs = StreamCostModelEvaluation(
            pickle_deepcopy(self.workload),
            pickle_deepcopy(self.accelerator),
            self.operands_to_prefetch,
            self.scheduling_order,
        )
        scme_dvfs.evaluate()
        dvfs_workload = scme_dvfs.workload
        dvfs_energy = sum(n.get_onchip_energy() for n in dvfs_workload.node_list)
        dvfs_latency = max(n.get_end() for n in dvfs_workload.node_list)
        
        return scme_dvfs, dvfs_energy, dvfs_latency
    def compute_metrics(self, energy, latency):
        energy_reduction = (self.base_energy - energy) / self.base_energy if self.base_energy > 0 else 0
        latency_overhead = (latency - self.base_latency) / self.base_latency if self.base_latency > 0 else 0
        return energy_reduction, latency_overhead
    def run_dvfs_ga(self):
        # run the brute force first to get a baseline
        brute_force_dvfs = self.brute_dvfs_opt()
        brute_scme, brute_energy, brute_latency = self.run_coala()
        self.brute_force_energy = brute_energy
        self.brute_force_latency = brute_latency
        print("Base Energy:", self.base_energy, "Base Latency:", self.base_latency)
        print(f"Brute Force DVFS Energy: {brute_energy}, Latency: {brute_latency}")
        brute_energy_reduction, brute_latency_overhead = self.compute_metrics(brute_energy, brute_latency)
        print(f"Brute Force DVFS Energy Reduction: {brute_energy_reduction*100:.2f}%, Latency Overhead: {brute_latency_overhead*100:.2f}%")
        self.plot_brute_force_dvfs(brute_energy, brute_latency)
        runtime_per_node = self.get_runtime_per_node()
        dvfs_node_id_list = [node_id for node_id, runtime in runtime_per_node.items() if runtime >= self.dvfs_switching_speed]
        print(f"Nodes considered for DVFS optimization: {dvfs_node_id_list}")
        fitness_evaluator = DvfsFitnessEvaluator(self.workload,
                                                 self.accelerator,
                                                 [],
                                                 self.operands_to_prefetch,
                                                 self.scheduling_order,
                                                 dvfs_node_id_list)
        individual_length = len(dvfs_node_id_list)
        valid_allocations = [min(self.dvfs_luts["vdd_lut"].keys()), max(self.dvfs_luts["vdd_lut"].keys())]
        pop_init = []
        for node_id in dvfs_node_id_list:
            # Find the DVFS level for this node_id from brute_force_dvfs
            dvfs_level = next((level for id_, level in brute_force_dvfs if id_ == node_id), 0)
            pop_init.append(dvfs_level)
        genetic_alg = DvfsGeneticAlgorithm(
            fitness_evaluator,
            individual_length,
            valid_allocations,
            self.ga_nb_generations,
            self.ga_nb_individuals,
            pop_init
        )
        pop, hof = genetic_alg.run()
        best_dvfs_level_allocation = hof[-1]
        results = fitness_evaluator.get_fitness(best_dvfs_level_allocation, return_scme = True)
        return results, hof
    def plot_brute_force_dvfs(self, brute_force_dvfs_energy, brute_force_dvfs_latency):
        os.makedirs(self.dvfs_output_path, exist_ok=True)
        fig_filename = os.path.join(self.dvfs_output_path, "brute_force_dvfs.png")
        plt.figure(figsize=(6, 4))
        plt.scatter(brute_force_dvfs_energy/self.base_energy, brute_force_dvfs_latency/self.base_latency, 
                    c='orange', s=50, edgecolors='black', linewidths=2,
                    marker='D', label='Brute Force DVFS', zorder=4)
        # Plot ideal DVFS curve
        dyn_energy_lut = self.dvfs_luts['dyn_energy_lut']
        freq_lut = self.dvfs_luts['freq_lut']
        # Get sorted DVFS levels for consistent curve
        dvfs_levels = sorted(dyn_energy_lut.keys())
        ideal_energy = []
        ideal_latency = []
        
        for level in dvfs_levels:
            # Normalized energy from dynamic energy scaling
            norm_energy = dyn_energy_lut[level]
            # Normalized latency from frequency scaling (inverse relationship)
            norm_latency = 1.0 / freq_lut[level]
            ideal_energy.append(norm_energy)
            ideal_latency.append(norm_latency)
        plt.plot(ideal_energy, ideal_latency, 
            'b-o', linewidth=2, markersize=6,
            label='Naive DVFS Curve', zorder=2)
        plt.xlim(0, 1.05)
        plt.ylim(0, 6)
        plt.xlabel('Normalized Energy Consumption', fontsize=12)
        plt.ylabel('Normalized Latency', fontsize=12)
        plt.title('DVFS Optimization: Pareto Front vs Naive Curve', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight', transparent=False)
    def plot_pareto(self, hof):
        os.makedirs(self.dvfs_output_path, exist_ok=True)
        fig_filename = os.path.join(self.dvfs_output_path, "pareto.png")
        meta_filename = os.path.join(self.dvfs_output_path, "dvfs_meta.pickle")
        plt.figure(figsize=(6, 4))
        
        # Plot Pareto front with normalized values
        pareto_front = hof
        if len(pareto_front) > 0:
            pf_energy = [ind.fitness.values[0] / self.base_energy for ind in pareto_front]
            pf_latency = [ind.fitness.values[1] / self.base_latency for ind in pareto_front]
            plt.scatter(pf_energy, pf_latency, 
                        c='red', s=50, edgecolors='black',
                        label='Pareto Front', zorder=3)
        
        # Plot the brute force point
        if self.brute_force_energy and self.brute_force_latency:
            plt.scatter(self.brute_force_energy/self.base_energy, self.brute_force_latency/self.base_latency, 
                        c='orange', s=50, edgecolors='black', linewidths=2,
                        marker='D', label='Brute Force DVFS', zorder=4)
        
        # Plot ideal DVFS curve
        dyn_energy_lut = self.dvfs_luts['dyn_energy_lut']
        freq_lut = self.dvfs_luts['freq_lut']
        
        # Get sorted DVFS levels for consistent curve
        dvfs_levels = sorted(dyn_energy_lut.keys())
        ideal_energy = []
        ideal_latency = []
        
        for level in dvfs_levels:
            # Normalized energy from dynamic energy scaling
            norm_energy = dyn_energy_lut[level]
            # Normalized latency from frequency scaling (inverse relationship)
            norm_latency = 1.0 / freq_lut[level]
            ideal_energy.append(norm_energy)
            ideal_latency.append(norm_latency)
        
        plt.plot(ideal_energy, ideal_latency, 
                 'b-o', linewidth=2, markersize=6,
                 label='Naive DVFS Curve', zorder=2)
        
        # Add base point at (1, 1) for reference
        plt.scatter(1.0, 1.0,
                    c='green', s=100, edgecolors='black', linewidths=2,
                    marker='*', label='Base (No DVFS)', zorder=4)
        
        plt.xlabel('Normalized Energy Consumption', fontsize=12)
        plt.ylabel('Normalized Latency', fontsize=12)
        plt.title('DVFS Optimization: Pareto Front vs Naive Curve', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits to show the full range
        if pf_energy and pf_latency:
            x_min = min(min(pf_energy), min(ideal_energy), 0.8)
            x_max = max(max(pf_energy), max(ideal_energy), 1.2)
            y_min = min(min(pf_latency), min(ideal_latency), 0.8)
            y_max = max(max(pf_latency), max(ideal_latency), 1.2)
            plt.xlim(x_min - 0.1, 1.05)
            plt.ylim(y_min - 0.1, 6)
        
        if fig_filename:
            plt.savefig(fig_filename, dpi=300, bbox_inches='tight', transparent=False)
        else:
            plt.show()
        
        # Save metadata with normalized values
        dvfs_meta = {
            "pf_energy_normalized": pf_energy if 'pf_energy' in locals() else [],
            "pf_latency_normalized": pf_latency if 'pf_latency' in locals() else [],
            "ideal_energy_normalized": ideal_energy,
            "ideal_latency_normalized": ideal_latency,
            "base_energy": self.base_energy,
            "base_latency": self.base_latency,
            "brute_force_energy": self.brute_force_energy,
            "brute_force_latency": self.brute_force_latency,
        }
        pickle_save(dvfs_meta, meta_filename)
    def is_leaf(self):
        return True