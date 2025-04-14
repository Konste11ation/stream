import itertools
import logging
import os
from time import time
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.parser.dvfs_parser import DvfsParser
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.hardware.architecture.accelerator import Accelerator
from collections import defaultdict
from stream.opt.allocation.genetic_algorithm.fitness_evaluator import DvfsFitnessEvaluator
from stream.opt.allocation.genetic_algorithm.genetic_algorithm import (
    DvfsGeneticAlgorithm,
)
logger = logging.getLogger(__name__)

class DvfsOptimizationStage(Stage):

    def __init__(self, 
                 list_of_callables: list[StageCallable],
                *,
                workload: ComputationNodeWorkload,
                accelerator: Accelerator,
                scheduling_order: list[tuple[int, int]],
                **kwargs
                ):
        super().__init__(list_of_callables, **kwargs)
        self.dvfs_output_fig_path = kwargs["dvfs_output_path"]
        self.workload = workload
        self.accelerator = accelerator
        self.scheduling_order = scheduling_order
        self.dvfs_parser = DvfsParser(kwargs["dvfs_cfg_path"])
        self.dvfs_luts = {}
        self.ga_nb_generations = kwargs["ga_nb_generations"]
        self.ga_nb_individuals = kwargs["ga_nb_individuals"]
    def run(self):
        logger.info("Start DvfsOptimizationStage.")
        base_energy = self.get_energy()
        base_latency = self.get_latency()

        self.parse_dvfs()
        self.update_dvfs_info()
        result, hof = self.run_dvfs_opt()
        opt_energy = result[0]
        opt_latency = result[1]
        opt_scme = result[2]
        print(f"Opt Energy = {opt_energy}, Opt Latency = {opt_latency}")
        logger.info("End DvfsOptimizationStage.")
        self.plot_pareto(hof,base_energy,base_latency)
        return opt_scme

    def parse_dvfs(self):
        self.dvfs_luts = self.dvfs_parser.run()
        
    def update_dvfs_info(self):
        for node in self.workload.node_list:
            node.set_vdd_lut(self.dvfs_luts["vdd_lut"])
            node.set_freq_lut(self.dvfs_luts["freq_lut"])
            node.set_energy_lut(self.dvfs_luts["energy_lut"])

    def get_energy(self):
        energy = sum(n.get_onchip_energy() for n in self.workload.node_list)
        return energy
    def get_latency(self):
        latency = max(n.end for n in self.workload.node_list)
        return latency
    def compute_dvfs_level(self, runtime, slack):
        freq_lut = self.dvfs_luts["freq_lut"]
        sorted_levels = sorted(freq_lut.keys(), reverse=True)
        for level in sorted_levels:
            freq_scaling = freq_lut[level]
            runtime_dvfs = int(runtime / freq_scaling)
            if runtime_dvfs <= runtime + slack:
                return level
        return min(freq_lut.keys())
    def get_communication_dic(self):
        """
        Return all the output transfer event as a dic
        key:(id,sub_id)
        value:{start, end, runtime, tensors}
        """
        active_links = set()
        node_events = {}
        for ky, link_pair in self.accelerator.communication_manager.pair_links.items():
            if link_pair:
                for link in link_pair:
                    if link.events:
                        active_links.add(link)
        for pair_link_id, cl in enumerate(active_links):
            for event in cl.events:
                start = event.start
                end = event.end
                runtime = end - start
                tensors = event.tensors
                node = event.tensors[0].origin
                tensor_type =  event.tensors[0].memory_operand
                node_id =  node.id
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
                    "Tensors": tensors 
                }
                node_events.setdefault(key, event_record)
        return node_events
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
    def brute_dvfs_opt(self):
        node_event_dic = self.get_communication_dic()
        start_time_per_core = self.get_start_time_per_core()
        for node in self.workload.node_list:
            cur_id = node.id
            cur_sub_id = node.sub_id
            cur_end = node.get_end()
            cur_runtime = node.get_runtime()
            cur_core = node.chosen_core_allocation
            successor_nodes = set(self.workload.successors(node))
            successor_nodes_start_times = [n.start for n in successor_nodes]
            # the current node is the exit node
            if successor_nodes_start_times == []:
                continue
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
            if slack>0:
                dvfs_level = self.compute_dvfs_level(cur_runtime,slack)
                node.dvfs_level=dvfs_level
    def run_dvfs_opt(self):
        # We use the brute force dvfs as the initial guess
        self.brute_dvfs_opt()
        fitness_evaluator = DvfsFitnessEvaluator(self.workload,
                                                 self.accelerator,
                                                 [],
                                                 [],
                                                 self.scheduling_order)
        individual_length = len([n for n in self.workload.node_list])
        valid_allocations = [min(self.dvfs_luts["vdd_lut"].keys()), max(self.dvfs_luts["vdd_lut"].keys())]
        pop_init = []
        dvfs_init = [n.dvfs_level for n in self.workload.node_list]
        pop_init.append(dvfs_init)
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
        results = fitness_evaluator.get_fitness(best_dvfs_level_allocation, return_scme=True)
        return results, hof

    def plot_pareto(self,hof,base_energy, base_latency):
        filename = self.dvfs_output_fig_path
        plt.figure(figsize=(10, 6))
        pareto_front = hof
        if len(pareto_front) > 0:
            pf_energy = [ind.fitness.values[0] for ind in pareto_front]
            pf_latency = [ind.fitness.values[1] for ind in pareto_front]
            plt.scatter(pf_energy, pf_latency, 
                        c='red', s=80, edgecolors='black',
                        label='Pareto Front', zorder=2) 
        plt.scatter(base_energy, base_latency,
                    c='green', s=80, edgecolors='black', linewidths=2,
                    marker='o', label='Base', zorder=4,
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        
        plt.xlabel('Energy Consumption', fontsize=12)
        plt.ylabel('Latency', fontsize=12)
        plt.title('Pareto Front Visualization', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
        else:
            plt.show()

    def is_leaf(self):
        return True