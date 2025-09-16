import itertools
import logging
import os
import random
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
from stream.cost_model.cost_model import StreamCostModelEvaluation
from zigzag.utils import pickle_save
logger = logging.getLogger(__name__)



# The DvfsOptimizationStage class
# This stage performs DVFS optimization using a genetic algorithm
# The input is a standard StreamCostModelEvaluation object and a dvfs configuration file
# The DVFS hapens at the core level
# We consider the real hw constraints such as the dvfs switching latency
# e.g. if min_dvfs_switch_latency = 1ms, that is to say we can change the dvfs level
# at most once per 1ms, and every tasks running on that core will follow this dvfs level

class DvfsOptimizationStage(Stage):

    def __init__(self, 
                 list_of_callables: list[StageCallable],
                 *,
                 scme: "StreamCostModelEvaluation",
                 **kwargs
                 ):
        super().__init__(list_of_callables, **kwargs)
        self.dvfs_output_dir = kwargs["dvfs_output_dir"]
        self.scme = scme
        self.workload = scme.workload
        self.accelerator = scme.accelerator
        self.scheduling_order = scme.scheduling_order
        self.dvfs_parser = DvfsParser(kwargs["dvfs_cfg_path"])
        self.dvfs_luts = {}
        self.num_cores = 0
        self.num_time_windows = 0
        self.dvfs_switching_latency = 1.0  # default 1ms
        self.system_clock_freq = 1.0  # default 1GHz
        self.nb_ga_generations = kwargs["nb_ga_generations_dvfs"]
        self.nb_ga_individuals = kwargs["nb_ga_individuals_dvfs"]

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
        print(f"Base Energy = {base_energy}, Base Latency = {base_latency}")
        print(f"Opt Energy = {opt_energy}, Opt Latency = {opt_latency}")
        logger.info("End DvfsOptimizationStage.")
        if hof:
            self.plot_pareto(hof, base_energy, base_latency)
            self.plot_dvfs_schedule(hof)
        yield self.scme, opt_scme


    def parse_dvfs(self):
        self.dvfs_luts = self.dvfs_parser.run()
        
    def update_dvfs_info(self):
        self.dvfs_switching_latency = self.dvfs_luts['min_dvfs_switch_latency']
        self.system_clock_freq = self.dvfs_luts['system_clock_freq']
        
        max_latency_in_CC = int(self.get_latency() / min(self.dvfs_luts["freq_lut"].values()))
        dvfs_switching_latency_in_CC = int(self.dvfs_switching_latency * self.system_clock_freq * 1e6)
        if max_latency_in_CC < dvfs_switching_latency_in_CC:
            print("The workload latency is smaller than the dvfs switching latency, no dvfs optimization is performed")
        
        self.num_time_windows = max_latency_in_CC // dvfs_switching_latency_in_CC + 1
        # the number of cores
        core_ids = self.get_core_ids()
        self.num_cores = len(core_ids)
        
        for node in self.workload.node_list:
            node.dvfs_level = 0
            node.set_vdd_lut(self.dvfs_luts["vdd_lut"])
            node.set_freq_lut(self.dvfs_luts["freq_lut"])
            node.set_energy_lut(self.dvfs_luts["energy_lut"])

    def get_energy(self):
        energy = sum(n.get_onchip_energy() for n in self.workload.node_list)
        return energy

    def get_latency(self):
        latency = max(n.end for n in self.workload.node_list)
        return latency

    def get_core_ids(self):
        core_ids = set()
        for node in self.workload.node_list:
            if node.chosen_core_allocation is not None:
                core_ids.add(node.chosen_core_allocation)
        return list(core_ids)

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
                tensor_type = event.tensors[0].memory_operand
                node_id = node.id
                node_sub_id = node.sub_id

                if runtime == 0:
                    continue
                if not tensor_type.is_output():
                    continue
                key = (node_id, node_sub_id)
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

    def find_next_start_time_per_core(self, start_time_per_core, core, end_time):
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

    def brute_task_level_dvfs_opt(self):
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
            deadline = min(est_successors - output_transfer_time,est_core)
            slack = deadline - cur_end
            if slack>0:
                dvfs_level = self.compute_dvfs_level(cur_runtime, slack)
                node.dvfs_level = dvfs_level

    def run_dvfs_opt(self):
        fitness_evaluator = DvfsFitnessEvaluator(self.workload,
                                                 self.accelerator,
                                                 [],
                                                 [],
                                                 self.scheduling_order,
                                                 self.dvfs_switching_latency,
                                                 self.system_clock_freq)
        
        # determine the number of time slots

        valid_allocations = list(self.dvfs_luts["vdd_lut"].keys())
        individual_length = self.num_cores * self.num_time_windows
        pop_init = self.create_initial_population(individual_length)

        genetic_alg = DvfsGeneticAlgorithm(
            fitness_evaluator=fitness_evaluator,
            num_cores=self.num_cores,
            num_time_windows=self.num_time_windows,
            valid_allocations=valid_allocations,
            num_generations=self.nb_ga_generations,
            num_individuals=self.nb_ga_individuals,
            pop=pop_init
        )
        pop, hof = genetic_alg.run()
        best_individual = hof[-1]
        best_allocation = self.convert_individual_to_allocation(best_individual)

        results = fitness_evaluator.get_fitness(best_allocation, return_scme=True)
        return results, hof
    



    def create_initial_population(self, individual_length):
        """Create an initial population for the genetic algorithm."""
        pop_init = []

        # Create an individual where all cores use the middle DVFS level for all time windows
        min_dvfs_level = min(self.dvfs_luts["vdd_lut"].keys())
        max_dvfs_level = max(self.dvfs_luts["vdd_lut"].keys())
        mid_dvfs_level = (min_dvfs_level + max_dvfs_level) // 2
        individual = [mid_dvfs_level] * individual_length
        pop_init.append(individual)

        # More heuristic initial individuals can be added here
        # For example, based on the results of a brute-force DVFS optimization

        return pop_init

    def convert_individual_to_allocation(self, individual):
        """Convert a 1D array to an allocation dictionary."""
        allocation = {}
        idx = 0
        for core in range(self.num_cores):
            allocation[core] = {}
            for time_window in range(self.num_time_windows):
                allocation[core][time_window] = individual[idx]
                idx += 1
        return allocation

    def plot_pareto(self, hof, base_energy, base_latency):
        os.makedirs(self.dvfs_output_dir, exist_ok=True)
        fig_filename = os.path.join(self.dvfs_output_dir, "pareto.png")
        meta_filename = os.path.join(self.dvfs_output_dir, "dvfs_meta.pickle")
        plt.figure(figsize=(10, 6))
        pareto_front = hof
        if len(pareto_front) > 0:
            print(f"Plotting {len(pareto_front)} Points to Pareto Front")
            pf_energy = [ind.fitness.values[0] for ind in pareto_front]
            pf_latency = [ind.fitness.values[1] for ind in pareto_front]
            plt.scatter(pf_energy, pf_latency, 
                        c='red', s=80, edgecolors='black',
                        label='Pareto Front', zorder=2) 
        # now plot the naive points
        dvfs_levels = sorted(self.dvfs_luts["vdd_lut"].keys())

        naive_dvfs_energy = [base_energy * self.dvfs_luts["energy_lut"][level] for level in dvfs_levels]
        naive_dvfs_latency = [int(base_latency / self.dvfs_luts["freq_lut"][level]) for level in dvfs_levels]

        plt.scatter(naive_dvfs_energy, naive_dvfs_latency,
                    c='blue', s=80, linewidths=2,
                    marker='x', label='Naive DVFS Levels', zorder=3)

        plt.scatter(base_energy, base_latency,
                    c='green', s=80, edgecolors='black', linewidths=2,
                    marker='o', label='Base', zorder=4,
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")])

        plt.xlabel('Energy Consumption', fontsize=12)
        plt.ylabel('Latency', fontsize=12)

        plt.title('Pareto Front Visualization', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        if fig_filename:
            plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        dvfs_meta = {
            "pf_energy": pf_energy,
            "pf_latency": pf_latency,
            "base_energy": base_energy,
            "base_latency": base_latency
        }
        pickle_save(dvfs_meta, meta_filename)
        
        
    def plot_dvfs_schedule(self, hof):
        # Draw 5 individuals from the Pareto front
        os.makedirs(self.dvfs_output_dir, exist_ok=True)
                
        hof_length = min(5, len(hof))
        selected_individuals = hof[-hof_length:] 
        for idx, individual in enumerate(selected_individuals):
            best_allocation = self.convert_individual_to_allocation(individual)
            plt.figure(figsize=(12, 6))
            
            fig_filename = os.path.join(self.dvfs_output_dir, f"dvfs_schedule_{idx}.png")
            
            time_window_duration = self.dvfs_switching_latency
            for core in range(self.num_cores):
                time_windows = list(best_allocation[core].keys())
                dvfs_levels = [best_allocation[core][tw] for tw in time_windows]
                times = [tw * time_window_duration for tw in time_windows]
                plt.step(times, dvfs_levels, where='post', label=f'Core {core}')
            plt.xlabel('Time (ms)', fontsize=12)
            plt.ylabel('DVFS Level', fontsize=12)
            plt.title('DVFS Schedule per Core', fontsize=14)
            plt.yticks(sorted(self.dvfs_luts["vdd_lut"].keys()))
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            if fig_filename:
                plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
            else:
                plt.show()

    def is_leaf(self):
        return True
    