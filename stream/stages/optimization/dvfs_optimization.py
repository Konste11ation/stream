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
from stream.cost_model.scheduler import accumulate_core_leakage_energy
from stream.cost_model.cost_model import StreamCostModelEvaluation
from zigzag.utils import pickle_save
from zigzag.utils import pickle_deepcopy
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
        origin_scme = pickle_deepcopy(self.scme)
        self.parse_dvfs()
        self.update_dvfs_info()
        base_energy = self.get_base_energy()
        base_latency = self.get_base_latency()
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
        yield origin_scme, opt_scme


    def parse_dvfs(self):
        self.dvfs_luts = self.dvfs_parser.run()
        
    def update_dvfs_info(self):
        self.dvfs_switching_latency = self.dvfs_luts['min_dvfs_switch_latency']
        self.system_clock_freq = self.dvfs_luts['system_clock_freq']
        
        max_latency_in_CC = int(self.get_base_latency() / min(self.dvfs_luts["freq_lut"].values()))
        dvfs_switching_latency_in_CC = int(self.dvfs_switching_latency * self.system_clock_freq * 1e6)
        if max_latency_in_CC < dvfs_switching_latency_in_CC:
            print("The workload latency is smaller than the dvfs switching latency, no dvfs optimization is performed")
        
        self.num_time_windows = max_latency_in_CC // dvfs_switching_latency_in_CC + 1
        # the number of cores
        core_ids = self.get_core_ids()
        self.num_cores = len(core_ids)
        
        for node in self.workload.node_list:
            node.set_vdd_lut(self.dvfs_luts["vdd_lut"])
            node.set_freq_lut(self.dvfs_luts["freq_lut"])
            node.set_energy_lut(self.dvfs_luts["dyn_energy_lut"])
        # set the static energy info to the accelerator
        self.accelerator.set_sta_energy_lut(self.dvfs_luts["sta_energy_lut"])
        # Now we set the default static power per core
        # currently we assume all the cores have the same static power
        # TODO: Directly parse the static power per core from the zigzag architecture spec
        default_core_uW = 100.0
        sta_power_per_core_uW = {core_id: default_core_uW for core_id in core_ids}
        self.accelerator.set_sta_power_per_core_uW(sta_power_per_core_uW)

    def get_base_latency(self):
        latency = max(n.end for n in self.workload.node_list)
        return latency
    def get_core_ids(self):
        core_ids = set()
        for node in self.workload.node_list:
            if node.chosen_core_allocation is not None:
                core_ids.add(node.chosen_core_allocation)
        return list(core_ids)
    def get_base_energy(self):
        latency_base = self.get_base_latency()
        
        energy_dyn = sum(n.get_onchip_energy() for n in self.workload.node_list)
        energy_sta = accumulate_core_leakage_energy(
            latency_cc=int(latency_base),
            system_clock_freq_ghz=self.system_clock_freq,
            dvfs_switching_latency_ms=self.dvfs_switching_latency,
            dvfs_allocations=None,
            sta_energy_lut=self.accelerator.get_sta_energy_lut(),
            static_power_per_core_uW=self.accelerator.get_sta_power_per_core_uW(),
            default_core_uW=100.0,
            default_dvfs_level=0,
        )
        base_energy = energy_dyn + energy_sta
        return base_energy


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
        fitness_evaluator = DvfsFitnessEvaluator(workload=self.workload,
                                                 accelerator=self.accelerator,
                                                 cost_lut=[],
                                                 operands_to_prefetch=[],
                                                 scheduling_order=self.scheduling_order,
                                                 dvfs_switching_latency=self.dvfs_switching_latency,
                                                 system_clock_freq=self.system_clock_freq,
                                                 vdd_lut=self.dvfs_luts["vdd_lut"])

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
            pop_init=pop_init
        )
        pop, hof = genetic_alg.run()
        best_individual = hof[-1]
        best_allocation = self.convert_individual_to_allocation(best_individual)

        results = fitness_evaluator.get_fitness(best_allocation, return_scme=True)
        return results, hof


    def create_initial_population(self, individual_length):
        """Create a diverse initial population for the genetic algorithm.

        Genome layout:
        - Individual is a flat list of length = num_cores * num_time_windows
        - Gene at index (core * num_time_windows + tw) is a DVFS level (int)

        Composition (in order; deduplicated; trimmed/padded to nb_ga_individuals):
        1) All-mid baseline
        2) All-min and all-max extremes
        3) Uniform-random individuals (per gene uniform over valid levels)
        4) Patterned individuals (staircase/zigzag per core)
        5) Heuristic-seeded individual (if feasible info absent, use balanced bias pattern)
        """
        pop_init: list[list[int]] = []
        valid_levels = sorted(self.dvfs_luts["vdd_lut"].keys())
        min_lvl = valid_levels[0]
        max_lvl = valid_levels[-1]
        mid_lvl = valid_levels[len(valid_levels) // 2]
        num_levels = len(valid_levels)
        rng = random.Random()
        # 1) mid baseline
        pop_init.append([mid_lvl] * individual_length)
        # 2) extremes
        pop_init.append([min_lvl] * individual_length)
        pop_init.append([max_lvl] * individual_length)
        # Helper to index (core, tw) -> flat idx
        def idx_of(core: int, tw: int) -> int:
            return core * self.num_time_windows + tw
        # 3) random individuals (50–70% of population target)
        target_pop = self.nb_ga_individuals
        remaining_slots = max(0, target_pop - len(pop_init) - 3)  # reserve for patterns + heuristic
        rand_count = max(4, remaining_slots // 2)  # ensure several randoms

        for _ in range(rand_count):
            individual = [rng.choice(valid_levels) for _ in range(individual_length)]
            pop_init.append(individual)
        # 4) patterned individuals to diversify linkage structure
        # 4.a Staircase decreasing over time windows per core
        stair_down = [mid_lvl] * individual_length
        for core in range(self.num_cores):
            for tw in range(self.num_time_windows):
                # map tw to a level index descending
                level_idx = max(0, num_levels - 1 - (tw * num_levels) // max(1, self.num_time_windows))
                stair_down[idx_of(core, tw)] = valid_levels[level_idx]
        pop_init.append(stair_down)
        # 4.b Staircase increasing
        stair_up = [mid_lvl] * individual_length
        for core in range(self.num_cores):
            for tw in range(self.num_time_windows):
                level_idx = min(num_levels - 1, (tw * num_levels) // max(1, self.num_time_windows))
                stair_up[idx_of(core, tw)] = valid_levels[level_idx]
        pop_init.append(stair_up)
        # 4.c Zigzag alternating low/high
        zigzag = [mid_lvl] * individual_length
        for core in range(self.num_cores):
            for tw in range(self.num_time_windows):
                zigzag[idx_of(core, tw)] = min_lvl if (tw % 2 == 0) else max_lvl
        pop_init.append(zigzag)
        # 4.d Checkerboard pattern across cores and time windows
        checker = [mid_lvl] * individual_length
        for core in range(self.num_cores):
            for tw in range(self.num_time_windows):
                if (core + tw) % 2 == 0:
                    checker[idx_of(core, tw)] = valid_levels[(num_levels // 3) if num_levels >= 3 else 0]
                else:
                    checker[idx_of(core, tw)] = valid_levels[(2 * num_levels // 3) if num_levels >= 3 else -1]
        pop_init.append(checker)
        # 5) heuristic-seeded individual
        heuristic = [mid_lvl] * individual_length
        for core in range(self.num_cores):
            for tw in range(self.num_time_windows):
                # bias: earlier windows lower levels, later windows higher levels
                if tw < self.num_time_windows // 2:
                    # choose from lower half
                    upper = max(1, num_levels // 2)
                    heuristic[idx_of(core, tw)] = rng.choice(valid_levels[:upper])
                else:
                    # choose from upper half
                    lower = max(1, num_levels // 2)
                    heuristic[idx_of(core, tw)] = rng.choice(valid_levels[lower:])
        pop_init.append(heuristic)
        # Deduplicate while preserving order
        seen = set() 
        unique_pop = []
        for ind in pop_init:
            t = tuple(ind)
            if t not in seen:
                unique_pop.append(ind)
                seen.add(t)
        # Truncate if too many
        if len(unique_pop) > self.nb_ga_individuals:
            unique_pop = unique_pop[: self.nb_ga_individuals]
        return unique_pop

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

    def plot_pareto(self, hof, base_energy_pJ, base_latency_cc):
        os.makedirs(self.dvfs_output_dir, exist_ok=True)
        fig_filename = os.path.join(self.dvfs_output_dir, "pareto.png")
        meta_filename = os.path.join(self.dvfs_output_dir, "dvfs_meta.pickle")
        plt.figure(figsize=(10, 6))

        pareto_front = hof

        # cycles -> ms: ms = cycles / (GHz * 1e6)
        denom_cc_to_ms = max(self.system_clock_freq * 1e6, 1e-9)

        # pJ -> mJ: mJ = pJ * 1e-9
        pj_to_mj = 1e-9

        pf_energy_mJ, pf_latency_ms = [], []
        if len(pareto_front) > 0:
            print(f"Plotting {len(pareto_front)} Points to Pareto Front")
            pf_energy_pJ = [ind.fitness.values[0] for ind in pareto_front]
            pf_latency_cc = [ind.fitness.values[1] for ind in pareto_front]

            pf_energy_mJ = [e * pj_to_mj for e in pf_energy_pJ]
            pf_latency_ms = [lat / denom_cc_to_ms for lat in pf_latency_cc]

            plt.scatter(pf_energy_mJ, pf_latency_ms,
                        c='red', s=80, edgecolors='black',
                        label='Pareto Front', zorder=2)

        base_energy_mJ = base_energy_pJ * pj_to_mj
        base_latency_ms = base_latency_cc / denom_cc_to_ms

        plt.scatter(base_energy_mJ, base_latency_ms,
                    c='blue', s=80, linewidths=2,
                    marker='x', label='Baseline', zorder=3)

        plt.xlabel('Energy (mJ)', fontsize=12)
        plt.ylabel('Latency (ms)', fontsize=12)

        plt.title('Pareto Front Visualization', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        if fig_filename:
            plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        dvfs_meta = {
            "pf_energy_mJ": pf_energy_mJ,
            "pf_latency_ms": pf_latency_ms,
            "base_energy_mJ": base_energy_mJ,
            "base_latency_ms": base_latency_ms
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
    