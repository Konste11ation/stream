import logging
import multiprocessing
import os
import pickle
from itertools import product
from collections import defaultdict
from typing import Any

import matplotlib
matplotlib.use('Agg')
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
from stream.visualization.perfetto import convert_scme_to_perfetto_json
logger = logging.getLogger(__name__)


global_baseline_fitness_evaluator = None
global_baseline_core_allocs = None
global_baseline_active_cores = None
global_baseline_level = 0
global_baseline_dyn_power_lut = None
global_baseline_fixed_node_schedule = None

def init_baseline_worker(
    evaluator,
    core_allocs,
    active_cores,
    baseline_level,
    dyn_power_lut,
    fixed_node_schedule=None,
):
    """Initialize worker process context for per-core baseline evaluation."""
    global global_baseline_fitness_evaluator
    global global_baseline_core_allocs
    global global_baseline_active_cores
    global global_baseline_level
    global global_baseline_dyn_power_lut
    global global_baseline_fixed_node_schedule

    global_baseline_fitness_evaluator = evaluator
    global_baseline_core_allocs = core_allocs
    global_baseline_active_cores = active_cores
    global_baseline_level = baseline_level
    global_baseline_dyn_power_lut = dyn_power_lut
    global_baseline_fixed_node_schedule = fixed_node_schedule

def evaluate_baseline_assignment(assignment: tuple[int, ...]) -> tuple[float, float]:
    """Evaluate one per-core DVFS assignment in a worker process."""
    evaluator = global_baseline_fitness_evaluator
    core_allocs = global_baseline_core_allocs
    active_cores = global_baseline_active_cores
    baseline_level = global_baseline_level
    dyn_power_lut = global_baseline_dyn_power_lut
    fixed_node_schedule = global_baseline_fixed_node_schedule

    if evaluator is None or core_allocs is None or active_cores is None or dyn_power_lut is None:
        raise RuntimeError("Baseline worker context is not initialized.")

    core_to_level = {core: assignment[idx] for idx, core in enumerate(active_cores)}
    for node in evaluator.workload.node_list:
        core = node.chosen_core_allocation
        node_level = core_to_level.get(int(core), baseline_level) if core is not None else baseline_level
        node.set_dvfs_level(node_level)
        node.set_dyn_power_lut(dyn_power_lut)
        node.set_dvfs_mode("PerCoreBaseline")

    energy, latency = evaluator.get_fitness(core_allocs, fixed_node_schedule=fixed_node_schedule)
    return float(energy), float(latency)


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
        self.dvfs_switching_speed = kwargs.get("dvfs_switching_speed", 200)
        self.output_path = kwargs.get("output_path", "outputs")
        self.num_procs = kwargs.get("num_procs", 1)
        # Tuning parameters for GA
        self.prob_crossover = kwargs.get("prob_crossover", 0.7)
        self.prob_mutation = kwargs.get("prob_mutation", 0.2)
        self.fitness_cache_size = kwargs.get("fitness_cache_size", 200_000)
        self.early_stopping_patience = kwargs.get("early_stopping_patience", 0)
        self.early_stopping_min_generations = kwargs.get("early_stopping_min_generations", 0)
        self.max_baseline_combinations = kwargs.get("max_baseline_combinations", 5_000)
        self.baseline_combo_sample_budget = kwargs.get("baseline_combo_sample_budget", 2_000)
        self.baseline_combo_seed = kwargs.get("baseline_combo_seed", 0)
        self.force_exhaustive_seed_baseline = kwargs.get("force_exhaustive_seed_baseline", True)
        self.max_seed_baseline_state_space = kwargs.get("max_seed_baseline_state_space", 500_000)
        self.enable_baseline_disk_cache = kwargs.get("enable_baseline_disk_cache", True)
        self.baseline_cache_file = kwargs.get(
            "baseline_cache_file", os.path.join(self.output_path, "baseline_sweep_cache.pkl")
        )
        self._baseline_sweep_cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self._preferred_baseline_core_allocs: tuple[int, ...] | None = None
        self._load_baseline_sweep_cache()

        self.flexible_nodes: list[ComputationNode] = []
        for n in self.workload.node_list:
            if not isinstance(n.chosen_core_allocation, int):
                self.flexible_nodes.append(n)

        # For each unique node get the possible core allocations by getting the ids of the cores in cost_lut
        self.valid_allocations: list[list[int]] = []
        for flexible_node in self.flexible_nodes:
            self.valid_allocations.append(flexible_node.core_allocation)

        if self.do_dvfs_cooptimization and self.dvfs_config_path:
            from stream.parser.dvfs_parser import DvfsParser

            dvfs_parser = DvfsParser(self.dvfs_config_path)
            self.dvfs_luts = dvfs_parser.parse_dvfs_data()
            
            # --- DVFS Mode Configuration ---
            # All nodes use DVFS with standard switching penalty
            self.dvfs_level_choices = sorted(self.dvfs_luts["freq_lut"].keys())
            
            # Initial setup of nodes with default LUTs
            # Since we removed the threshold logic, all nodes share the same DVFS configuration space
            sys_clock = self.dvfs_luts.get("system_clock_mhz", 1000)
            sta_power = self.dvfs_luts.get("base_static_power_mw", None)
            self.dvfs_switching_speed = self.dvfs_luts.get("dvfs_switching_speed", self.dvfs_switching_speed)

            for node in self.workload.node_list:
                node.set_dvfs_level(0) # Default to max performance (level 0 usually)
                node.set_vdd_lut(self.dvfs_luts["vdd_lut"])
                node.set_freq_lut(self.dvfs_luts["freq_lut"])
                node.set_dyn_power_lut(self.dvfs_luts["dyn_power_lut"])
                node.set_sta_power_lut(self.dvfs_luts["sta_power_lut"])
                node.set_dvfs_mode("DVFS")
                
                # Apply absolute static power from CACTI if available
                node.system_clock_mhz = sys_clock
                if sta_power is not None:
                    node.set_absolute_static_power(sta_power)

            self.always_short_nodes_dvfs: list[ComputationNode] = []
            self.flexible_nodes_dvfs: list[ComputationNode] = []
            for node in self.workload.node_list:
                possible_cores = (
                    node.core_allocation
                    if isinstance(node.core_allocation, list)
                    else [node.chosen_core_allocation]
                )
                possible_cores = [core_id for core_id in possible_cores if core_id is not None]

                latencies = []
                equal_unique_node = self.cost_lut.get_equal_node(node) or node
                for core_id in possible_cores:
                    core = self.accelerator.get_core(core_id)
                    cme = self.cost_lut.get_cme(equal_unique_node, core)
                    latencies.append(int(getattr(cme, self.latency_attr)))

                if latencies and max(latencies) < self.dvfs_switching_speed:
                    self.always_short_nodes_dvfs.append(node)
                else:
                    self.flexible_nodes_dvfs.append(node)

            logger.info(
                "DVFS GA encoding reduced by switching threshold %s cycles: %s encoded (long), %s skipped (always short), %s total.",
                self.dvfs_switching_speed,
                len(self.flexible_nodes_dvfs),
                len(self.always_short_nodes_dvfs),
                len(self.workload.node_list),
            )
            
            # Populate valid allocations for the gene space
            for node in self.flexible_nodes_dvfs:
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
                self.dvfs_switching_speed,
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
        self.genetic_algorithm = None

        if not self.do_dvfs_cooptimization and self.individual_length > 0:
            self.genetic_algorithm = self._build_genetic_algorithm()

    def _build_genetic_algorithm(self, pop=None):
        """Build a GA instance using the stage's shared tuning parameters."""
        if pop is None:
            pop = []

        return GeneticAlgorithm(
            self.fitness_evaluator,
            self.individual_length,
            self.valid_allocations,
            self.nb_generations,
            self.nb_individuals,
            pop=pop,
            num_processes=self.num_procs,
            prob_crossover=self.prob_crossover,
            prob_mutation=self.prob_mutation,
            fitness_cache_size=self.fitness_cache_size,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_generations=self.early_stopping_min_generations,
        )

    def run(self):
        """
        Run the InterCoreMappingStage.
        When DVFS Co-Optimization is enabled, this executes in 3 explicit stages:
        - STAGE 1: Evaluate a level-0 nominal baseline mapping.
        - STAGE 2: Exhaustive/Sampled per-core post-scheduling DVFS sweep based on the baseline mapping.
                   Outputs from Stage 1 & 2 are persisted and their Pareto fronts extracted to seed Stage 3.
        - STAGE 3: Run the Genetic Algorithm (GA) to simultaneously co-optimize Layer-to-Core allocation 
                   and Per-Node DVFS. Finally plot/save best configurations and comparisons.
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

            hof = None
            if self.do_dvfs_cooptimization:
                # Stage 1: Evaluate a level-0 nominal baseline mapping.
                self._run_stage_1_baseline()
                # Stage 2: Exhaustive/Sampled per-core post-scheduling DVFS sweep based on the baseline mapping.
                if self._preferred_baseline_core_allocs is not None:
                    self._run_stage_2_baseline_sweep(list(self._preferred_baseline_core_allocs))
                # Prepare seeds for Stage 3 based on Stage 1 & 2 results
                if self._preferred_baseline_core_allocs is not None:
                    hof = self._run_stage_3_ga_optimization(list(self._preferred_baseline_core_allocs))

                final_scme = self.plot_comparison(hof)
                if final_scme:
                    yield final_scme, None
                    logger.info("Finished GeneticAlgorithmAllocationStage.")
                    return
            else:
                if self.genetic_algorithm is None:
                    self.genetic_algorithm = self._build_genetic_algorithm()
                _, hof = self.genetic_algorithm.run()

            if not hof:
                logger.warning("Genetic algorithm did not produce a hall of fame result.")
                logger.info("Finished GeneticAlgorithmAllocationStage.")
                return

            # Return the SCME of the last individual in the hall of fame
            best_core_allocations = hof[-1]
            res = self.fitness_evaluator.get_fitness(best_core_allocations, return_scme=True)
            scme = res[2] if len(res) == 3 else None
            if scme:
                logger.info(f"Best allocation energy: {scme.energy:.2e}, latency: {scme.latency:.2e}")
                yield scme, None
        logger.info("Finished GeneticAlgorithmAllocationStage.")

    def _run_stage_1_baseline(self):
        """
        [STAGE 1]: NOMINAL BASELINE EVALUATION
        Now uses a mini Genetic Algorithm (GA) to find a high-quality layer-to-core allocation
        while keeping DVFS levels fixed at the baseline level (nominal frequency).
        This established mapping is then used as the anchor for the Stage 2 per-core sweep.
        """
        import random
        baseline_level = 0 if 0 in self.dvfs_level_choices else min(self.dvfs_level_choices)
        logger.info("Starting Stage 1: Finding anchor mapping via core-only GA (DVFS level=%s)...", baseline_level)

        # 1. Temporarily force all nodes to baseline DVFS level
        original_valid_allocations = self.valid_allocations
        num_flex_cores = len(self.flexible_nodes)
        num_flex_dvfs = len(self.flexible_nodes_dvfs)
        
        # Chromosome for core-only GA: [core_gene_0, ..., core_gene_N, dvfs_gene_0, ..., dvfs_gene_M]
        # We constrain the DVFS genes to only allow the baseline_level.
        constrained_allocations = []
        for i, choices in enumerate(original_valid_allocations):
            if i < num_flex_cores:
                constrained_allocations.append(choices) # Core genes remain flexible
            else:
                constrained_allocations.append([baseline_level]) # DVFS genes fixed
        
        # 2. Initialize a "mini" GA for core-only optimization
        mini_ga = GeneticAlgorithm(
            self.fitness_evaluator,
            self.individual_length,
            constrained_allocations,
            num_generations=min(self.nb_generations, 50), # Fewer generations for baseline
            num_individuals=self.nb_individuals,
            num_processes=self.num_procs,
            prob_crossover=self.prob_crossover,
            prob_mutation=self.prob_mutation,
            fitness_cache_size=self.fitness_cache_size,
        )
        
        # 3. Run the core-only GA
        _, hof = mini_ga.run()
        best_individual = hof[-1]
        
        # Extract the core-allocation subset from the best individual
        anchor_core_allocs = list(best_individual[:num_flex_cores])
        self._preferred_baseline_core_allocs = tuple(anchor_core_allocs)
        
        # 4. Save results for Stage 1
        stage1_dir = os.path.join(self.output_path, 'stage1_base')
        os.makedirs(stage1_dir, exist_ok=True)

        # Plot Pareto Front for Stage 1
        import matplotlib.pyplot as plt
        pf_energies = []
        pf_latencies = []
        for ind in hof:
            res = self.fitness_evaluator.get_fitness(ind, return_scme=False)
            pf_energies.append(res[0])
            pf_latencies.append(res[1])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(pf_latencies, pf_energies, c='blue', label='Pareto Front (Core-only)')
        ax.set_xlabel("Latency (Cycles)")
        ax.set_ylabel("Energy (uJ)")
        ax.set_title("Stage 1: Core-Allocation Pareto Front (Fixed Nominal DVFS)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.savefig(os.path.join(stage1_dir, "stage1_pareto.png"))
        plt.close()

        # Save nominal baseline SCME (Stage 1 Best result)
        res = self.fitness_evaluator.get_fitness(best_individual, return_scme=True)
        nominal_scme = res[2] if len(res) == 3 else None
        
        if nominal_scme:
            self._save_scme_to_json(nominal_scme, os.path.join(stage1_dir, 'stage1_best_edp.json'))
            logger.info("Stage 1 anchor mapping found. Best Energy: %.2e, Latency: %.2e", 
                        nominal_scme.energy, nominal_scme.latency)
        else:
            logger.error("Stage 1 nominal evaluation failed to produce SCME.")

    def _run_stage_2_baseline_sweep(self, core_allocs_list: list[int]):
        """
        [STAGE 2]: EXHAUSTIVE FREQUENCY SWEEP
        Perform a sweep of per-core DVFS levels for the fixed anchor mapping.
        Saves the best EDP SCME and an exhaustive search plot to the stage2 directory.
        """
        import multiprocessing
        import os
        import pickle
        from stream.visualization.perfetto import convert_scme_to_perfetto_json
        
        cache_key = tuple(core_allocs_list)
        if cache_key in self._baseline_sweep_cache:
            logger.info("Stage 2 cache hit for baseline sweep.")
            return
        
        global_energies: list[float] = []
        global_latencies: list[float] = []
        global_assignments: list[tuple[int, ...]] = []
        
        active_cores = sorted(
            core.id
            for core in self.accelerator.cores.node_list
            if core.id != self.accelerator.offchip_core_id
        )
        if not active_cores: active_cores = [0]
        
        baseline_assignments, exhaustive_baseline = self.generate_baseline_core_dvfs_assignments(active_cores)
        baseline_level = 0 if 0 in self.dvfs_level_choices else min(self.dvfs_level_choices)
        total_baseline_assignments = len(baseline_assignments)
        
        self.fitness_evaluator.set_node_core_allocations(core_allocations=core_allocs_list)
        
        if exhaustive_baseline and self.num_procs > 1 and total_baseline_assignments > 1:
            worker_count = min(self.num_procs, total_baseline_assignments)
            chunksize = max(1, total_baseline_assignments // (worker_count * 4))
            with multiprocessing.Pool(
                processes=worker_count,
                initializer=init_baseline_worker,
                initargs=(self.fitness_evaluator, core_allocs_list, active_cores, baseline_level, self.dvfs_luts["dyn_power_lut"], None),
            ) as pool:
                for assignment, result in zip(baseline_assignments, pool.imap(evaluate_baseline_assignment, baseline_assignments, chunksize=chunksize)):
                    energy, latency = result
                    global_energies.append(energy)
                    global_latencies.append(latency)
                    global_assignments.append(assignment)
        else:
            for assignment in baseline_assignments:
                core_to_level = {core: assignment[i] for i, core in enumerate(active_cores)}
                for node in self.workload.node_list:
                    c = node.chosen_core_allocation
                    node.set_dvfs_level(core_to_level.get(int(c), baseline_level) if c is not None else baseline_level)
                g_res = self.fitness_evaluator.get_fitness(core_allocs_list)
                global_energies.append(float(g_res[0]))
                global_latencies.append(float(g_res[1]))
                global_assignments.append(assignment)
        
        # Find best EDP point from sweep
        global_edps = [e * l for e, l in zip(global_energies, global_latencies)]
        best_idx = global_edps.index(min(global_edps))
        best_assignment = global_assignments[best_idx]
        
        # Save Stage 2 Specific Outputs
        stage2_dir = os.path.join(self.output_path, 'stage2_dvfs_sweep')
        os.makedirs(stage2_dir, exist_ok=True)
        
        # 1. Save results to cache
        baseline_result = {
            "global_energies": global_energies,
            "global_latencies": global_latencies,
            "global_assignments": global_assignments,
            "active_cores": active_cores,
            "exhaustive_baseline": exhaustive_baseline,
            "base_energy": global_energies[0],
            "base_latency": global_latencies[0],
        }
        self._baseline_sweep_cache[cache_key] = baseline_result
        self._save_baseline_sweep_cache()
        
        # 2. Re-evaluate best EDP assignment to get SCME and save it
        core_to_level_best = {core: best_assignment[i] for i, core in enumerate(active_cores)}
        for node in self.workload.node_list:
            c = node.chosen_core_allocation
            node.set_dvfs_level(core_to_level_best.get(int(c), baseline_level) if c is not None else baseline_level)
        
        best_res = self.fitness_evaluator.get_fitness(core_allocs_list, return_scme=True)
        if len(best_res) == 3:
            best_scme = best_res[2]
            scme_save_path = os.path.join(stage2_dir, 'stage2_best_edp.json')
            convert_scme_to_perfetto_json(best_scme, self.cost_lut, scme_save_path)
            with open(os.path.join(stage2_dir, "stage2_best_edp.pkl"), "wb") as f:
                pickle.dump(best_scme, f)
        
        # 3. Save exhaustive plot
        self.plot_exhaustive_baseline_only(
            global_energies,
            global_latencies,
            best_idx,
            global_energies[0], 
            global_latencies[0],
            os.path.join(stage2_dir, "stage2_exhaustive_search.png")
        )
        
        logger.info("Stage 2 baseline sweep completed.")

    def _run_stage_3_ga_optimization(self, anchor_mapping: list[int]):
        """
        [STAGE 3]: GENETIC ALGORITHM (CO-OPTIMIZATION SEARCH)
        Initialize the GA using Pareto-optimal seeds extracted from Stage 2 results.
        """
        import random
        
        # 1. Prepare GA Seeds directly using Stage 2 results
        pop_seeds = []
        num_flex = len(self.flexible_nodes)
        num_dvfs = len(self.flexible_nodes_dvfs)
        
        cache_key = tuple(anchor_mapping)
        baseline_data = self._baseline_sweep_cache.get(cache_key)
        
        if baseline_data:
            # Extract Pareto-guided seeds from the cached sweep results
            # This follows the logic of get_baseline_best_edp_seed_chromosomes but using in-memory baseline_data
            energies = baseline_data["global_energies"]
            latencies = baseline_data["global_latencies"]
            assignments = baseline_data["global_assignments"]
            
            pareto_assignments = self.extract_pareto_assignments(energies, latencies, assignments)
            for assignment in pareto_assignments:
                # Chromosome = [anchor_mapping] + [per-core-to-per-node mapping]
                # Stage 2 sweep uses per-core assignments, we need to map them back to nodes
                active_cores = baseline_data["active_cores"]
                core_to_level = {core: assignment[i] for i, core in enumerate(active_cores)}
                
                dvfs_genes = []
                baseline_level = 0 if 0 in self.dvfs_level_choices else min(self.dvfs_level_choices)
                for node in self.flexible_nodes_dvfs:
                    c = node.chosen_core_allocation
                    level = core_to_level.get(int(c), baseline_level) if c is not None else baseline_level
                    dvfs_genes.append(level)
                
                pop_seeds.append(anchor_mapping + dvfs_genes)
            
            logger.info("Initialized GA with %s seeds from Stage 2 Pareto results.", len(pop_seeds))
        else:
            logger.warning("No Stage 2 results found for %s! GA starting with default population.", cache_key)

        # 2. Add extra 'blanket' seeds for diversification
        for level in self.dvfs_level_choices:
            core_genes = [random.choice(self.valid_allocations[i]) for i in range(num_flex)]
            dvfs_genes = [level for _ in range(num_dvfs)]
            pop_seeds.append(core_genes + dvfs_genes)
        
        # 3. Initialize and run GA
        self.genetic_algorithm = self._build_genetic_algorithm(pop=pop_seeds)
        pop, hof = self.genetic_algorithm.run()
        logger.info("Finished Genetic Algorithm.")
        return hof

    def is_leaf(self) -> bool:
        return True

    def plot_comparison(self, hall_of_fame):
        """
        Extract results from Stage 3 and compare them against Stage 1/2 baselines.
        Generates Pareto plots and saves the top-performing co-optimized configurations.
        """
        import os
        import pickle
        from stream.visualization.perfetto import convert_scme_to_perfetto_json

        stage3_dir = os.path.join(self.output_path, 'stage3_co')
        os.makedirs(stage3_dir, exist_ok=True)

        # 1. Extract GA Pareto points (Energy, Latency)
        pf_energies = [ind.fitness.values[0] for ind in hall_of_fame]
        pf_latencies = [ind.fitness.values[1] for ind in hall_of_fame]

        # 2. Retrieve Stage 2 Baseline Data from Cache
        anchor_key = tuple(self._preferred_baseline_core_allocs) if self._preferred_baseline_core_allocs else None
        baseline_data = self._baseline_sweep_cache.get(anchor_key)
        
        if not baseline_data:
            logger.error("Stage 2 result missing from cache; comparison aborted.")
            return None

        # 3. Compute Comparison Metrics (Area Under Curve, EDP improvements, etc.)
        metrics = self.compute_comparison_metrics(
            pf_energies, pf_latencies,
            baseline_data["global_energies"], baseline_data["global_latencies"],
            baseline_data["global_assignments"],
            baseline_data["base_energy"], baseline_data["base_latency"]
        )

        # 4. Logging Summary
        logger.info("="*40)
        logger.info("DVFS CO-OPTIMIZATION SUMMARY")
        logger.info(f"Baseline (Nominal) EDP: {metrics['baseline_edp']:.4f}")
        logger.info(f"Best Stage 2 Sweep EDP: {metrics['best_global_edp']:.4f}")
        logger.info(f"Best Stage 3 GA EDP   : {metrics['best_ga_edp']:.4f}")
        logger.info(f"Overall EDP Improvement: {(metrics['baseline_edp'] - metrics['best_ga_edp'])/metrics['baseline_edp']*100:.1f}%")
        
        improvement_15x = metrics['energy_at_15x_improvement']
        improvement_15x_str = f"{improvement_15x:.1f}%" if improvement_15x is not None else "N/A"
        logger.info(f"Energy Improvement @ 1.5x Latency: {improvement_15x_str}")
        
        ga_auc = metrics['ga_auc']
        global_auc = metrics['global_auc']
        auc_improvement = ((global_auc - ga_auc) / global_auc * 100) if global_auc else 0.0
        logger.info(f"Pareto AUC (Global / GA): {global_auc:.4f} / {ga_auc:.4f} ({auc_improvement:.1f}% improvement)")
        logger.info("="*40)

        # 5. Generate Comparison Plot
        plot_data = {
            **metrics, 
            "exhaustive_baseline": baseline_data["exhaustive_baseline"],
            "base_energy": baseline_data["base_energy"],
            "base_latency": baseline_data["base_latency"]
        }
        self.plot_comparison_figure(plot_data, os.path.join(stage3_dir, 'dvfs_comparison.png'))

        # 6. Save Best SCME (The 'Winning' Configuration)
        best_ind = hall_of_fame[metrics['best_ga_edp_idx']]
        res = self.fitness_evaluator.get_fitness(best_ind, return_scme=True)
        final_scme = res[2] if len(res) == 3 else None

        if final_scme:
            # Attach metrics metadata to the SCME object for downstream tools
            for key, val in metrics.items():
                setattr(final_scme, key, val) 
            
            # Save to Stage 3 Dir
            self._save_scme_to_json(final_scme, os.path.join(stage3_dir, 'stage3_best_edp.json'))
            with open(os.path.join(stage3_dir, 'stage3_best_edp.pkl'), "wb") as f:
                pickle.dump(final_scme, f)
            
            # Generate Timeline Visualization for the best co-optimized point
            self.export_core_dvfs_timeline(
                final_scme, 
                os.path.join(stage3_dir, "stage3_best_timeline.png"),
                title="Co-Optimized Core DVFS Timeline"
            )

        return final_scme

    def _load_baseline_sweep_cache(self):
        """Load baseline sweep cache from disk if enabled and available."""
        if not self.enable_baseline_disk_cache:
            return
        if not os.path.exists(self.baseline_cache_file):
            return
        try:
            with open(self.baseline_cache_file, "rb") as handle:
                loaded = pickle.load(handle)
            if isinstance(loaded, dict):
                self._baseline_sweep_cache = loaded
                logger.info(
                    "Loaded baseline sweep cache: %s entries from %s",
                    len(self._baseline_sweep_cache),
                    self.baseline_cache_file,
                )
        except Exception as exc:
            logger.warning("Failed to load baseline sweep cache from %s: %s", self.baseline_cache_file, exc)

    def _save_baseline_sweep_cache(self):
        """Persist baseline sweep cache to disk if enabled."""
        if not self.enable_baseline_disk_cache:
            return
        try:
            os.makedirs(os.path.dirname(self.baseline_cache_file), exist_ok=True)
            with open(self.baseline_cache_file, "wb") as handle:
                pickle.dump(self._baseline_sweep_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.warning("Failed to save baseline sweep cache to %s: %s", self.baseline_cache_file, exc)

    def _calculate_auc(self, points: list[tuple[float, float]], min_e_bound: float = 0.0) -> float:
        filtered_points = [p for p in points if p[0] >= min_e_bound]
        if len(filtered_points) < 2:
            return 0.0
        area = 0.0
        for i in range(len(filtered_points) - 1):
            x1, y1 = filtered_points[i]
            x2, y2 = filtered_points[i + 1]
            area += (x2 - x1) * (y1 + y2) / 2.0
        return area

    def _get_value_at_target(
        self,
        target_val: float,
        points: list[tuple[float, float]],
        x_axis: str = "energy",
    ) -> float | None:
        """Interpolate/extrapolate Y at target X from 2D points."""
        if not points:
            return None

        idx_x = 0 if x_axis == "energy" else 1
        idx_y = 1 if x_axis == "energy" else 0
        eval_points = sorted(points, key=lambda p: p[idx_x])

        min_x = eval_points[0][idx_x]
        max_x = eval_points[-1][idx_x]

        if target_val < min_x:
            if len(eval_points) < 2:
                return eval_points[0][idx_y]
            p1, p2 = eval_points[0], eval_points[1]
            x1, y1 = p1[idx_x], p1[idx_y]
            x2, y2 = p2[idx_x], p2[idx_y]
            if x2 == x1:
                return y1
            return y1 + ((y2 - y1) / (x2 - x1)) * (target_val - x1)

        if target_val > max_x:
            if len(eval_points) < 2:
                return eval_points[-1][idx_y]
            p1, p2 = eval_points[-2], eval_points[-1]
            x1, y1 = p1[idx_x], p1[idx_y]
            x2, y2 = p2[idx_x], p2[idx_y]
            if x2 == x1:
                return y1
            return y2 + ((y2 - y1) / (x2 - x1)) * (target_val - x2)

        for i in range(len(eval_points) - 1):
            p1, p2 = eval_points[i], eval_points[i + 1]
            x1, y1 = p1[idx_x], p1[idx_y]
            x2, y2 = p2[idx_x], p2[idx_y]
            if x1 <= target_val <= x2:
                if x2 == x1:
                    return y1
                ratio = (target_val - x1) / (x2 - x1)
                return y1 + ratio * (y2 - y1)
        return None

    def compute_comparison_metrics(
        self,
        pf_energies: list[float],
        pf_latencies: list[float],
        global_energies: list[float],
        global_latencies: list[float],
        global_assignments: list[tuple[int, ...]],
        base_energy: float | None,
        base_latency: float | None,
    ) -> dict[str, Any]:
        """Compute normalized curves and scalar comparison metrics independent from plotting."""
        pf_energies_norm = list(pf_energies)
        pf_latencies_norm = list(pf_latencies)
        global_energies_norm = list(global_energies)
        global_latencies_norm = list(global_latencies)

        if base_energy and base_latency:
            pf_energies_norm = [e / base_energy for e in pf_energies_norm]
            pf_latencies_norm = [l / base_latency for l in pf_latencies_norm]
            global_energies_norm = [e / base_energy for e in global_energies_norm]
            global_latencies_norm = [l / base_latency for l in global_latencies_norm]

        base_energy_norm, base_latency_norm = 1.0, 1.0
        baseline_edp = base_energy_norm * base_latency_norm

        ga_edps = [e * l for e, l in zip(pf_energies_norm, pf_latencies_norm)]
        best_ga_edp_idx = ga_edps.index(min(ga_edps))
        best_ga_edp = ga_edps[best_ga_edp_idx]

        global_edps = [e * l for e, l in zip(global_energies_norm, global_latencies_norm)]
        best_global_edp_idx = global_edps.index(min(global_edps))
        best_global_edp = global_edps[best_global_edp_idx]
        best_global_assignment = global_assignments[best_global_edp_idx]

        all_ga_points = sorted(zip(pf_energies_norm, pf_latencies_norm))
        ga_pareto_curve: list[tuple[float, float]] = []
        current_min_latency = float("inf")
        for e, l in all_ga_points:
            if l < current_min_latency:
                ga_pareto_curve.append((e, l))
                current_min_latency = l

        global_points = sorted(zip(global_energies_norm, global_latencies_norm))
        global_pareto_curve: list[tuple[float, float]] = []
        current_min_latency = float("inf")
        for e, l in global_points:
            if l < current_min_latency:
                global_pareto_curve.append((e, l))
                current_min_latency = l

        min_energy_bound = min(ga_pareto_curve[0][0], global_pareto_curve[0][0]) if ga_pareto_curve and global_pareto_curve else 0.0

        ga_auc = self._calculate_auc(ga_pareto_curve, min_energy_bound)
        global_auc = self._calculate_auc(global_pareto_curve, min_energy_bound)

        target_latency = 1.5
        energy_at_15x_ga = self._get_value_at_target(target_latency, ga_pareto_curve, x_axis="latency")
        energy_at_15x_global = self._get_value_at_target(target_latency, global_pareto_curve, x_axis="latency")

        energy_at_15x_improvement = None
        if energy_at_15x_ga and energy_at_15x_global:
            energy_at_15x_improvement = ((energy_at_15x_global - energy_at_15x_ga) / energy_at_15x_global) * 100

        avg_latency_reduction = None
        if global_auc > 0 and ga_auc > 0:
            energy_range = 1.0 - min_energy_bound
            if energy_range > 0:
                avg_latency_reduction = (global_auc - ga_auc) / energy_range

        return {
            "pf_energies": pf_energies_norm,
            "pf_latencies": pf_latencies_norm,
            "global_energies": global_energies_norm,
            "global_latencies": global_latencies_norm,
            "base_energy_norm": base_energy_norm,
            "base_latency_norm": base_latency_norm,
            "best_ga_edp_idx": best_ga_edp_idx,
            "best_ga_edp": best_ga_edp,
            "best_global_edp_idx": best_global_edp_idx,
            "best_global_edp": best_global_edp,
            "best_global_assignment": best_global_assignment,
            "baseline_edp": baseline_edp,
            "ga_pareto_curve": ga_pareto_curve,
            "global_points": global_points,
            "ga_auc": ga_auc,
            "global_auc": global_auc,
            "energy_at_15x_ga": energy_at_15x_ga,
            "energy_at_15x_global": energy_at_15x_global,
            "energy_at_15x_improvement": energy_at_15x_improvement,
            "avg_latency_reduction": avg_latency_reduction,
        }

    def plot_comparison_figure(self, plot_data: dict[str, Any], fig_path: str):
        """Render comparison figure from precomputed data only."""
        energy_at_15x_ga = plot_data["energy_at_15x_ga"]
        energy_at_15x_ga_str = f"{energy_at_15x_ga:.2f}" if energy_at_15x_ga else "N/A"

        pf_energies = plot_data["pf_energies"]
        pf_latencies = plot_data["pf_latencies"]
        ga_pareto_curve = plot_data["ga_pareto_curve"]
        best_ga_edp_idx = plot_data["best_ga_edp_idx"]
        best_ga_edp = plot_data["best_ga_edp"]
        global_energies = plot_data["global_energies"]
        global_latencies = plot_data["global_latencies"]
        best_global_edp_idx = plot_data["best_global_edp_idx"]
        best_global_edp = plot_data["best_global_edp"]
        exhaustive_baseline = plot_data["exhaustive_baseline"]
        base_energy = plot_data["base_energy"]
        base_latency = plot_data["base_latency"]
        base_energy_norm = plot_data["base_energy_norm"]
        base_latency_norm = plot_data["base_latency_norm"]
        ga_auc = plot_data.get("ga_auc", 0.0)
        global_auc = plot_data.get("global_auc", 0.0)

        plt.figure(figsize=(5, 4))

        best_ga_energy_norm = pf_energies[best_ga_edp_idx]
        best_ga_latency_norm = pf_latencies[best_ga_edp_idx]
        best_baseline_energy_norm = global_energies[best_global_edp_idx]
        best_baseline_latency_norm = global_latencies[best_global_edp_idx]

        plt.scatter(pf_energies, pf_latencies, c="red", label=f"Co-optimized GA (E@1.5xL={energy_at_15x_ga_str}, AUC={ga_auc:.2f})", alpha=0.4, s=20, zorder=3)
        plt.scatter(
            [best_ga_energy_norm],
            [best_ga_latency_norm],
            c="gold",
            marker="*",
            s=200,
            label=f"Best GA (EDP={best_ga_edp:.2f}, E={best_ga_energy_norm:.2f}, L={best_ga_latency_norm:.2f})",
            zorder=6,
            edgecolors='black',
        )

        if ga_pareto_curve:
            ga_px, ga_py = zip(*ga_pareto_curve)
            plt.plot(ga_px, ga_py, 'r--', alpha=0.3, label='GA Pareto Boundary')

        energy_at_15x_global = plot_data["energy_at_15x_global"]
        energy_at_15x_global_str = f"{energy_at_15x_global:.2f}" if energy_at_15x_global else "N/A"
        baseline_label = f"Post-scheduling per-core Pareto (n={{n_pareto}}, E@1.5xL={energy_at_15x_global_str}, AUC={global_auc:.2f})"

        baseline_points = sorted(zip(global_energies, global_latencies))
        baseline_pareto_curve: list[tuple[float, float]] = []
        current_min_latency = float("inf")
        for energy_val, latency_val in baseline_points:
            if latency_val < current_min_latency:
                baseline_pareto_curve.append((energy_val, latency_val))
                current_min_latency = latency_val

        if baseline_pareto_curve:
            baseline_px, baseline_py = zip(*baseline_pareto_curve)
            plt.scatter(
                baseline_px,
                baseline_py,
                c='green',
                marker='o',
                s=45,
                alpha=0.7,
                label=baseline_label.replace("{n_pareto}", str(len(baseline_pareto_curve))),
                zorder=1,
            )
            plt.plot(baseline_px, baseline_py, color='green', linestyle='-', alpha=0.35, zorder=1)

        plt.scatter(
            [best_baseline_energy_norm],
            [best_baseline_latency_norm],
            c='limegreen',
            marker='s',
            s=90,
            edgecolors='black',
            label=f"Best Post-scheduling Pareto (EDP={best_global_edp:.2f}, E={best_baseline_energy_norm:.2f}, L={best_baseline_latency_norm:.2f})",
            zorder=2,
        )

        if base_energy is not None and base_latency is not None:
            plt.scatter(
                [base_energy_norm],
                [base_latency_norm],
                c="black",
                marker="x",
                s=120,
                label=f"Baseline (L0, E={base_energy_norm:.2f}, L={base_latency_norm:.2f})",
                zorder=5,
            )

        plt.xlabel("Normalized Energy")
        plt.ylabel("Normalized Latency")
        plt.title("DVFS Optimization Comparison (Normalized)")
        # plt.ylim(bottom=0.5, top=2)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Comparison figure saved to {fig_path}")

    def export_core_dvfs_timeline(self, scme, output_file: str, title: str = "Core DVFS Timeline"):
        """Export per-core DVFS state over time.

        X-axis is time, Y-axis is DVFS state, and each core is color-coded.
        """
        workload = scme.workload

        core_segments: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for node in workload.node_list:
            core = node.chosen_core_allocation
            start_time = node.get_start()
            end_time = node.get_end()
            dvfs_level = node.get_dvfs_level()

            if core is None or start_time is None or end_time is None:
                continue
            if end_time <= start_time:
                continue

            core_segments[int(core)].append((int(start_time), int(end_time), int(dvfs_level)))

        if not core_segments:
            logger.warning("No DVFS segments found, skipping DVFS timeline export.")
            return

        cores = sorted(core_segments.keys())
        cmap = plt.get_cmap("tab20", max(len(cores), 1))
        core_to_color = {core: cmap(idx) for idx, core in enumerate(cores)}

        plt.figure(figsize=(10, 4.5))

        for core in cores:
            segments = sorted(core_segments[core], key=lambda seg: seg[0])
            color = core_to_color[core]
            for start_time, end_time, dvfs_level in segments:
                plt.plot(
                    [start_time, end_time],
                    [dvfs_level, dvfs_level],
                    color=color,
                    linewidth=2,
                    alpha=0.9,
                )

        # Legend: one entry per core
        for core in cores:
            plt.plot([], [], color=core_to_color[core], linewidth=3, label=f"Core {core}")

        all_starts = [segment[0] for segments in core_segments.values() for segment in segments]
        all_ends = [segment[1] for segments in core_segments.values() for segment in segments]
        all_levels = [segment[2] for segments in core_segments.values() for segment in segments]
        min_time = min(all_starts) if all_starts else 0
        max_time = max(all_ends) if all_ends else 1
        min_level = min(all_levels) if all_levels else 0
        max_level = max(all_levels) if all_levels else 1

        plt.xlim(min_time, max_time)
        plt.ylim(min_level - 0.2, max_level + 0.2)
        plt.xlabel("Time")
        plt.ylabel("DVFS State")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="best", ncol=2, fontsize=8)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=250)
        plt.close()
        logger.info("Core DVFS timeline saved to %s", output_file)

    def generate_baseline_core_dvfs_assignments(self, active_cores: list[int]) -> tuple[list[tuple[int, ...]], bool]:
        """Generate per-core DVFS assignments.

        Returns:
            assignments, exhaustive_flag
        """
        import random

        levels = sorted(self.dvfs_level_choices)
        nb_cores = len(active_cores)
        if nb_cores == 0:
            return [tuple()], True

        total_space = len(levels) ** nb_cores
        exhaustive = total_space <= self.max_baseline_combinations
        if exhaustive:
            return list(product(levels, repeat=nb_cores)), True

        sample_budget = min(self.baseline_combo_sample_budget, total_space)
        rng = random.Random(self.baseline_combo_seed)
        sampled: set[tuple[int, ...]] = set()

        baseline_level = 0 if 0 in levels else levels[0]
        sampled.add(tuple([baseline_level] * nb_cores))

        # Include all-uniform anchors
        for level in levels:
            sampled.add(tuple([level] * nb_cores))

        # Include one-core-at-a-time sweeps around baseline level
        for core_idx in range(nb_cores):
            for level in levels:
                assign = [baseline_level] * nb_cores
                assign[core_idx] = level
                sampled.add(tuple(assign))

        # Fill the remaining budget with random combinations
        while len(sampled) < sample_budget:
            sampled.add(tuple(rng.choice(levels) for _ in range(nb_cores)))

        return sorted(sampled), False

    def get_spread_core_allocations(self, *, log_distribution: bool = True) -> list[int]:
        """Build a fixed core-allocation vector by spreading assignments across valid cores."""
        core_allocs: list[int] = []
        core_load: dict[int, int] = defaultdict(int)
        for node in self.flexible_nodes:
            candidate_cores = node.core_allocation if isinstance(node.core_allocation, list) else [node.core_allocation]
            candidate_cores = [core_id for core_id in candidate_cores if core_id is not None]
            if not candidate_cores:
                raise ValueError(f"No valid core allocation candidates for node {node.name}.")

            chosen_core = min(candidate_cores, key=lambda core_id: (core_load[int(core_id)], int(core_id)))
            chosen_core = int(chosen_core)
            core_allocs.append(chosen_core)
            core_load[chosen_core] += 1
        if log_distribution:
            logger.info("Spread core-allocation seed uses core load distribution: %s", dict(sorted(core_load.items())))
        return core_allocs

    def plot_exhaustive_baseline_only(
        self,
        global_energies: list[float],
        global_latencies: list[float],
        best_global_edp_idx: int,
        base_energy_norm: float,
        base_latency_norm: float,
        fig_path: str,
    ):
        """Plot only exhaustive baseline search points (no GA points)."""
        plt.figure(figsize=(5, 4))
        plt.scatter(
            global_energies,
            global_latencies,
            c='green',
            marker='o',
            s=35,
            alpha=0.6,
            label=f"Exhaustive baseline points (n={len(global_energies)})",
            zorder=1,
        )
        plt.scatter(
            [global_energies[best_global_edp_idx]],
            [global_latencies[best_global_edp_idx]],
            c='limegreen',
            marker='s',
            s=100,
            edgecolors='black',
            label="Best exhaustive baseline (EDP min)",
            zorder=2,
        )
        plt.scatter(
            [base_energy_norm],
            [base_latency_norm],
            c="black",
            marker="x",
            s=120,
            label="Baseline (L0)",
            zorder=3,
        )
        plt.xlabel("Normalized Energy")
        plt.ylabel("Normalized Latency")
        plt.title("Exhaustive Baseline Sweep (Normalized)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.savefig(fig_path)
        plt.close()
        logger.info("Exhaustive baseline-only figure saved to %s", fig_path)

    def extract_pareto_assignments(
        self,
        energies: list[float],
        latencies: list[float],
        assignments: list[tuple[int, ...]],
    ) -> list[tuple[int, ...]]:
        """Return non-dominated per-core assignments minimizing (energy, latency)."""
        points = sorted(zip(energies, latencies, assignments), key=lambda x: (x[0], x[1]))
        pareto_assignments: list[tuple[int, ...]] = []
        best_latency = float("inf")
        for _, latency, assignment in points:
            if latency < best_latency:
                pareto_assignments.append(assignment)
                best_latency = latency
        return pareto_assignments

    def _save_scme_to_json(self, scme, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            from stream.visualization.perfetto import convert_scme_to_perfetto_json
            cost_lut = getattr(self, "cost_lut", None)
            if cost_lut is not None:
                convert_scme_to_perfetto_json(scme, cost_lut, path)
            else:
                raise ValueError("cost_lut is None")
        except Exception as e:
            import json
            import logging
            logging.getLogger(__name__).warning("Could not save perfetto JSON: " + str(e))
            data = {"energy": getattr(scme, "energy", -1), "latency": getattr(scme, "latency", -1)}
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
