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

logger = logging.getLogger(__name__)


global_baseline_fitness_evaluator = None
global_baseline_core_allocs = None
global_baseline_active_cores = None
global_baseline_level = 0
global_baseline_dyn_energy_lut = None
global_baseline_fixed_node_schedule = None

def init_baseline_worker(
    evaluator,
    core_allocs,
    active_cores,
    baseline_level,
    dyn_energy_lut,
    fixed_node_schedule=None,
):
    """Initialize worker process context for per-core baseline evaluation."""
    global global_baseline_fitness_evaluator
    global global_baseline_core_allocs
    global global_baseline_active_cores
    global global_baseline_level
    global global_baseline_dyn_energy_lut
    global global_baseline_fixed_node_schedule

    global_baseline_fitness_evaluator = evaluator
    global_baseline_core_allocs = core_allocs
    global_baseline_active_cores = active_cores
    global_baseline_level = baseline_level
    global_baseline_dyn_energy_lut = dyn_energy_lut
    global_baseline_fixed_node_schedule = fixed_node_schedule

def evaluate_baseline_assignment(assignment: tuple[int, ...]) -> tuple[float, float]:
    """Evaluate one per-core DVFS assignment in a worker process."""
    evaluator = global_baseline_fitness_evaluator
    core_allocs = global_baseline_core_allocs
    active_cores = global_baseline_active_cores
    baseline_level = global_baseline_level
    dyn_energy_lut = global_baseline_dyn_energy_lut
    fixed_node_schedule = global_baseline_fixed_node_schedule

    if evaluator is None or core_allocs is None or active_cores is None or dyn_energy_lut is None:
        raise RuntimeError("Baseline worker context is not initialized.")

    core_to_level = {core: assignment[idx] for idx, core in enumerate(active_cores)}
    for node in evaluator.workload.node_list:
        core = node.chosen_core_allocation
        node_level = core_to_level.get(int(core), baseline_level) if core is not None else baseline_level
        node.set_dvfs_level(node_level)
        node.set_dyn_energy_lut(dyn_energy_lut)
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
        self.max_baseline_combinations = kwargs.get("max_baseline_combinations", 20_000)
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
            from stream_dvfs.src.dvfs_parser import DvfsParser

            dvfs_parser = DvfsParser(self.dvfs_config_path)
            self.dvfs_luts = dvfs_parser.parse_dvfs_data()
            
            # --- DVFS Mode Configuration ---
            # All nodes use DVFS with standard switching penalty
            self.dvfs_level_choices = sorted(self.dvfs_luts["freq_lut"].keys())
            
            # Initial setup of nodes with default LUTs
            # Since we removed the threshold logic, all nodes share the same DVFS configuration space
            sys_clock = self.dvfs_luts.get("system_clock_mhz", 1000)
            sta_power = self.dvfs_luts.get("base_static_power_mw", None)

            for node in self.workload.node_list:
                node.set_dvfs_level(0) # Default to max performance (level 0 usually)
                node.set_vdd_lut(self.dvfs_luts["vdd_lut"])
                node.set_freq_lut(self.dvfs_luts["freq_lut"])
                node.set_dyn_energy_lut(self.dvfs_luts["dyn_energy_dvfs_lut"])
                node.set_sta_energy_lut(self.dvfs_luts["sta_energy_lut"])
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
                logger.info(
                    "GA chromosome layout: core genes=%s, dvfs genes=%s, total genes=%s.",
                    num_flex,
                    num_dvfs,
                    num_flex + num_dvfs,
                )

                # Preferred experiment flow:
                # 1) run per-core baseline sweep first (anchor mapping only for deriving DVFS templates)
                # 2) extract baseline Pareto points
                # 3) combine Pareto-guided DVFS templates with diverse core-allocation seeds
                baseline_seed_chromosomes, exhaustive_used = self.get_baseline_best_edp_seed_chromosomes(
                    target_seed_count=self.nb_individuals,
                )
                if baseline_seed_chromosomes:
                    pop_seeds.extend(baseline_seed_chromosomes)
                    logger.info(
                        "Initialized GA with %s baseline Pareto-guided seeds + diverse core allocations (%s sweep).",
                        len(baseline_seed_chromosomes),
                        "exhaustive" if exhaustive_used else "sampled",
                    )
                # Seed with global DVFS levels (all nodes at same level)
                for level in self.dvfs_level_choices:
                    core_allocs = [random.choice(self.valid_allocations[i]) for i in range(num_flex)]
                    
                    # DVFS Levels with Constraint Checking
                    dvfs_levels = []
                    # The indices for DVFS genes start after the core allocation genes
                    dvfs_gene_offset = num_flex 
                    
                    # Iterate through all DVFS nodes and check if the 'global level' is valid for them
                    for i in range(num_dvfs):
                        node_dvfs_gene_index = dvfs_gene_offset + i
                        valid_choices = self.valid_allocations[node_dvfs_gene_index]
                        
                        if level in valid_choices:
                            dvfs_levels.append(level)
                        else:
                            # Fallback: Closest valid level (or just first valid)
                            # Since restricted nodes are usually high-voltage, level 0 is safe
                            dvfs_levels.append(valid_choices[0]) 

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
                prob_crossover=self.prob_crossover,
                prob_mutation=self.prob_mutation,
                fitness_cache_size=self.fitness_cache_size,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_min_generations=self.early_stopping_min_generations,
            )
            # Run the genetic algorithm and get the results
            pop, hof = self.genetic_algorithm.run()
            logger.info("Finished Genetic Algorithm.")

            if self.do_dvfs_cooptimization:
                final_scme = self.plot_comparison(hof)
                if final_scme:
                    yield final_scme, None
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

    def is_leaf(self) -> bool:
        return True

    def plot_comparison(self, hall_of_fame):
        """Plot the comparison between the GA Pareto front and the naive DVFS results."""
        import matplotlib.pyplot as plt
        import os
        from stream.visualization.perfetto import convert_scme_to_perfetto_json
        
        os.makedirs(self.output_path, exist_ok=True)
        fig_path = os.path.join(self.output_path, "dvfs_comparison.png")

        # 1. Get GA Pareto points
        pf_energies = []
        pf_latencies = []
        # Analyze/Save ALL Pareto Individuals
        pareto_dir = os.path.join(self.output_path, "pareto_scmes")
        os.makedirs(pareto_dir, exist_ok=True)
        
        for idx, ind in enumerate(hall_of_fame):
            pf_energies.append(ind.fitness.values[0])
            pf_latencies.append(ind.fitness.values[1])
            # Only save SCMEs for unique fitness points to save space/time
            # Or just save top 5
            if idx < 5 or idx % 10 == 0: 
                 res = self.fitness_evaluator.get_fitness(ind, return_scme=True)
                 if len(res) == 3:
                     p_scme = res[2]
                     json_name = f"scme_pareto_{idx}_E{ind.fitness.values[0]:.2e}_L{ind.fitness.values[1]:.2e}.json"
                     convert_scme_to_perfetto_json(p_scme, self.cost_lut, os.path.join(pareto_dir, json_name))

        # 2. Get Naive DVFS point
        # Take the best latency mapping and set DVFS to 0 as baseline
        best_latency_ind = sorted(hall_of_fame, key=lambda x: x.fitness.values[1])[0]
        num_flex = len(self.flexible_nodes)
        core_allocs = best_latency_ind[:num_flex]

        # Evaluate base mapping (No DVFS) -> Level 0 Baseline
        self.fitness_evaluator.set_node_core_allocations(core_allocations=core_allocs)
        for node in self.workload.node_list:
            node.set_dvfs_level(0)
            node.set_dvfs_mode("Global 0") # Temporary tag for export
        
        # Explicit type checking for base_result
        base_result = self.fitness_evaluator.get_fitness(core_allocs, return_scme=True)
        if len(base_result) == 3:
            base_energy = base_result[0]
            base_latency = base_result[1]
            base_scme = base_result[2]
            # Save Baseline SCME
            convert_scme_to_perfetto_json(base_scme, self.cost_lut, os.path.join(self.output_path, "scme_baseline_level0.json"))
        else:
            raise ValueError("Fitness evaluator did not return SCME even though return_scme=True was set.")

        # 3. Per-Core DVFS Baseline Sweep (full or sampled combinations)
        baseline_data = self.compute_per_core_baseline(core_allocs)
        global_energies = baseline_data["global_energies"]
        global_latencies = baseline_data["global_latencies"]
        global_assignments = baseline_data["global_assignments"]
        active_cores = baseline_data["active_cores"]
        exhaustive_baseline = baseline_data["exhaustive_baseline"]
        base_energy = baseline_data["base_energy"]
        base_latency = baseline_data["base_latency"]

        metric_data = self.compute_comparison_metrics(
            pf_energies,
            pf_latencies,
            global_energies,
            global_latencies,
            global_assignments,
            base_energy,
            base_latency,
        )

        pf_energies = metric_data["pf_energies"]
        pf_latencies = metric_data["pf_latencies"]
        global_energies = metric_data["global_energies"]
        global_latencies = metric_data["global_latencies"]
        base_energy_norm = metric_data["base_energy_norm"]
        base_latency_norm = metric_data["base_latency_norm"]
        best_ga_edp_idx = metric_data["best_ga_edp_idx"]
        best_ga_edp = metric_data["best_ga_edp"]
        best_global_edp_idx = metric_data["best_global_edp_idx"]
        best_global_edp = metric_data["best_global_edp"]
        best_global_assignment = metric_data["best_global_assignment"]
        baseline_edp = metric_data["baseline_edp"]
        ga_pareto_curve = metric_data["ga_pareto_curve"]
        global_points = metric_data["global_points"]
        ga_auc = metric_data["ga_auc"]
        global_auc = metric_data["global_auc"]
        energy_2x_ga = metric_data["energy_2x_ga"]
        energy_2x_global = metric_data["energy_2x_global"]
        energy_2x_improvement = metric_data["energy_2x_improvement"]
        avg_latency_reduction = metric_data["avg_latency_reduction"]

        energy_2x_gap_str = ""
        if energy_2x_ga and energy_2x_global:
            energy_2x_gap_str = f" | GA is {energy_2x_improvement:.1f}% lower energy"

        logger.info("="*40)
        logger.info("DVFS Optimization Results")
        logger.info(
            "Per-core baseline points: %s (%s)",
            len(global_points),
            "exhaustive" if exhaustive_baseline else "sampled",
        )
        logger.info(f"Baseline (Level 0) EDP: {baseline_edp:.4f} (Normalized)")
        logger.info(
            "Best Per-Core Baseline EDP: %.4f (assignment=%s over cores=%s)",
            best_global_edp,
            list(best_global_assignment),
            active_cores,
        )
        logger.info(f"Best GA Co-Optimization EDP: {best_ga_edp:.4f}")
        logger.info(f"Improvement over Baseline (EDP): {(baseline_edp - best_ga_edp)/baseline_edp * 100:.2f}%")
        # logger.info("-" * 20)
        # logger.info(f"Global Scaling AUC: {global_auc:.4f}")
        # logger.info(f"GA Pareto AUC: {ga_auc:.4f}")
        # logger.info(f"Improvement (AUC): {(global_auc - ga_auc)/global_auc * 100:.2f}%")
        logger.info("-" * 20)
        logger.info(f"Energy @ 1.2x Latency (Iso-Latency):")
        logger.info(f"  Global: {energy_2x_global if energy_2x_global else 'N/A'}")
        logger.info(f"  GA    : {energy_2x_ga if energy_2x_ga else 'N/A'}{energy_2x_gap_str}")
        if avg_latency_reduction:
            logger.info(f"Avg Latency Reduction: {avg_latency_reduction:.2f} (Avg distance between curves)")
        logger.info("="*40)

        # 5. Prepare and plot comparison figure (plotting decoupled from computation)
        plot_data = {
            "pf_energies": pf_energies,
            "pf_latencies": pf_latencies,
            "ga_pareto_curve": ga_pareto_curve,
            "best_ga_edp_idx": best_ga_edp_idx,
            "best_ga_edp": best_ga_edp,
            "global_energies": global_energies,
            "global_latencies": global_latencies,
            "best_global_edp_idx": best_global_edp_idx,
            "best_global_edp": best_global_edp,
            "exhaustive_baseline": exhaustive_baseline,
            "energy_2x_ga": energy_2x_ga,
            "energy_2x_global": energy_2x_global,
            "base_energy": base_energy,
            "base_latency": base_latency,
            "base_energy_norm": base_energy_norm,
            "base_latency_norm": base_latency_norm,
        }
        self.plot_comparison_figure(plot_data, fig_path)
        if exhaustive_baseline:
            exhaustive_fig_path = os.path.join(self.output_path, "dvfs_exhaustive_baseline_only.png")
            self.plot_exhaustive_baseline_only(
                global_energies=global_energies,
                global_latencies=global_latencies,
                best_global_edp_idx=best_global_edp_idx,
                base_energy_norm=base_energy_norm,
                base_latency_norm=base_latency_norm,
                fig_path=exhaustive_fig_path,
            )

        # Attach Baseline metrics to the best SCME for external analysis
        # "hall_of_fame" contains the best individuals. We assume the last one is good, 
        # but let's pick the Best EDP individual to return as "The Result"
        best_scme_ind = hall_of_fame[best_ga_edp_idx] # The individual corresponding to best EDP
        
        # We need to re-evaluate to get the SCME object
        res = self.fitness_evaluator.get_fitness(best_scme_ind, return_scme=True)
        final_scme = res[2] if len(res) == 3 else None
        
        final_scme_to_return = None
        if final_scme:
            final_scme.baseline_energy = base_energy
            final_scme.baseline_latency = base_latency
            final_scme.best_global_energy_norm = global_energies[best_global_edp_idx]
            final_scme.best_global_latency_norm = global_latencies[best_global_edp_idx]
            final_scme.best_ga_edp = best_ga_edp
            final_scme.ga_auc = ga_auc
            final_scme.global_auc = global_auc
            final_scme.energy_2x_ga = energy_2x_ga
            final_scme.energy_2x_global = energy_2x_global
            final_scme.energy_2x_improvement = energy_2x_improvement
            final_scme.avg_latency_reduction = avg_latency_reduction
            final_scme_to_return = final_scme

            self.export_core_dvfs_timeline(
                final_scme,
                os.path.join(self.output_path, "core_dvfs_timeline_ga_best.png"),
                title="Core DVFS Timeline (Best GA EDP)",
            )
            
            # Save the SCME for the best GA point
            from stream.visualization.perfetto import convert_scme_to_perfetto_json
            import pickle
            convert_scme_to_perfetto_json(
                final_scme,
                self.cost_lut,
                os.path.join(self.output_path, "scme_ga_best_edp.json")
            )
            with open(os.path.join(self.output_path, "scme_ga_best_edp.pkl"), "wb") as f:
                pickle.dump(final_scme, f)
            
            logger.info(f"Saved Best EDP allocation SCME to scme_ga_best_edp.json/pkl (Energy: {final_scme.energy:.2e}, Latency: {final_scme.latency:.2e})")
             
        return final_scme_to_return

    def compute_per_core_baseline(self, core_allocs: list[int]) -> dict[str, Any]:
        """Compute per-core baseline sweep results independent from plotting."""
        from stream.visualization.perfetto import convert_scme_to_perfetto_json

        cache_key = tuple(int(core_id) for core_id in core_allocs)
        if cache_key in self._baseline_sweep_cache:
            logger.info("Using cached per-core baseline sweep for core allocations: %s", list(cache_key))
            return self._baseline_sweep_cache[cache_key]

        base_energy = None
        base_latency = None
        global_energies: list[float] = []
        global_latencies: list[float] = []
        global_assignments: list[tuple[int, ...]] = []

        global_dir = os.path.join(self.output_path, "global_levels_scmes")
        os.makedirs(global_dir, exist_ok=True)

        active_cores = sorted(
            core.id
            for core in self.accelerator.cores.node_list
            if core.id != self.accelerator.offchip_core_id
        )
        if not active_cores:
            active_cores = [0]

        candidate_cores = sorted(
            {
                int(core_id)
                for node in self.flexible_nodes
                for core_id in (
                    node.core_allocation if isinstance(node.core_allocation, list) else [node.chosen_core_allocation]
                )
                if core_id is not None
            }
        )
        logger.info(
            "Baseline sweep core sets: active_core_ids=%s (all hardware cores), candidate_core_ids=%s",
            active_cores,
            candidate_cores,
        )

        baseline_assignments, exhaustive_baseline = self.generate_baseline_core_dvfs_assignments(active_cores)
        logger.info(
            "Per-core DVFS baseline sweep: active_cores=%s, levels=%s, evaluated=%s (%s).",
            len(active_cores),
            len(self.dvfs_level_choices),
            len(baseline_assignments),
            "exhaustive" if exhaustive_baseline else "sampled subset",
        )

        baseline_level = 0 if 0 in self.dvfs_level_choices else min(self.dvfs_level_choices)
        total_baseline_assignments = len(baseline_assignments)
        progress_interval = max(1, total_baseline_assignments // 20)

        # ---------------------------------------------------------------------
        # Extract NOMINAL Temporal Schedule
        # ---------------------------------------------------------------------
        logger.info("Evaluating nominal baseline to extract strict temporal schedule...")
        for node in self.workload.node_list:
            core = node.chosen_core_allocation
            node.set_dvfs_level(baseline_level)
            node.set_dyn_energy_lut(self.dvfs_luts["dyn_energy_dvfs_lut"])
            node.set_dvfs_mode("PerCoreBaseline")
        
        nominal_fitness = self.fitness_evaluator.get_fitness(core_allocs, return_scme=True)
        nominal_scme = nominal_fitness[2]
        # Store a list of node IDs based on the scheduling sequence
        if hasattr(nominal_scme, 'scheduled_node_sequence') and nominal_scme.scheduled_node_sequence:
            nominal_schedule_ids = [n.id for n in nominal_scme.scheduled_node_sequence]
        else:
            logger.warning("No scheduled_node_sequence found on nominal_scme!")
            nominal_schedule_ids = []

        logger.info(f"Extracted strict nominal schedule of length {len(nominal_schedule_ids)}.")
        # ---------------------------------------------------------------------

        if exhaustive_baseline and self.num_procs > 1 and total_baseline_assignments > 1:
            worker_count = min(self.num_procs, total_baseline_assignments)
            logger.info(
                "Running exhaustive per-core baseline sweep with multiprocessing (%s workers).",
                worker_count,
            )
            chunksize = max(1, total_baseline_assignments // (worker_count * 8))
            baseline_dvfs_tuple = tuple([baseline_level] * len(active_cores))
            baseline_tuple_idx = None

            with multiprocessing.Pool(
                processes=worker_count,
                initializer=init_baseline_worker,
                initargs=(
                    self.fitness_evaluator,
                    core_allocs,
                    active_cores,
                    baseline_level,
                    self.dvfs_luts["dyn_energy_dvfs_lut"],
                    None, # was nominal_schedule_ids
                ),
            ) as pool:
                for assignment_idx, (assignment, result) in enumerate(
                    zip(
                        baseline_assignments,
                        pool.imap(evaluate_baseline_assignment, baseline_assignments, chunksize=chunksize),
                    ),
                    start=1,
                ):
                    energy, latency = result
                    global_energies.append(energy)
                    global_latencies.append(latency)
                    global_assignments.append(assignment)

                    if assignment == baseline_dvfs_tuple:
                        base_energy, base_latency = energy, latency
                        baseline_tuple_idx = assignment_idx

                    if (
                        assignment_idx == 1
                        or assignment_idx == total_baseline_assignments
                        or assignment_idx % progress_interval == 0
                    ):
                        logger.info(
                            "Per-core baseline progress: %s/%s (%.1f%%)",
                            assignment_idx,
                            total_baseline_assignments,
                            100.0 * assignment_idx / total_baseline_assignments,
                        )

            if baseline_tuple_idx is None:
                logger.warning("Did not find baseline tuple in evaluated assignments; using first point as normalization baseline.")
        else:
            for assignment_idx, assignment in enumerate(baseline_assignments, start=1):
                core_to_level = {core: assignment[idx] for idx, core in enumerate(active_cores)}
                for node in self.workload.node_list:
                    core = node.chosen_core_allocation
                    node_level = core_to_level.get(int(core), baseline_level) if core is not None else baseline_level
                    node.set_dvfs_level(node_level)
                    node.set_dyn_energy_lut(self.dvfs_luts["dyn_energy_dvfs_lut"])
                    node.set_dvfs_mode("PerCoreBaseline")

                g_res = self.fitness_evaluator.get_fitness(core_allocs, fixed_node_schedule=None)
                energy_val = g_res[0]
                latency_val = g_res[1]
                if energy_val is None or latency_val is None:
                    raise ValueError("Per-core baseline evaluation returned None for energy or latency.")

                global_energies.append(float(energy_val))
                global_latencies.append(float(latency_val))
                global_assignments.append(assignment)

                if all(level == baseline_level for level in assignment):
                    base_energy, base_latency = float(energy_val), float(latency_val)

                if (
                    assignment_idx == 1
                    or assignment_idx == total_baseline_assignments
                    or assignment_idx % progress_interval == 0
                ):
                    logger.info(
                        "Per-core baseline progress: %s/%s (%.1f%%)",
                        assignment_idx,
                        total_baseline_assignments,
                        100.0 * assignment_idx / total_baseline_assignments,
                    )

        if base_energy is None or base_latency is None:
            base_energy = global_energies[0]
            base_latency = global_latencies[0]

        global_edps_abs = [e * l for e, l in zip(global_energies, global_latencies)]
        best_global_idx_abs = global_edps_abs.index(min(global_edps_abs))
        best_global_assignment_abs = global_assignments[best_global_idx_abs]
        best_assignment_str = "_".join(map(str, best_global_assignment_abs))
        best_core_to_level_abs = {
            core: best_global_assignment_abs[idx] for idx, core in enumerate(active_cores)
        }
        for node in self.workload.node_list:
            core = node.chosen_core_allocation
            node_level = best_core_to_level_abs.get(int(core), baseline_level) if core is not None else baseline_level
            node.set_dvfs_level(node_level)
            node.set_dyn_energy_lut(self.dvfs_luts["dyn_energy_dvfs_lut"])
            node.set_dvfs_mode("PerCoreBaselineBest")

        best_g_res = self.fitness_evaluator.get_fitness(core_allocs, return_scme=True)
        if len(best_g_res) == 3:
            import pickle
            convert_scme_to_perfetto_json(
                best_g_res[2],
                self.cost_lut,
                os.path.join(global_dir, f"scme_best_per_core_assignment_{best_assignment_str}.json"),
            )
            with open(os.path.join(self.output_path, "scme_baseline_best_edp.pkl"), "wb") as f:
                pickle.dump(best_g_res[2], f)

        baseline_result = {
            "global_energies": global_energies,
            "global_latencies": global_latencies,
            "global_assignments": global_assignments,
            "active_cores": active_cores,
            "exhaustive_baseline": exhaustive_baseline,
            "base_energy": base_energy,
            "base_latency": base_latency,
        }
        self._baseline_sweep_cache[cache_key] = baseline_result
        self._save_baseline_sweep_cache()
        return baseline_result

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
        min_energy_bound = global_points[0][0] if global_points else 0.0

        ga_auc = self._calculate_auc(ga_pareto_curve, min_energy_bound)
        global_auc = self._calculate_auc(global_points)
        ga_auc, global_auc = 0.0, 0.0

        target_latency = 1.2
        energy_2x_ga = self._get_value_at_target(target_latency, ga_pareto_curve, x_axis="latency")
        energy_2x_global = self._get_value_at_target(target_latency, global_points, x_axis="latency")

        energy_2x_improvement = 0.0
        if energy_2x_ga and energy_2x_global:
            energy_2x_improvement = ((energy_2x_global - energy_2x_ga) / energy_2x_global) * 100

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
            "energy_2x_ga": energy_2x_ga,
            "energy_2x_global": energy_2x_global,
            "energy_2x_improvement": energy_2x_improvement,
            "avg_latency_reduction": avg_latency_reduction,
        }

    def plot_comparison_figure(self, plot_data: dict[str, Any], fig_path: str):
        """Render comparison figure from precomputed data only."""
        energy_2x_ga = plot_data["energy_2x_ga"]
        energy_2x_global = plot_data["energy_2x_global"]
        energy_2x_ga_str = f"{energy_2x_ga:.2f}" if energy_2x_ga else "N/A"
        energy_2x_global_str = f"{energy_2x_global:.2f}" if energy_2x_global else "N/A"

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

        plt.figure(figsize=(5, 4))

        best_ga_energy_norm = pf_energies[best_ga_edp_idx]
        best_ga_latency_norm = pf_latencies[best_ga_edp_idx]
        best_baseline_energy_norm = global_energies[best_global_edp_idx]
        best_baseline_latency_norm = global_latencies[best_global_edp_idx]

        plt.scatter(pf_energies, pf_latencies, c="red", label=f"Co-optimized GA (E@1.2xL={energy_2x_ga_str})", alpha=0.4, s=20, zorder=3)
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

        baseline_label = (
            f"Post-scheduling per-core Pareto (n={{n_pareto}}, E@1.2xL={energy_2x_global_str})"
        )

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
                label=baseline_label.format(n_pareto=len(baseline_pareto_curve)),
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

    def plot_bandwidth_comparison(self, scme_base, scme_comp):
        """Plots off-chip bandwidth usage of Baseline vs Comparison (e.g., Level 1)."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        def get_bw_profile(scme):
            # Aggregate transfer events
            events = []
            acc = scme.accelerator
            # Go through all links
            for links in acc.communication_manager.all_pair_links.values():
                for link_pair in links:
                    for link in link_pair:
                         # Filter for off-chip links (DRAM) if possible, or just all links
                         # Assuming 'DRAM' or similar in name for offchip, or just plot all for congestion proxy
                         for event in link.events:
                             if (event.end - event.start) > 0:
                                 events.append((event.start, event.end, link.bandwidth)) # Storing util or BW?
            
            # Simplified: Plot 'Active Transfers' count or similar, since raw BW is hard to sum without time-binning
            # Better: Use CostModelEvaluation properties if available.
            # Fallback: Time-binning
            max_time = int(scme.latency)
            bins = np.zeros(max_time // 100 + 1) # Downsample 
            # This is complex to do efficiently in Python loop for many events. 
            # Alternative: Just return the SCME latency for annotation
            return max_time
            
        # Since full BW plotting is complex, let's plot a simplified "Activity" view or just annotated latencies
        # Real "Slower is Faster" proof requires looking at Communication contention.
        # We will save the detailed SCMEs (already done above) and user can analyze in Perfetto.
        # Here we just output a text summary or simple bar chart of components.
        
        # Breakdown Latency
        labels = ['Level 0', 'Level 1']
        latencies = [scme_base.latency, scme_comp.latency]
        
        ax.bar(labels, latencies, color=['gray', 'green'])
        ax.set_ylabel("Total Latency (Cycles)")
        ax.set_title("Latency Comparison: Max Freq (L0) vs Slower (L1)\n(Slower Logic -> Less Contention -> Faster System)")
        
        # Annotate
        for i, v in enumerate(latencies):
            ax.text(i, v, str(int(v)), ha='center', va='bottom')
            
        bw_fig_path = os.path.join(self.output_path, "bandwidth_latency_comparison.png")
        plt.savefig(bw_fig_path)
        plt.close()

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

    def get_spread_core_allocation_pool(self, pool_size: int) -> list[list[int]]:
        """Generate diverse core-allocation chromosomes for seeding.

        Deterministic round-robin across each node's valid cores to avoid fixing one core-allocation.
        """
        pool_size = max(1, int(pool_size))
        if not self.flexible_nodes:
            return [[]]

        pool: list[list[int]] = []
        for seed_idx in range(pool_size):
            alloc: list[int] = []
            for node_idx, node in enumerate(self.flexible_nodes):
                candidate_cores = node.core_allocation if isinstance(node.core_allocation, list) else [node.core_allocation]
                candidate_cores = [int(core_id) for core_id in candidate_cores if core_id is not None]
                if not candidate_cores:
                    raise ValueError(f"No valid core allocation candidates for node {node.name}.")
                core_pick = candidate_cores[(seed_idx + node_idx) % len(candidate_cores)]
                alloc.append(core_pick)
            pool.append(alloc)
        return pool

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

    def get_baseline_best_edp_seed_chromosomes(
        self,
        target_seed_count: int,
    ) -> tuple[list[list[int]], bool]:
        """Run baseline sweep first and convert Pareto points into GA seed chromosomes.

        Core-allocation genes are diversified and not fixed to the baseline anchor allocation.
        """
        core_allocs = self.get_spread_core_allocations(log_distribution=False)
        self._preferred_baseline_core_allocs = tuple(core_allocs)
        self.fitness_evaluator.set_node_core_allocations(core_allocs)
        active_cores = sorted(
            core.id
            for core in self.accelerator.cores.node_list
            if core.id != self.accelerator.offchip_core_id
        )
        if not active_cores:
            active_cores = [0]
        total_state_space = len(self.dvfs_level_choices) ** len(active_cores)

        old_max_baseline_combinations = self.max_baseline_combinations
        if self.force_exhaustive_seed_baseline:
            if total_state_space <= self.max_seed_baseline_state_space:
                self.max_baseline_combinations = max(self.max_baseline_combinations, total_state_space)
                logger.info(
                    "For GA seeding: forcing exhaustive per-core baseline sweep (state-space=%s).",
                    total_state_space,
                )
            else:
                logger.warning(
                    "For GA seeding: exhaustive baseline state-space too large (%s > %s); falling back to sampled baseline sweep.",
                    total_state_space,
                    self.max_seed_baseline_state_space,
                )

        baseline_data = self.compute_per_core_baseline(core_allocs)
        self.max_baseline_combinations = old_max_baseline_combinations

        assignments = baseline_data["global_assignments"]
        energies = baseline_data["global_energies"]
        latencies = baseline_data["global_latencies"]
        exhaustive_baseline = baseline_data["exhaustive_baseline"]
        active_cores = baseline_data["active_cores"]

        pareto_assignments = self.extract_pareto_assignments(energies, latencies, assignments)
        pareto_set = set(pareto_assignments)
        pareto_points = [
            (energy, latency, assignment)
            for energy, latency, assignment in zip(energies, latencies, assignments)
            if assignment in pareto_set
        ]
        if not pareto_points:
            return [], exhaustive_baseline

        pareto_points_sorted = sorted(
            pareto_points,
            key=lambda p: (p[0] * p[1], p[1], p[0]),
        )
        pareto_assignments_ranked = [assignment for _, _, assignment in pareto_points_sorted]

        core_seed_pool = self.get_spread_core_allocation_pool(max(target_seed_count, len(pareto_assignments_ranked)))
        node_key_to_seed_pos = {
            (node.id, node.sub_id): idx for idx, node in enumerate(self.flexible_nodes)
        }

        baseline_level = 0 if 0 in self.dvfs_level_choices else min(self.dvfs_level_choices)
        num_flex = len(self.flexible_nodes)
        seen: set[tuple[int, ...]] = set()
        seed_chromosomes: list[list[int]] = []

        for assignment in pareto_assignments_ranked:
            core_to_level = {core: assignment[idx] for idx, core in enumerate(active_cores)}

            for core_alloc_seed in core_seed_pool:
                dvfs_levels: list[int] = []
                dvfs_gene_offset = num_flex
                for i, node in enumerate(self.flexible_nodes_dvfs):
                    node_key = (node.id, node.sub_id)
                    seed_pos = node_key_to_seed_pos.get(node_key)
                    if seed_pos is not None:
                        core = core_alloc_seed[seed_pos]
                    else:
                        core = node.chosen_core_allocation
                    chosen_level = core_to_level.get(int(core), baseline_level) if core is not None else baseline_level
                    valid_choices = self.valid_allocations[dvfs_gene_offset + i]
                    if chosen_level in valid_choices:
                        dvfs_levels.append(chosen_level)
                    else:
                        dvfs_levels.append(valid_choices[0])

                chromosome = list(core_alloc_seed) + dvfs_levels
                chromosome_key = tuple(chromosome)
                if chromosome_key not in seen:
                    seed_chromosomes.append(chromosome)
                    seen.add(chromosome_key)

                if len(seed_chromosomes) >= target_seed_count:
                    break

            if len(seed_chromosomes) >= target_seed_count:
                break

        logger.info(
            "Pareto-guided seed generation: pareto_points=%s, selected_seeds=%s, target=%s.",
            len(pareto_assignments_ranked),
            len(seed_chromosomes),
            target_seed_count,
        )

        self.get_spread_core_allocations(log_distribution=True)

        return seed_chromosomes, exhaustive_baseline

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
