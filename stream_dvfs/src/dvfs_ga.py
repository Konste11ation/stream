import sys
import os
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_DIR = STREAM_DVFS_DIR.parent
import array
import random

from deap import algorithms, base, creator, tools

from stream.opt.allocation.genetic_algorithm.statistics_evaluator import (
    StatisticsEvaluator,
)
import matplotlib.pyplot as plt
import multiprocessing

class DvfsGeneticAlgorithm:
    def __init__(
        self,
        fitness_evaluator,
        individual_length,
        valid_allocations,
        num_generations=250,
        num_individuals=64,
        num_processes=4,
        pop_init=None,
    ) -> None:
        if hasattr(creator, 'FitnessMulti'):
            del creator.FitnessMulti
        if hasattr(creator, 'Individual'):
            del creator.Individual
        self.num_generations = num_generations  # number of generations
        self.num_individuals = num_individuals  # number of individuals in initial generation
        self.para_mu = int(num_individuals / 2)  # number of indiviuals taken from previous generation
        self.para_lambda = num_individuals  # number of indiviuals in generation
        self.prob_crossover = 0.3  # probablility to perform corssover
        self.prob_mutation = 0.7  # probablility to perform mutation
        self.valid_allocations = valid_allocations
        self.num_processes = num_processes

        self.individual_length = individual_length

        self.fitness_evaluator = fitness_evaluator  # class to evaluate fitness of each indiviual
        # class to track statistics of certain generations
        self.statistics_evaluator = StatisticsEvaluator(self.fitness_evaluator)

        # define target of fitness function
        creator.create("FitnessMulti", base.Fitness, weights=self.fitness_evaluator.weights)
        # define individual in population
        creator.create("Individual", array.array, typecode="i", fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()  # initialize DEAP toolbox
        self.hof = tools.ParetoFront()  # initialize Hall-of-Fame as Pareto Front

        # attribute generator
        self.toolbox.register(
            "attr_int", random.randint, valid_allocations[0], valid_allocations[1]
        )  # single attribute of indiviuals can encode core allocation for HW

        # structure initializers
        self.toolbox.register(
            "individual", 
            tools.initRepeat, 
            creator.Individual, 
            self.toolbox.attr_int, 
            n=individual_length
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )  # define polulation based on indiviudal

        # link user defined fitness function to toolbox
        self.toolbox.register("evaluate", self.fitness_evaluator.get_fitness)
        if self.individual_length > 10:
            self.toolbox.register("mate", tools.cxOrdered)  # for big graphs use cxOrdered crossover function
        elif self.individual_length >= 2:
            self.toolbox.register("mate", tools.cxTwoPoint)  # for small graphs use two point crossover function
        else:
            # For indivual length 0 or 1, crossover is not possible/doesn't make sense.
            # We register a dummy mate function that returns the individuals unchanged.
            self.toolbox.register("mate", lambda ind1, ind2: (ind1, ind2))

        self.toolbox.register("mutate", tools.mutUniformInt, low=valid_allocations[0], up=valid_allocations[1], indpb=0.1)
        # use non-dominated sorting genetic algorithm for multi-objective optimization
        self.toolbox.register("select", tools.selNSGA2)

        # Create the inital population
        self.pop_init = pop_init
        self.pop = self._create_initial_population()

    def _normalize_pop_init(self):
        """Normalize initial population input to a list of valid seed individuals."""
        if self.pop_init is None:
            return []

        if isinstance(self.pop_init, (list, tuple)) and len(self.pop_init) == self.individual_length and all(
            isinstance(value, int) for value in self.pop_init
        ):
            candidates = [list(self.pop_init)]
        elif isinstance(self.pop_init, (list, tuple)):
            candidates = [
                list(candidate)
                for candidate in self.pop_init
                if isinstance(candidate, (list, tuple))
            ]
        else:
            return []

        valid_candidates = []
        low, high = self.valid_allocations
        for candidate in candidates:
            if len(candidate) != self.individual_length:
                continue
            clipped = [min(max(int(value), low), high) for value in candidate]
            valid_candidates.append(clipped)
        return valid_candidates

    def _create_initial_population(self):
        """Create the initial population, optionally seeded with pop_init."""
        pop = self.toolbox.population(n=self.num_individuals)

        seed_individuals = self._normalize_pop_init()
        if not seed_individuals:
            return pop

        max_seed_count = min(len(seed_individuals), self.num_individuals // 2 if self.num_individuals > 1 else 1)
        for i in range(max_seed_count):
            pop[i] = creator.Individual(seed_individuals[i])

        mutation_budget = min(self.num_individuals - max_seed_count, max(1, max_seed_count))
        for i in range(mutation_budget):
            target_idx = max_seed_count + i
            if target_idx >= len(pop):
                break
            base_seed = seed_individuals[i % max_seed_count]
            pop[target_idx] = creator.Individual(base_seed)
            self.toolbox.mutate(pop[target_idx])
            del pop[target_idx].fitness.values
        
        return pop
    def _adjust_mutation_probability(self, generation):
        """Dynamically adjust mutation probability based on the current generation."""
        max_generations = self.num_generations
        return max(0.1, self.prob_mutation * (1 - generation / max_generations))
    def _adjust_crossover_probability(self, diversity):
        """Adjust crossover probability based on population diversity."""
        return max(0.2, min(0.8, diversity))

    def _compute_diversity(self):
        """Compute diversity as the average distance between individuals."""
        distances = []
        for i, ind1 in enumerate(self.pop):
            for j, ind2 in enumerate(self.pop):
                if i < j:
                    distances.append(sum(abs(x - y) for x, y in zip(ind1, ind2)))
        return sum(distances) / len(distances) if distances else 0

    def run(self):
        # Register map with multiprocessing pool if num_processes > 1
        pool = None
        if self.num_processes > 1:
            pool = multiprocessing.Pool(processes=self.num_processes)
            self.toolbox.register("map", pool.map)

        # plot statistics during evolution
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register(
            "avg (" + ", ".join(self.fitness_evaluator.metrics) + ")",
            self.statistics_evaluator.get_avg,
        )
        stats.register(
            "std (" + ", ".join(self.fitness_evaluator.metrics) + ")",
            self.statistics_evaluator.get_std,
        )
        stats.register(
            "min (" + ", ".join(self.fitness_evaluator.metrics) + ")",
            self.statistics_evaluator.get_min,
        )
        stats.register(
            "max (" + ", ".join(self.fitness_evaluator.metrics) + ")",
            self.statistics_evaluator.get_max,
        )

        for gen in range(self.num_generations):
            diversity = self._compute_diversity()
            self.toolbox.cxpb = self._adjust_crossover_probability(diversity)
            self.toolbox.mutpb = self._adjust_mutation_probability(gen)
            algorithms.eaMuPlusLambda(
                self.pop,
                self.toolbox,
                mu=self.para_mu,
                lambda_=self.para_lambda,
                cxpb=self.prob_crossover,
                mutpb=self.toolbox.mutpb,
                ngen=1,  # Run one generation at a time
                stats=stats,
                halloffame=self.hof,
            )
        
        # Close the pool correctly
        if pool:
            pool.close()
            pool.join()
            
        return self.pop, self.hof
    
    def plot_pareto_front(self,filename=None):
        plt.figure(figsize=(10, 6))
        pareto_front = self.hof
        if len(pareto_front) > 0:
            pf_energy = [ind.fitness.values[0] for ind in pareto_front]
            pf_latency = [ind.fitness.values[1] for ind in pareto_front]
            plt.scatter(pf_energy, pf_latency, 
                        c='red', s=80, edgecolors='black',
                        label='Pareto Front', zorder=3)
        plt.xlabel('Energy Consumption', fontsize=12)
        plt.ylabel('Latency', fontsize=12)
        plt.title('Pareto Front Visualization', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()