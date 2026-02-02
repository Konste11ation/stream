import array
import random
import multiprocessing

from deap import algorithms, base, creator, tools

from stream.opt.allocation.genetic_algorithm.statistics_evaluator import StatisticsEvaluator


# Global variable to hold the evaluator in worker processes
global_fitness_evaluator = None

def init_worker(weights, evaluator=None):
    """
    Initialize the worker process by creating the Fitness and Individual classes in the worker's global scope.
    Also initializes the global fitness evaluator to avoid repeated pickling.
    """
    global global_fitness_evaluator
    if evaluator is not None:
        global_fitness_evaluator = evaluator

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        import array
        creator.create("Individual", array.array, typecode="i", fitness=creator.FitnessMulti)

def evaluate_wrapper(individual):
    """Wrapper to call the global evaluator's get_fitness method."""
    return global_fitness_evaluator.get_fitness(individual)


class GeneticAlgorithm:
    def __init__(
        self,
        fitness_evaluator,
        individual_length,
        valid_allocations,
        num_generations=250,
        num_individuals=64,
        pop=None,
        num_processes=4,
        prob_crossover=0.7,
        prob_mutation=0.2,
    ) -> None:
        if pop is None:
            pop = []
        self.num_generations = num_generations  # number of generations
        self.num_individuals = num_individuals  # number of individuals in initial generation
        self.para_mu = int(num_individuals / 2)  # number of indiviuals taken from previous generation
        self.para_lambda = num_individuals  # number of indiviuals in generation
        self.prob_crossover = prob_crossover  # probablility to perform corssover
        self.prob_mutation = prob_mutation  # probablility to perform mutation
        self.valid_allocations = valid_allocations
        self.num_processes = num_processes

        self.individual_length = individual_length

        self.fitness_evaluator = fitness_evaluator  # class to evaluate fitness of each indiviual
        # class to track statistics of certain generations
        self.statistics_evaluator = StatisticsEvaluator(self.fitness_evaluator)

        # define target of fitness function
        creator.create("FitnessMulti", base.Fitness, weights=self.fitness_evaluator.weights)
        # define individual in population
        creator.create("Individual", array.array, typecode="i", fitness=creator.FitnessMulti)  # type: ignore

        self.toolbox = base.Toolbox()  # initialize DEAP toolbox
        self.hof = tools.ParetoFront()  # initialize Hall-of-Fame as Pareto Front

        def get_random_individual():
            """Returns a random individual by randomly choosing from the valid allocations of each node"""
            return [random.choice(choices) for choices in valid_allocations]

        # attribute generator
        self.toolbox.register(
            "attr_bool", get_random_individual
        )  # single attribute of indiviuals can encode core allocation for HW

        # structure initializers
        self.toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,  # type: ignore
            self.toolbox.attr_bool,  # type: ignore
        )  # indivual has #nodes in graph attributes
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,  # type: ignore
        )  # define polulation based on indiviudal

        # REGISTER THE EVALUATE FUNCTION
        # If using multiprocessing, use the wrapper. Otherwise use the method.
        if self.num_processes > 1:
            self.toolbox.register("evaluate", evaluate_wrapper)
        else:
             # If single process, just use the bound method (or we could set the global and use wrapper too)
            global global_fitness_evaluator
            global_fitness_evaluator = self.fitness_evaluator
            self.toolbox.register("evaluate", evaluate_wrapper)

        # Register map with multiprocessing pool if num_processes > 1
        if self.num_processes > 1:
            self.pool = multiprocessing.Pool(
                processes=self.num_processes,
                initializer=init_worker,
                initargs=(self.fitness_evaluator.weights, self.fitness_evaluator)
            )
            self.toolbox.register("map", self.pool.map)

        # Always use cxTwoPoint. 
        # cxOrdered is for permutations (TSP) and moves values to different indices, 
        # which breaks node-specific core validity constraints.
        self.toolbox.register("mate", tools.cxTwoPoint)

        # link user defined mutation function to toolbox
        self.toolbox.register("mutate", self.mutate)
        # use non-dominated sorting genetic algorithm for multi-objective optimization
        self.toolbox.register("select", tools.selNSGA2)

        # populate random initial generation
        self.pop = self.toolbox.population(n=self.num_individuals)  # type: ignore

        # replace sub part of initial generation with user provided individuals
        for indv_index, seed_ind in enumerate(pop):
            if indv_index >= self.num_individuals:
                break
            for i in range(min(len(seed_ind), self.individual_length)):
                self.pop[indv_index][i] = seed_ind[i]

            # don't bias initial population too much
            if indv_index >= self.num_individuals / 2:
                break

    def run(self):
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
        # stats.register("saved", self.save_population)

        try:
            algorithms.eaMuPlusLambda(
                self.pop,
                self.toolbox,
                mu=self.para_mu,
                lambda_=self.para_lambda,
                cxpb=self.prob_crossover,
                mutpb=self.prob_mutation,
                ngen=self.num_generations,
                stats=stats,
                halloffame=self.hof,
            )
        finally:
            if self.num_processes > 1 and hasattr(self, "pool"):
                self.pool.close()
                self.pool.join()
                
        return self.pop, self.hof

    def mutate(self, individual):
        prob_mutation = 1 / len(individual)

        # change one of the position's core allocation
        change_percentage = 0.75
        if random.random() < change_percentage:
            for position in range(len(list(individual))):
                if random.random() < prob_mutation:
                    current_core_allocation = individual[position]
                    valid_allocs = self.valid_allocations[position]
                    # If there's only 1 valid allocation, we can't mutate to a different one
                    if len(valid_allocs) > 1:
                        valid_new_core_allocations = sorted(
                            set(valid_allocs) - set([current_core_allocation])
                        )
                        individual[position] = random.choice(valid_new_core_allocations)
        
        # swap the core allocation of two randomly chosen positions
        else:
            # We must Ensure that the swap results in a valid allocation for both positions
            # We try a fixed number of times to find a valid swap pair
            for _ in range(10):  # Try 10 times to find a valid swap
                pos1, pos2 = random.sample(range(len(individual)), 2)
                val1 = individual[pos1]
                val2 = individual[pos2]
                
                # Check if val2 is valid for pos1 AND val1 is valid for pos2
                if val2 in self.valid_allocations[pos1] and val1 in self.valid_allocations[pos2]:
                    individual[pos1] = val2
                    individual[pos2] = val1
                    break

        return (individual,)

    def save_population(self, x):
        if self.statistics_evaluator.current_generation % self.statistics_evaluator.evaluation_periode == 0:
            self.statistics_evaluator.append_generation(list(self.pop))
            self.statistics_evaluator.current_generation += 1
            return True
        else:
            self.statistics_evaluator.current_generation += 1
            return False
