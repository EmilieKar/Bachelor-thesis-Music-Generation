# Libraries
import random as rnd
import math
import timeit
import pandas as pd
import numpy as np
from EvolutionaryMusic.sims_tree import SimsTree as MusicRep


class DataGenerator:
    def __init__(
        self,
        fitness,
        parent_selection,
        survivor_selection,
        starting_population=None,
        number_of_generations=30,
        mutation_probability=0.3,
        population_size=10,
        early_stopping=None,
    ):
        self.fitness = fitness
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection
        self.number_of_generations = number_of_generations
        self.mutation_probability = mutation_probability
        self.population = starting_population

        if not self.population:
            self.__generate_new_population(population_size)
            self.population_size = population_size
        else:
            self.population_size = len(self.population)
        self.number_of_survivors = max(1, int(0.01 * self.population_size))

    def run(
        self,
    ):
        # For testing purposes
        print("Starting fitness values: ")
        fitness_values = self.get_fitness_of_population()
        print(fitness_values)

        self.generator_data = pd.DataFrame()
        self.fitness_data = pd.DataFrame()
        self.cross_data = pd.DataFrame()

        for generation in range(self.number_of_generations):
            # fitness
            start_time = timeit.default_timer()
            self.__fitness()
            fitness1_time = timeit.default_timer()
            fitness_scores = np.array([individual.fitness_score for individual in self.population])
            numpy_len = np.array([len(individual) for individual in self.population])
            avg_length = np.sum(numpy_len) / self.population_size
            numpy_nodes = np.array([individual.number_of_nodes() for individual in self.population])
            avg_nmb_nodes = np.sum(numpy_nodes) / self.population_size
            fitness2_time = timeit.default_timer()

            # survivor selection
            survivor_generator = self.survivor_selection(self.population)
            survivors = [next(survivor_generator) for _ in range(self.number_of_survivors)]

            # parent generator
            parent_generator = self.parent_selection(self.population)
            generator_time = timeit.default_timer()

            # crossover
            (
                self.population,
                sum_deepcopy_time,
                sum_cross_time,
                deepcopy_time,
                cross_time,
                parent1_len,
                parent1_nodes,
                parent2_len,
                parent2_nodes,
            ) = self.__crossover(parent_generator)
            crossover_time = timeit.default_timer()

            # mutate
            self.__mutate()
            mutate_time = timeit.default_timer()

            # insert survivors
            for i in range(len(survivors)):
                self.population[i] = survivors[i]

            new_row = {
                "fitness": fitness1_time - start_time,
                "generators": generator_time - fitness2_time,
                # "crossover": crossover_time - generator_time,
                "deeptime": sum_deepcopy_time,
                "crosstime": sum_cross_time,
                "mutation": mutate_time - crossover_time,
                "avg length": avg_length,
                "avg number of nodes": avg_nmb_nodes,
            }

            new_panda = pd.DataFrame(
                data=[deepcopy_time, parent1_len, parent1_nodes, parent2_len, parent2_nodes]
            ).T

            self.cross_data = self.cross_data.append(new_panda)
            self.generator_data = self.generator_data.append(new_row, ignore_index=True)
            self.fitness_data = self.fitness_data.append(
                [np.sort(fitness_scores)], ignore_index=True
            )

        print("Final fitness values: ")
        fitness_values = self.get_fitness_of_population()
        print(fitness_values)

    #                   Generator functions
    # ---------------------------------------------------------------

    def __fitness(self):
        for individual in self.population:
            self.fitness(individual)

    def __crossover(self, parent_generator):
        population = []

        deepcopy_time = np.array([])
        parent1_len = np.array([])
        parent2_len = np.array([])
        parent1_nodes = np.array([])
        parent2_nodes = np.array([])
        cross_time = np.array([])
        for i in range(math.ceil(len(self.population) / 2)):
            parent1 = next(parent_generator)
            parent2 = next(parent_generator)
            children, time1, time2 = parent1.crossover(parent2)
            population.append(children[0])
            population.append(children[1])
            deepcopy_time = np.append(deepcopy_time, time1)
            parent1_len = np.append(parent1_len, len(parent1))
            parent2_len = np.append(parent2_len, len(parent2))
            parent1_nodes = np.append(parent1_nodes, parent1.number_of_nodes())
            parent2_nodes = np.append(parent2_nodes, parent2.number_of_nodes())
            cross_time = np.append(cross_time, time2)

        if len(population) < self.population_size:
            population.append(parent1.crossover(parent2)[0])

        return (
            population,
            np.sum(deepcopy_time),
            np.sum(cross_time),
            deepcopy_time,
            cross_time,
            parent1_len,
            parent1_nodes,
            parent2_len,
            parent2_nodes,
        )

    def __mutate(self):
        for individual in self.population:
            if rnd.random() < self.mutation_probability:
                individual.mutate()

    def __generate_new_population(self, size):
        self.population = [MusicRep() for _ in range(size)]

    def get_fitness_of_population(self):
        for music in self.population:
            self.fitness(music)
        return str([round(individual.fitness_score, 4) for individual in self.population])

    def get_best_individual(self) -> MusicRep:
        # this will be the final piece of music
        if len(self.population) == 0:
            return None

        return sorted(self.population, key=lambda individual: individual.fitness_score)[-1]
