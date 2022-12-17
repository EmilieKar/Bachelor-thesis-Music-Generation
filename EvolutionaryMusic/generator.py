# Libraries
import random as rnd
import math
from EvolutionaryMusic.sims_tree import SimsTree as MusicRep
from EvolutionaryMusic.midi import Midi


class Generator:
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
        self.early_stopping = (
            [] if early_stopping is None else list(map(int, early_stopping.split(":")))
        )

        if not self.population:
            self.__generate_new_population(population_size)
            self.population_size = population_size
        else:
            self.population_size = len(self.population)
        self.number_of_survivors = max(1, int(0.01 * self.population_size))

    def run(self):
        # For testing purposes
        # print("Starting fitness values: ")
        # fitness_values = self.get_fitness_of_population()
        # print(fitness_values)

        print(
            "\nStarting generator with\n"
            f"\tfitness: {type(self.fitness).__name__}\n"
            f"\tparent_selection: {self.parent_selection.__name__}\n"
            f"\tsurvivor_selection: {self.survivor_selection.__name__}\n"
            f"\tgenerations: {self.number_of_generations}\n"
            f"\tmutation_probability: {self.mutation_probability}\n"
            f"\tpopulation_size: {self.population_size}\n"
        )

        for generation in range(self.number_of_generations):
            print(f"Generation {generation+1}/{self.number_of_generations}")

            # fitness
            self.__fitness()

            # survivor selection
            survivor_generator = self.survivor_selection(self.population)
            survivors = [next(survivor_generator) for _ in range(self.number_of_survivors)]

            # parent generator
            parent_generator = self.parent_selection(self.population)

            # crossover
            self.population = self.__crossover(parent_generator)

            # mutate
            self.__mutate()

            # insert survivors
            for i in range(len(survivors)):
                self.population[i] = survivors[i]

            # preview for early stopping
            if len(self.early_stopping) > 0 and self.early_stopping[0] == generation + 1:
                self.early_stopping.pop(0)

                midi_player = Midi()
                samples = rnd.sample(self.population, math.ceil(len(self.population) * 0.3))
                for s in samples:
                    midi_player.play(s)

                midi_player.close()
                del midi_player

                ans = input("Do you want to continue? (y/n)")
                if ans == "n":
                    break

        self.__fitness()

    #                   Generator functions
    # ---------------------------------------------------------------

    def __fitness(self):
        for individual in self.population:
            self.fitness(individual)

    def __crossover(self, parent_generator):
        population = []

        for i in range(math.ceil(len(self.population) / 2)):
            parent1 = next(parent_generator)
            parent2 = next(parent_generator)
            children, time1, time2 = parent1.crossover(parent2)
            population.append(children[0])
            population.append(children[1])

        if len(population) < self.population_size:
            population.append(parent1.crossover(parent2)[0])

        return population

    def __mutate(self):
        for individual in self.population:
            if rnd.random() < self.mutation_probability:
                individual.mutate()

    def __generate_new_population(self, size):
        self.population = [MusicRep() for _ in range(size)]

    def get_fitness_of_population(self):
        self.__fitness()
        return str([round(individual.fitness_score, 2) for individual in self.population])

    def get_best_individual(self) -> MusicRep:
        # this will be the final piece of music
        if len(self.population) == 0:
            return None

        return sorted(self.population, key=lambda individual: individual.fitness_score)[-1]

    def individuals(self):
        if len(self.population) == 0:
            return None

        return sorted(self.population, key=lambda individual: individual.fitness_score)
