import random as rnd
import numpy as np
import math


def survivor_selection(population):
    function = rnd.choice([tournament_selection, elitism])
    return function(population)


def elitism(population):
    fitness_array = sorted(population, key=lambda x: -x.fitness_score)
    for i in range(len(population)):
        yield fitness_array[i]


def parent_generator_selection(population):
    function = rnd.choice(
        [
            tournament_selection,
            roulette_selection,
            sus_selection,
            rank_selection,
        ]
    )
    return function(population)


def tournament_selection(population, tournament_portion=0.3):
    tournament_size = int(len(population) * tournament_portion)
    for i in range(len(population)):
        candidates = rnd.sample(population, tournament_size)
        yield max(candidates, key=lambda x: x.fitness_score)


def roulette_selection(population):
    fitness_array = np.array(
        [individual.fitness_score for individual in population]
    )
    cumulative_fitness = np.cumsum(fitness_array)
    for i in range(len(population)):
        spin_wheel = rnd.random() * cumulative_fitness[-1]
        parent = population[find_nearest(cumulative_fitness, spin_wheel)]
        yield parent


# Stochastic universal sampling selection
def sus_selection(population):
    fitness_array = np.array(
        [individual.fitness_score for individual in population]
    )
    cumulative_fitness = np.cumsum(fitness_array)
    for i in range(math.ceil(len(population) / 2)):
        spin_wheel = []
        spin_wheel.append(rnd.random() * cumulative_fitness[-1])
        spin_wheel.append(
            (spin_wheel[0] + len(population) / 2) % len(population)
        )

        for index in spin_wheel:
            yield population[find_nearest(cumulative_fitness, index)]


# Rank selection: Potentially useful at end of runs
def rank_selection(population):
    fitness_array = np.array(
        [individual.fitness_score for individual in population]
    )
    index_sort = np.argsort(fitness_array)
    rank_array = np.zeros(len(fitness_array))
    for i in range(len(fitness_array)):
        rank_array[index_sort[i]] = i
    cumulative_rank = np.cumsum(rank_array)

    for i in range(len(population)):
        spin_wheel = rnd.random() * cumulative_rank[-1]
        parent = population[find_nearest(cumulative_rank, spin_wheel)]
        yield parent


# TODO: borrowed function for now
def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


# yet to be implemented
"""
def elitism_selection():
"""


"""
# test
from list_representation import NoteList as music_rep
from fitness import calculate_fitness_only_C as fitness

population = music_rep.generate_new_population(11)
print("Starting fitness values: ")
for individual in population:
    fitness(individual)
print([individual.fitness_score for individual in population])

rank = rank_selection(population)
for parent in rank:
    print(parent.fitness_score)

sus = sus_selection(population)
for parent in sus:
    print(parent.fitness_score)

roulette = roulette_selection(population)
for parent in roulette:
    print(parent.fitness_score)

tournament = tournament_selection(population)
for parent in tournament:
    print(parent.fitness_score)
"""
