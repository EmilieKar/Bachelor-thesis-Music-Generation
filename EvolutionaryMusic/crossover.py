import random as rnd
import copy


# Chooses between possible crossovers
def crossover_list(parent1, parent2):
    function = rnd.choice([crossover_single_point_list, crossover_multi_point_list, crossover_uniform_list])
    return function(parent1, parent2)


def crossover_tree(parent1, parent2):
    function = rnd.choice([crossover_uniform_tree])
    return function(parent1, parent2)


# Crossover at a single point
# Note: size of parent 1 is used to determine "length of child"
def crossover_single_point_list(parent1, parent2, cross_index=None):
    if cross_index is None:
        cross_index = rnd.randrange(0, len(parent1), 1)
    child1 = parent1[:cross_index] + parent2[cross_index:]
    child2 = parent2[:cross_index] + parent1[cross_index:]
    return [child1, child2]


# Crosses over list at multiple points
def crossover_multi_point_list(parent1, parent2):
    children = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
    # could use some thought
    points = [
        rnd.randint(0, int(len(parent1) / 2)) for _ in range(0, int(len(parent1) / 2))
    ]
    for i in points:
        children = crossover_single_point_list(children[0], children[1], i)
    return children


# Crosses over uniformly
def crossover_uniform_list(parent1, parent2):
    children = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
    for i in range(len(parent1)):
        if rnd.random() < 0.5:
            temp = children[0][i]
            children[0][i] = children[1][i]
            children[1][i] = temp
    return children


# For every common node in parents flip a coin to see if you should change children
def crossover_uniform_tree(parent1, parent2):  # send in copies of parents roots
    if (parent1 is not None and parent2 is not None):
        if rnd.random() > 0.5:
            parent1.switch(parent2)
        crossover_uniform_tree(parent1.left, parent2.left)
        crossover_uniform_tree(parent1.right, parent2.right)
    return [parent1, parent2]


# Crosses over using the average value of the pitches
# Returns two identical children
# Probably shouldn't be used for this application (maybe useful later)
def crossover_arithmetic(parent1, parent2):
    children = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
    for i in range(len(parent1)):
        average_pitch = int((children[0][i].pitch + children[1][i].pitch) / 2)
        children[0][i].pitch = average_pitch
        children[1][i].pitch = average_pitch
    return children


# TEST
"""
song1 = [1,2,3,4,5,6,7,8,9]
song2 = [11,12,13,14,15,16,17,18,19]


song1 = [Note(1, 1), Note(2, 1),
        Note(3, 1), Note(4, 1),
        Note(5, 1), Note(6, 1),
        Note(7, 1), Note(8, 1)]


song2 = [Note(10, 1), Note(20, 1),
        Note(30, 1), Note(40, 1),
        Note(50, 1), Note(60, 1),
        Note(70, 1), Note(80, 1)]


print(song1)
print(song2)
print(crossover_arithmetic(song1,song2))

print(crossover_single_point(song1, song2))
print(crossover_multi_point(song1, song2))
print(crossover_uniform(song1, song2))
"""
