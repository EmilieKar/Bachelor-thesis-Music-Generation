#  A suggested version of trees inspired by Karl Sims image algorithm
#  The tree saves a root pitch and then calculates a note for all leaf nodes
#  using functions saved in each node. The leaf nodes final note values become
#  relative to the root_note value.

import random as rnd
import timeit

from EvolutionaryMusic.sims_node import SimsNode

# consider creating Node.mutate_function() in SimsNode for better encapsulation
# and avoiding import from nodefunctions
from EvolutionaryMusic.node_functions import NoteFunction

from EvolutionaryMusic.generic_tree import Tree
from EvolutionaryMusic.note import Note


class SimsTree(Tree):
    """
    Tree that saves a function in each node, used for calculating the pitch.
    """

    def __init__(self, root_node=None, root_note=None):
        self.root_note = root_note
        super().__init__(root_node)

    def randomize(self):
        """ Generates a random tree """
        self.root_node = SimsNode().generate_tree()
        self.root_note = Note.generate_random_note()

    def get_notes(self):
        """ Get list of notes in-order traversal (in the same order as they should be played) """
        return self.root_node.get_notes(self.root_note)

    def print_tree(self):
        self.root_node.print_tree()

    #                           Generator
    # ---------------------------------------------------------------

    def uniform_crossover(self, mate):
        """crossover using uniform crossover, returning modified copies of the parents"""
        time0 = timeit.default_timer()
        self_copy = self.root_node.copy_node()
        mate_copy = mate.root_node.copy_node()
        time1 = timeit.default_timer()
        cross_children = self_copy.crossover_uniform(mate_copy)
        time2 = timeit.default_timer()
        return [
            SimsTree(root_node=cross_children[0], root_note=self.root_note),
            SimsTree(root_node=cross_children[1], root_note=mate.root_note),
        ], time1 - time0, time2 - time1

    def single_point_crossover(self, mate):
        time0 = timeit.default_timer()
        child1 = self.copy_tree()
        child2 = mate.copy_tree()
        time1 = timeit.default_timer()
        node1 = rnd.choice(child1.node_list)
        node2 = rnd.choice(child2.node_list)
        node1.switch(node2)
        time2 = timeit.default_timer()
        return [child1, child2], time1 - time0, time2 - time1

    def _get_crossovers(self):
        functions = [self.uniform_crossover, self.single_point_crossover]
        weights = [50, 50]
        return functions, weights

    #                           Mutation
    # ---------------------------------------------------------------
    def _get_mutations(self):
        """
        Returns a list of the possible mutations.
        """
        return [
            self.change_root_note,
            self.mutate_swap,
            self.mutate_function,
            self.mutate_parameter,
        ]

    def change_root_note(self):
        """
        Change the root note of the tree.
        """
        self.root_note = Note.generate_random_note()

    def mutate_swap(self):
        """ Swap node function for two randomly selected nodes """
        nodes = self.root_node.get_nodes()
        if len(nodes) < 2:
            # print("Tree is too small for swap")
            # self.root_node.print_tree()
            pass
        else:
            node1, node2 = rnd.sample(nodes, k=2)
            node1.function, node2.function = node2.function, node1.function

    def mutate_function(self):
        """ Generates a new function and parameter  """
        nodes = self.root_node.get_nodes()
        node = rnd.choice(nodes)
        node.function = NoteFunction()

    def mutate_parameter(self):
        """ Generates a new parameter but doesn't change function type """
        nodes = self.root_node.get_nodes()
        node = rnd.choice(nodes)
        node.function.new_parameters()

    def copy_tree(self):
        tree = SimsTree()
        tree.root_node = self.root_node.copy_node()
        tree.root_note = self.root_note.copy()
        tree.node_list = tree.root_node.get_nodes()
        return tree
