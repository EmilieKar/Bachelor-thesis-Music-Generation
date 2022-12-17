import random as rnd
import copy

from EvolutionaryMusic.generic_node import Node
from EvolutionaryMusic.node_functions import NoteFunction, SequenceFunction
from EvolutionaryMusic.note import Note


class SimsNode(Node):
    """Node used for the SimsTree."""

    def __init__(self, note_function=None, sequence_function=None):
        super().__init__()
        if note_function:
            self.function = note_function
        else:
            self.function = NoteFunction()
        if sequence_function:
            self.sequence_function = sequence_function
        else:
            self.sequence_function = SequenceFunction()
        # NodeFunction.get_random()

    def __str__(self):
        return f"{self.function}, {self.sequence_function}"

    # Returns a list of notes (in order traversal) from the tree.
    def get_notes(self, note, all_nodes=False):

        node_note = self.function(note)

        left = self.children[0].get_notes(node_note, all_nodes) if self.children else []
        right = self.children[1].get_notes(node_note, all_nodes) if self.children else []
        middle = [node_note] if all_nodes or not self.children else []

        sublist = left + middle + right
        # if len(sublist) > 3:
        #    sublist = self.sequence_function(sublist)
        return self.sequence_function(sublist)

    #  Generated a random tree with decreasing probability to
    #  generate new nodes as the tree gets deeper
    def generate_tree(self, prob=1, a=0.8):
        if rnd.random() < prob:
            child1 = SimsNode().generate_tree(prob * a)
            child2 = SimsNode().generate_tree(prob * a)
            self.children = [child1, child2]
        return self

    # switches children and value of two nodes from two different trees (used in crossover)
    def switch(self, other):
        """ Switch two nodes """
        self.children, other.children = other.children, self.children
        self.function, other.function = other.function, self.function
        self.sequence_function, other.sequence_function = other.sequence_function, self.sequence_function

    def copy_node(self):
        node = SimsNode()
        node.function = self.function.copy()
        node.sequence_function = self.sequence_function.copy()
        node.children = [child.copy_node() for child in self.children if child]
        return node
