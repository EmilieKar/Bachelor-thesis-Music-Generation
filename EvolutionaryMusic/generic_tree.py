import random as rnd


class Tree:
    """
    Generic Tree class with common functions for all tree implementations.

    Updates the note list automatically.
    """

    def __init__(self, root_node=None):
        self.root_node = root_node
        self.fitness_score = 0

        if root_node is None:
            self.randomize()

        self.node_list = self.root_node.get_nodes()

    def __str__(self):
        res = ""
        n = 4
        for i, note in enumerate(self.get_notes()):
            res += f"[{note}]\t"
            res += "\n" if (i + 1) % n == 0 else ""

        return res

    def __repr__(self):
        return str(self.get_notes())

    def __getitem__(self, index):
        """returns the note from get_notes at given index"""
        return self.get_notes()[index]

    def __len__(self):
        """returns length of note_list derived from tree"""
        return len(self.get_notes())
    
    def number_of_nodes(self):
        return len(self.node_list)

    def get_notes(self):
        self.root_node.get_notes()

    #                           Generator
    # ---------------------------------------------------------------
    def crossover(self, mate):
        functions, weights = self._get_crossovers()
        # weigted selection of crossover function to make single point less likely.
        [child1, child2], time1, time2 = rnd.choices(functions, weights)[0](mate)
        return [child1, child2], time1, time2  # rnd.choices(functions, weights)[0](mate)

    #                           Mutations
    # ---------------------------------------------------------------
    def mutate(self):
        """chooses a random mutation from specific tree representation and generic mutations"""
        rnd.choice(self._get_mutations() + self._get_generic_mutations())()
        self.node_list = self.root_node.get_nodes()

    def _get_generic_mutations(self):
        """returns list of the available generic tree mutations"""
        return [self.mutate_new_treebranch]

    def mutate_new_treebranch(self):
        """generates a new tree and replaces some node in the tree with the new tree"""
        node = rnd.choice(self.node_list)
        node.generate_tree(1, 0.3)
