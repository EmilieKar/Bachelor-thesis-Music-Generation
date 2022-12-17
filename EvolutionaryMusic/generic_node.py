import random as rnd


class Node:
    """Generic Node class with common functions for all tree node implementations"""
    def __init__(self):
        self.children = []

    # Prints the tree in a readable format.
    def print_tree(self, str_='', symb=''):
        """ Print the tree starting from this node """
        angle = '|--' if symb == '|' else '\'--'
        print(str_ + angle, self)
        str_ += symb + ' ' * 4
        children = [child for child in self.children if child]
        for i in range(len(children)):
            symb = '|' if i != len(children) - 1 else ' '
            children[i].print_tree(str_, symb)

    # Returns all nodes in a list (inorder traversal).
    def get_nodes(self):
        if not self.children:
            return [self]
        return self.children[0].get_nodes() + [self] + self.children[1].get_nodes()

    #                           Generator
    # ---------------------------------------------------------------
    # Uniform crossover: send in nodes
    def crossover_uniform(self, mate):
        """Uniform crossover for all common nodes in two trees"""
        if rnd.random() > 0.5:
            self.switch(mate)
        if self.children and mate.children:
            self.children[0].crossover_uniform(mate.children[0])
            self.children[1].crossover_uniform(mate.children[1])
        return [self, mate]
