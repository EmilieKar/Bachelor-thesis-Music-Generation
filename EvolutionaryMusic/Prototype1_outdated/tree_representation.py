import random as rnd
import copy
from crossover import crossover_tree
from node import Node


class Tree:
    def __init__(self, root=None):
        self.root = root
        self.fitness_score = 0

    def __getitem__(self, idx):
        return self.get_notes()[idx]

    """
    def __setitem__(self, idx, item):
        self.get_notes()[idx] = item
    """

    def __repr__(self):
        return str(self.get_notes())

    def generate_new_population(pop_size=1):
        new_population = []
        for i in range(pop_size):
            individual = Tree()
            individual.randomize()
            new_population.append(individual)
        return new_population

    def mutate(self):
        rnd.choice(self._get_mutations())()

    def mutate_swap(self):
        node1 = self.root.left.get_random_node()
        node2 = self.root.right.get_random_node()
        node1.switch(node2)
        self.root.update()

    def mutate_new_treebranch(self):
        # todo : bättre att välja bara lövnoder ?
        node = self.root.get_random_node()
        node.note = None
        node.generate_bar(prob=1, a=0.2)

    # Fungerar inte när man kör det i generatorn
    def mutate_repeat(self):
        node1 = self.root.left.get_random_node()
        node2 = self.root.right.get_random_node()
        node2.left = node1.left
        node2.right = node1.right
        node2.note = node1.note
        self.tree.update()

    def mutate_transpose(self):
        """ Transposes all children of a random node """
        pitch_diff = rnd.randint(-6, 6)

        def transpose(node):
            if node.note:
                node.note.pitch += pitch_diff

        node = self.root.get_random_node()
        node.for_all(transpose)

    def mutate_split(self):
        """Mutates a random note into two smaller notes of the same pitch"""
        node = self.root.get_random_leafnode()
        node.generate_bar(prob=1, a=0)
        node.left.note = node.note
        node.right.note = node.note
        node.update()
        node.note = None

    def _get_mutations(self):
        """ Return a list of this class' mutation methods """
        return [self.mutate_swap, self.mutate_new_treebranch,
                self.mutate_transpose, self.mutate_split]

    def crossover(self, mate):
        children = crossover_tree(copy.deepcopy(self.root), copy.deepcopy(mate.root))
        return [Tree(children[0]), Tree(children[1])]

    def get_notes(self):
        return self.root.to_list()

    def randomize(self):
        self.root = Node().generate_bar()

    def __len__(self):
        return len(self.get_notes())


# tree demo
def demo():

    tree = Tree()
    tree.randomize()
    print("\nGenerated this tree:")
    tree.root.print_tree()
    print("\nWhich represents this bar of music: ")
    print(tree)

    tree1 = Tree()
    tree1.randomize()
    print("Tree 1")
    tree1.root.print_tree()
    tree2 = Tree()
    tree2.randomize()
    print("Tree 2")
    tree2.root.print_tree()
    children = tree1.crossover(tree2)
    print("Child 1")
    children[0].root.print_tree()
    print("Child 2")
    children[1].root.print_tree()

if __name__ == "__main__":
    demo()


"""
notes

get_random_node method borrowed from
https://hackernoon.com/how-to-select-a-random-node-from-a-tree-with-equal-probability-childhood-moments-with-father-today-0ip32dp
"""
