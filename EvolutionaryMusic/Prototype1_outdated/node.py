import random as rnd
from EvolutionaryMusic.note import Note


class Node:
    def __init__(self, note=None, depth=0):
        self.left = None
        self.right = None
        self.note = note
        self.depth = depth

    def __str__(self):
        if(self.left is None and self.right is None):
            return f'Node({str(self.note)})'
        else:
            return f'Node({str(self.depth)})'

    # For debugging
    def check_invariant(self):
        if self.left and self.right:
            if self.note:
                print("\n nåt konstigt!!")
                print(f"Nod {self} har not {self.note} \
                        och barn {self.left}, {self.right}")
            self.left.check_invariant()
            self.right.check_invariant()

    def generate_tree(self, n):
        """ Generate a binary tree with n 'levels', with a note in each node"""
        self.note = Note(60 + rnd.randint(-12, 12))
        if n > 0:
            self.left = Node().generate_tree(n - 1)
            self.right = Node().generate_tree(n - 1)
        return self

    def update(self):
        """ Update this node's children's depth and durations """
        for child in [self.left, self.right]:
            if child:
                child.depth = self.depth + 1
                child.update()
                if child.note:
                    child.note.duration = 2**(2 - child.depth)

    def for_all(self, fun):
        """ Apply function fun this node and all its children """
        fun(self)
        for child in [self.left, self.right]:
            if child:
                fun(child)
                child.for_all(fun)

    def switch(self, other):
        """ Switch two nodes """
        self.left, other.left = other.left, self.left
        self.right, other.right = other.right, self.right
        self.note, other.note = other.note, self.note

    def generate_bar(self, prob=1, a=0.5):
        """ Generate a bar of music

        Using tree representation to determine rhythms described in <länk>
        The leaves contain notes with random pitches. The duration of
        each note is related to its depth in the tree.

        Parameters:
        prob - The initial probability of a node having children
        a - The probability of a node having children is multiplied
            with a**depth
        """
        if rnd.random() < prob:
            self.left = Node(depth=self.depth + 1).generate_bar(prob * a)
            self.right = Node(depth=self.depth + 1).generate_bar(prob * a)
        else:
            pitch = 60 + rnd.randint(-12, 12)
            duration = 2**(2 - self.depth)
            self.note = Note(pitch, duration)
        return self

    def to_list(self, onlyLeaves=True):
        """ Return a list of all nodes (inorder) """
        left = self.left.to_list(onlyLeaves) if self.left else []
        right = self.right.to_list(onlyLeaves) if self.right else []
        if onlyLeaves:
            middle = [self.note] if self.note else []
        else:
            middle = [str(self)]
        return left + middle + right

    def print_tree(self, str_='', symb=''):
        """ Print the tree starting from this node """
        angle = '|--' if symb == '|' else '\'--'
        print(str_ + angle, self)
        str_ += symb + ' ' * 4
        children = [child for child in [self.left, self.right] if child]
        # ^ easy exentable to
        # children = [child for child in self.children if child]
        for i in range(len(children)):
            symb = '|' if i != len(children) - 1 else ' '
            children[i].print_tree(str_, symb)

    def size(self):
        """ Return the number of children of this node """
        if self.left is None and self.right is None:
            return 0
        return self.left.size() + 2 + self.right.size()

    # Borrowed from internet. See link at end of document for reference
    def get_random_node(self):
        """ Return a random node """
        children = self.size()

        if not self.left and not self.right:
            return self

        left_children = self.left.size()
        r = rnd.randint(1, children)
        if r <= left_children:
            return self.left.get_random_node()
        elif r == left_children + 1:
            return self
        else:
            return self.right.get_random_node()

    def get_random_leafnode(self):
        """ Return a random leafnode"""
        node = self.get_random_node()
        if node.left is None and node.right is None:
            return node
        else:
            return node.get_random_leafnode()
