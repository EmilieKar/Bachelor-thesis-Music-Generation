import random as rnd

from EvolutionaryMusic.note import Note
from EvolutionaryMusic.crossover import crossover_list


class NoteList:
    def __init__(self, notes=[]):  # optional list as init argument
        self.notes = notes  # list: a list of Notes with pitches
        self.fitness_score = 0.0

    # makes it possible to call individual[i] instead of
    # individual.get_notes()[i], could be an alternative to get_notes()
    def __getitem__(self, idx):
        return self.notes[idx]

    def __setitem__(self, idx, item):
        self.notes[idx] = item

    def __len__(self):
        return len(self.notes)

    def __str__(self):
        pitches_string = "[ "
        for note in self.notes:
            pitches_string += str(note) + ", "
        return pitches_string + "]"

    def __repr__(self):
        return str(self.get_notes())

    def get_notes(self):
        return self.notes

    # population size = how many music compositions,
    # individual_size = how many indviduals in each population
    # population size = how many music compositions
    # individual_size = how many indviduals in each population
    def generate_new_population(pop_size=1, individual_size=10):
        return [
            NoteList(
                [Note.generate_random_note() for _ in range(individual_size)]
            )
            for _ in range(pop_size)
        ]

    def crossover(self, mate):
        children = crossover_list(self.notes, mate.notes)
        return [NoteList(children[0]), NoteList(children[1])]

    def mutate(self):
        """ Mutates by a random mutation on object """
        rnd.choice(self._get_mutations())()

    def scramble_mutation(self):
        """ Scramble a random section of the note list """
        [idx1, idx2] = sorted(rnd.sample(range(len(self.notes) + 1), 2))
        middle = self.notes[idx1:idx2]
        rnd.shuffle(middle)
        self.notes = self.notes[:idx1] + middle + self.notes[idx2:]

    def swap_mutation(self):
        """ Swap two random notes """
        [idx1, idx2] = sorted(rnd.sample(range(len(self.notes)), 2))
        self.notes[idx1], self.notes[idx2] = self.notes[idx2], self.notes[idx1]

    def inversion_mutation(self):
        """ Inverse a random section of the note list """
        [idx1, idx2] = sorted(rnd.sample(range(len(self.notes) + 1), 2))
        middle = self.notes[idx1:idx2]
        middle.reverse()
        self.notes = self.notes[:idx1] + middle + self.notes[idx2:]

    def _get_mutations(self):
        """ Return a list of this class' mutation methods """
        return [
            self.swap_mutation,
            self.inversion_mutation,
            self.scramble_mutation,
        ]


def demo():
    n = NoteList.generate_new_population()[0]
    print(n)
    n.mutate()
    print(n)


if __name__ == "__main__":
    demo()
