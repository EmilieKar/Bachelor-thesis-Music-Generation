import random as rnd

from abc import ABC, abstractmethod
from EvolutionaryMusic.note import Note


# Store all parameters at the same place
DURATION_PROB = 0.5
PITCH_PROB = 0.8
VOLUME_PROB = 0.5
ARTICULATION_PROB = 0.2
NEXT_NOTE_PROB = 0.3

TEMPO_PROB = 0
REPEAT_PROB = 0.1
DYNAMICS_PROB = 0.1
INVERSE_PROB = 0.1

MAX_DURATION = 16  # beats - four bars in 4/4 measure
MIN_DURATION = 4 / 128  # beats - 128th note, or, a semihemidemisemiquaver:)


class NodeFunction(ABC):
    """ Base class for node functions """

    def __init__(self, param=None):
        self.param = param if param else self.generate_parameters()

    def __repr__(self):
        return "<" + type(self).__name__ + ": " + str(self.param) + ">"

    def new_parameters(self):
        self.param = self.generate_parameters()

    def copy(self):
        new = type(self)()
        new.param = self.param
        return new

    @abstractmethod
    def generate_parameters(self):
        """ Generate new parameters """
        pass

    @abstractmethod
    def __call__(self):
        pass


class Pitch(NodeFunction):
    """ Change pitch """

    prob = PITCH_PROB

    def __call__(self, note):
        note.pitch = max(0, min(127, note.pitch + self.param))

    def generate_parameters(self):
        return int(rnd.gauss(0, 3))


class Duration(NodeFunction):
    """ Change the duration."""

    prob = DURATION_PROB

    def __call__(self, note):
        # smallest duration is 1/128th, longest duration is 4 bars
        # old diff tries to set next_note to keep the same pause as before when duration is changed
        old_diff = note.duration - note.next_note
        note.duration = max(MIN_DURATION, min(MAX_DURATION, note.duration * self.param))
        note.next_note = max(MIN_DURATION, min(MAX_DURATION, note.duration - old_diff))

    def generate_parameters(self):
        return 2 ** int(rnd.gauss(0, 1))


class Volume(NodeFunction):
    """ Change volume """

    prob = VOLUME_PROB

    def __call__(self, note):
        note.volume = max(1, min(127, note.volume + self.param))

    def generate_parameters(self):
        return int(rnd.gauss(0, 5))


class NextNote(NodeFunction):
    """
    Changes if the node will play simultaneously as the next note or not
    If next_note > 1 there will be a pause after the note
    """

    prob = NEXT_NOTE_PROB

    def __call__(self, note):
        # if pause, make sure it's not longer than max_duration
        note.next_note = min(note.duration * self.param, MAX_DURATION + note.duration)

    def generate_parameters(self):
        # TODO use a better math function here
        # If 0 notes are played simultaneously, 1 note played immediately after
        # > duration there will be a pause after the note
        if rnd.random() < 0.2:  # chord_prob = 0.2 TODO move out these hyperparameters
            return 0.0
        if rnd.random() < 0.8:  # directly_after_prob = 0.8
            return 1.0
        return 2 ** int(rnd.gauss(0, 1))


class Repeat(NodeFunction):
    """ Repeat the sequence """

    prob = REPEAT_PROB

    def __call__(self, note_list):
        note_list *= self.param

    def generate_parameters(self):
        return rnd.choice([2, 3])


class Tempo(NodeFunction):
    """ Exponential tempo change of sequence (accelerando/ritardando) """

    prob = TEMPO_PROB

    def __call__(self, note_list):
        for i, note in enumerate(note_list):
            note.duration *= self.param ** i

    def generate_parameters(self):
        return rnd.gauss(1, 0.05)


class Dynamics(NodeFunction):
    """ Linear volume change of sequence (crescendo/diminuendo) """

    prob = DYNAMICS_PROB

    def __call__(self, note_list):
        if note_list:
            dv = self.param / len(note_list)
            for i, note in enumerate(note_list):
                volume = int(note.volume + i * dv)
                note.volume = max(1, min(127, volume))

    def generate_parameters(self):
        return rnd.choice([-30, -20, -10, +10, +20, +30])


class RepeatInverse(NodeFunction):
    """ Repeat sequence inverted """

    prob = INVERSE_PROB

    def __call__(self, list_):
        list_ += list_[::-1]

    def generate_parameters(self):
        pass


class FunctionCombo:
    def __init__(self, functions=[]):
        self.funs = [f() for f in functions if f.prob > rnd.random()]

    def __call__(self, item):
        for f in self.funs:
            f(item)
        return item

    def __repr__(self):
        return ", ".join([str(f) for f in self.funs])

    def new_parameters(self):
        if self.funs:
            rnd.choice(self.funs).new_parameters()

    def copy(self):
        new = type(self)()
        new.funs = [f.copy() for f in self.funs]
        return new


class NoteFunction(FunctionCombo):
    def __init__(self):
        functions = [Pitch, Duration, Volume, NextNote]
        super().__init__(functions)

    def __call__(self, note):
        note = note.copy()
        return super().__call__(note)


class SequenceFunction(FunctionCombo):
    def __init__(self):
        functions = [Repeat, RepeatInverse, Dynamics, Tempo]
        super().__init__(functions)


if __name__ == "__main__":
    f = SequenceFunction()
    g = NoteFunction()

    notelist = [Note.generate_random_note() for _ in range(4)]
    note = Note(60, 2, 100)
    print(f)
    f.new_parameters()
    print(f)
    print(notelist)
    print(f(notelist))
