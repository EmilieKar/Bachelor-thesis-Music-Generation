import os
import pytest

from EvolutionaryMusic.list_representation import NoteList
from EvolutionaryMusic.note import Note
from EvolutionaryMusic.node_functions import (
    Pitch,
    Duration,
    Volume,
    Repeat,
    RepeatInverse,
)
from EvolutionaryMusic.sims_tree import SimsTree
from EvolutionaryMusic.sims_node import SimsNode


@pytest.fixture
def song1():
    return NoteList(
        [
            Note(60, 1),
            Note(61, 1),
            Note(61, 1),
            Note(60, 1),
            Note(59, 1),
            Note(57, 2),
            Note(60, 1),
        ]
    )


@pytest.fixture
def song2():
    return NoteList(
        [
            Note(60, 1),
            Note(61, 1),
            Note(62, 1),
            Note(63, 1),
            Note(64, 1),
            Note(65, 1),
            Note(66, 1),
        ]
    )


@pytest.fixture
def song4():
    return NoteList(
        [
            # Note(pitch, duration, volume, next_note)
            Note(10, 4, 100, 1),
            Note(63, 1, 40, 0),
            Note(64, 1, 50, 4),
            Note(60, 1, next_note=1),
            Note(59, 1, next_note=1),
            Note(57, 2, next_note=2),
            Note(60, 1, next_note=1),
            Note(60, 1 / 4, next_note=0),
        ]
    )


@pytest.fixture
def song_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../songs/tests"))


@pytest.fixture
def song_names():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../songs/tests"))

    return [
        os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    ]


@pytest.fixture
def song_sims():
    return [Note(5, 1, 100), Note(100, 4, 50), Note(61, 0.25, 100)]


@pytest.fixture
def add_3():
    return Pitch(3)


@pytest.fixture
def add_neg_3():
    return Pitch(-3)


@pytest.fixture
def mult_2():
    return Duration(2)


@pytest.fixture
def mult_div_2():
    return Duration(1 / 2)


@pytest.fixture
def volume_5():
    return Volume(5)


@pytest.fixture
def repeat_2():
    return Repeat(2)


@pytest.fixture
def repeat_inv():
    return RepeatInverse(1)


@pytest.fixture
def sims_tree():
    root_node = SimsNode(RepeatInverse(1))
    root_node.children = [SimsNode(Pitch(0), Repeat(2)), SimsNode(Pitch(2), Repeat(1))]
    root_node.children[0].children = [
        SimsNode(Pitch(1), Repeat(1)),
        SimsNode(Pitch(2), Repeat(1)),
    ]
    return SimsTree(root_node, 60)
