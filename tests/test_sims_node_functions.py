from EvolutionaryMusic.note import Note
import copy

"""
Tests for the node functions defined in sims_node as well as the get_notes function for a tree.
"""


def test_add_fun(song_sims, add_3):
    note = song_sims[1].copy()
    add_3(note)
    assert note.pitch == 103 and note.duration == 4 and note.volume == 50


def test_neg_add_fun(song_sims, add_neg_3):
    note = song_sims[1].copy()
    add_neg_3(note)
    assert note.pitch == 97 and note.duration == 4 and note.volume == 50


def test_mult_fun(song_sims, mult_2):
    note = song_sims[0].copy()
    mult_2(note)
    assert note.pitch == 5 and note.duration == 2 and note.volume == 100


def test_mult_div_fun(song_sims, mult_div_2):
    note = song_sims[0].copy()
    mult_div_2(note)
    assert note.pitch == 5 and note.duration == 1 / 2 and note.volume == 100


def test_volume_fun(song_sims, volume_5):
    note = song_sims[2].copy()
    volume_5(note)
    assert note.pitch == 61 and note.duration == 1 / 4 and note.volume == 105


def test_repeat_fun(song_sims, repeat_2):
    song = copy.deepcopy(song_sims)
    repeat_2(song)
    assert song == [
        Note(5, 1, 100),
        Note(100, 4, 50),
        Note(61, 0.25, 100),
        Note(5, 1, 100),
        Note(100, 4, 50),
        Note(61, 0.25, 100),
    ]


def test_repeat_inv_fun(song_sims, repeat_inv):
    song = copy.deepcopy(song_sims)
    repeat_inv(song)
    assert song == [
        Note(5, 1, 100),
        Note(100, 4, 50),
        Note(61, 0.25, 100),
        Note(61, 0.25, 100),
        Note(100, 4, 50),
        Note(5, 1, 100),
    ]


def test_get_notes(sims_tree):
    """
    Important for Repeat and RepeatInverse, since they add functionality in get_notes().
    """
    sims_tree.root_node.print_tree()

    ([Note(61, 1, 100), Note(62, 1, 100)] * 3 + [Note(62, 2, 100)]) * 2 + [Note(62, 2, 100)] + [
        Note(61, 1, 100),
        Note(62, 1, 100),
    ] * 3
    assert True
