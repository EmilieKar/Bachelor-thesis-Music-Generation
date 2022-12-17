import pytest

from EvolutionaryMusic.fitness import (
    FitnessSubrater,
    neighbouring_pitch_range,
    direction_of_melody,
    stability_of_melody,
    pitch_range,
    unique_notes,
    equal_consecutive_notes,
    unique_rhythms,
)
from EvolutionaryMusic.midi import Midi


def test_subrater_fitness(song_path, song_names):
    calibration = FitnessSubrater(song_path)

    songs = [Midi.convert_midi_to_list(name) for name in song_names]
    score = 0
    for song in songs:
        calibration.fitness(song)
        score += song.fitness_score
    score /= len(song_names)

    assert score >= 0.0 and score <= 1.0


# Test if fitness values are in the defined range (between 0 and 1)
@pytest.mark.parametrize("song", ["song1", "song2"])
def test_neighbouring_pitch_range(song, request):
    s = request.getfixturevalue(song)
    score = neighbouring_pitch_range(s)

    assert score >= 0.0 and score <= 1.0


@pytest.mark.parametrize("song", ["song1", "song2"])
def test_direction_of_melody(song, request):
    s = request.getfixturevalue(song)
    score = direction_of_melody(s)

    assert score >= 0.0 and score <= 1.0


@pytest.mark.parametrize("song", ["song1", "song2"])
def test_stability_of_melody(song, request):
    s = request.getfixturevalue(song)
    score = stability_of_melody(s)

    assert score >= 0.0 and score <= 1.0


@pytest.mark.parametrize("song", ["song1", "song2"])
def test_pitch_range(song, request):
    s = request.getfixturevalue(song)
    score = pitch_range(s)

    assert score >= 0.0 and score <= 1.0


@pytest.mark.parametrize("song", ["song1", "song2"])
def test_unique_notes(song, request):
    s = request.getfixturevalue(song)
    score = unique_notes(s)

    assert score >= 0.0 and score <= 1.0


@pytest.mark.parametrize("song", ["song1", "song2"])
def test_equal_consecutive_notes(song, request):
    s = request.getfixturevalue(song)
    score = equal_consecutive_notes(s)

    assert score >= 0.0 and score <= 1.0


@pytest.mark.parametrize("song", ["song1", "song2"])
def test_unique_rhythms(song, request):
    s = request.getfixturevalue(song)
    score = unique_rhythms(s)

    assert score >= 0.0 and score <= 1.0


# Test that every subrater works with a concrete example


def test_neighbouring_pitch_range_2(song4):
    score = neighbouring_pitch_range(song4)
    assert score == pytest.approx(1 / 8)


def test_direction_of_melody_2(song4):
    score = direction_of_melody(song4)
    assert score == pytest.approx(2 / 8)


def test_stability_of_melody_2(song4):
    score = stability_of_melody(song4)
    assert score == pytest.approx(2 / 8)


def test_pitch_range_2(song4):
    score = pitch_range(song4)
    assert score == pytest.approx(10 / 64)


def test_unique_notes_2(song4):
    score = unique_notes(song4)
    assert score == pytest.approx(6 / 8)


def test_equal_consecutive_notes_2(song4):
    score = equal_consecutive_notes(song4)
    assert score == pytest.approx(1 / 8)


def test_unique_rhythms_2(song4):
    score = unique_rhythms(song4)
    assert score == pytest.approx(4 / 8)
