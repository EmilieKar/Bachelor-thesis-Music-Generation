import pytest
from EvolutionaryMusic.midi import Midi

@pytest.mark.parametrize("song", ["song1", "song2"])
def test_on_equals_off(song, request):
    s = request.getfixturevalue(song)
    
    mid = Midi.convert_to_midi(s)

    on = 0
    off = 0

    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == "note_on":
                on = on + 1
            elif msg.type == "note_off":
                off = off + 1

    assert on == off
