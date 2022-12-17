import pyo
import atexit
from threading import Thread
from mido import Message, MidiFile, MidiTrack
from math import inf

from EvolutionaryMusic.list_representation import NoteList
from EvolutionaryMusic.note import Note


class Midi:
    def __init__(self):
        self.server = pyo.Server(duplex=0).boot()
        atexit.register(self.close)

    def close(self):
        print("Closing pyo server")
        self.server.shutdown()

    @staticmethod
    def display_midi_info(mid, ignore_control_change=True):
        if isinstance(mid, str):
            try:
                mid = MidiFile(mid)
            except FileNotFoundError:
                print(f"{mid} not found, skipping file.")
                return

        for i, track in enumerate(mid.tracks):
            print("Track {}: {}".format(i, track.name))
            for msg in track:
                if ignore_control_change and msg.type == "control_change":
                    continue
                print(msg)

    def play(self, individual, msg="press enter to continue..."):
        song = self.convert_to_midi(individual)

        self.server.start()

        # i'm not sure if this is how you're supposed to do it,
        # but you can use this to change the volume
        # default = 1
        volume_multiplier = 0.5

        # set up a synth (i have no idea how this works)
        mid = pyo.Notein()
        amp = pyo.MidiAdsr(mid["velocity"])
        pit = pyo.MToF(mid["pitch"])
        osc = pyo.Osc(pyo.SquareTable(), freq=pit, mul=amp * volume_multiplier).mix(1)
        _ = pyo.STRev(osc, revtime=1, cutoff=4000, bal=0.2).out()

        # create mutable score object for input thread
        score = {"value": None}
        thread = Thread(
            target=self.set_score,
            args=(
                msg,
                score,
            ),
        )
        thread.start()

        # play the song (blocking)
        for message in song.play():
            # break if we 'score' the song before it has finished
            if score["value"] is not None:
                break
            self.server.addMidiEvent(*message.bytes())

        self.server.stop()

        # wait for the set_score thread if we have not entered a score yet
        thread.join()

        # replace the score object with the score value
        score = score["value"]

        return score

    def set_score(self, msg, score):
        score["value"] = input(msg)

    @staticmethod
    def convert_to_midi(individual) -> MidiFile:
        """
        Convert a list of notes to MidiFile
        """
        ticks = 480  # ticks per beat
        mid = MidiFile()
        mid.ticks_per_beat = ticks
        track = MidiTrack()
        mid.tracks.append(track)

        time = 0  # time to wait since last message
        current_notes = []  # currently sounding notes
        next_notes = [0]  # when next notes should start
        index = 0  # index of next note in individual

        while current_notes or next_notes:

            # sort both lists (needed for .pop later on)
            current_notes.sort(key=lambda x: -x.duration)
            next_notes.sort()

            next_off = current_notes[-1].duration if current_notes else inf
            next_on = next_notes[-1] if next_notes else inf
            next_event = min(next_on, next_off)

            time = int(next_event * ticks + 0.5)  # round to closest integer

            # update next notes and current notes durations
            next_notes = [n - next_event for n in next_notes]
            for note in current_notes:
                note.duration -= next_event

            if next_off < next_on:  # turn off note
                note = current_notes.pop()
                track.append(Message("note_off", note=note.pitch, velocity=0, time=time))
                time = 0

            else:  # start next note
                next_notes.pop()
                note = individual[index].copy()

                # check if there is already a sounding note with the same pitch
                same_pitches = [n for n in current_notes if n.pitch == note.pitch]
                if same_pitches:
                    old_note = same_pitches[0]
                    track.append(Message("note_off", note=note.pitch, velocity=0, time=time))
                    time = 0
                    track.append(
                        Message("note_on", note=note.pitch, velocity=note.volume, time=time)
                    )

                    # decide whether the new or old note defines the duration (pick the longest)
                    if old_note.duration <= note.duration:
                        current_notes.remove(old_note)
                        current_notes.append(note)

                else:
                    current_notes.append(note)
                    track.append(
                        Message("note_on", note=note.pitch, velocity=note.volume, time=time)
                    )
                    time = 0

                # update next_note list and move index
                if index < len(individual) - 1:  # update next_note if there are notes left
                    next_notes.append(note.next_note)
                index += 1

        return mid

    @staticmethod
    def convert_midi_to_NNinput(mid, count=0):
        """
        Read pitches, offsets, durations and volumes from midi file

        Input: MidiFile object or path to a midi file
        Return: A tuple of pitches, offsets, durations and volumes,
        where a note's offset is relative to the start of the previous note

        Example:
        Stand in EvolutionaryMusic
        >>> import midi
        >>> midi.Midi.convert_midi_to_NNinput("../songs/tests/test_song1.mid")
        output:
            ([36,  36,  36,  36,  36,  36,  36,  36,  48,  48],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
        """
        if isinstance(mid, str):
            try:
                mid = MidiFile(mid)
            except FileNotFoundError:
                print(f"{mid} not found, skipping file.")
                return

        notes_on = {}
        pitches = []
        offsets = []
        durations = []
        volumes = []

        for track in mid.tracks:
            time = 0
            for msg in track:
                time += msg.time / mid.ticks_per_beat
                if msg.type not in ("note_on", "note_off"):
                    continue

                if msg.velocity > 0:
                    for key, value in notes_on.items():
                        notes_on[key] = (value[0], value[1] + time)
                    notes_on[msg.note] = (len(pitches), 0)
                    pitches.append(msg.note)
                    offsets.append(time)
                    volumes.append(msg.velocity)
                    time = 0
                else:
                    order, t = notes_on[msg.note]
                    durations.append((order, t + time))

                if count != 0 and count == len(pitches):
                    break

        durations = [d[1] for d in sorted(durations, key=lambda d: d[0])]
        return pitches, offsets, durations, volumes

    @staticmethod
    def convert_midi_to_list(name):
        """
        Convert midi to NoteList

        Input: MidiFile object or path to midi file
        Return: NoteList (for compatibilty with tests ?)
        """
        pitches, offsets, durations, volumes = Midi.convert_midi_to_NNinput(name)
        next_notes = offsets[1:] + [0]  # shift offsets to convert to next_notes
        return NoteList([Note(*note) for note in zip(pitches, durations, volumes, next_notes)])

    @staticmethod
    def save(individual, name="test_song.mid"):
        """
        Save a list of Notes as a midi file

        Input: a list of Notes or a NoteList object
        """
        mid = Midi.convert_to_midi(individual)
        mid.save(name)
