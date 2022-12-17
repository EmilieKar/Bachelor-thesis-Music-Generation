import random as rnd


class Note:
    def __init__(self, pitch=60, duration=1.0, volume=100, next_note=None):
        self.pitch = pitch
        self.duration = duration
        self.volume = volume

        # Duration until next note is played. If it is 0 then notes are played simultaneously,
        # if it is bigger than 0 than next note starts after this note.
        # if it is bigger than duration there is a pause between the notes.
        if next_note is None:
            # Default is to play note immediately after without pause
            self.next_note = self.duration
        else:
            self.next_note = next_note

    def __repr__(self):
        return (
            f"{self.pitch_to_str(): <5}{round(self.next_note, 2): <5}"
            f"{round(self.duration, 2): <4}{self.volume: >4}"
        )

    def __eq__(self, note):
        return (
            self.pitch == note.pitch
            and self.duration == note.duration
            and self.volume == note.volume
            # Next note distance doesn't determine wether a note is equal to another
        )

    def pitch_to_str(self, useSharps=True):
        """Return the name of an integer pitch. Middle C = 'C4' = pitch 60. """
        if self.pitch is None:
            return "pause"
        notes = [
            "C",
            "Db/C#",
            "D",
            "Eb/D#",
            "E",
            "F",
            "Gb/F#",
            "G",
            "Ab/G#",
            "A",
            "Bb/A#",
            "B",
        ]
        name = notes[self.pitch % 12].split("/")[-1 * useSharps]
        octave = self.pitch // 12 - 1
        return f"{name}{octave}"

    @staticmethod
    def generate_random_note(mu=64, sigma=10):
        """
        Generate a note with normal distributed pitch

        Parameters:
        mu - mean pitch
        sigma - standard deviation of pitch

        volume and next note have default values
        """
        pitch = max(0, min(127, int(rnd.gauss(mu, sigma))))
        duration = max(4 / 128, min(16, 2 ** int(rnd.gauss(0, 1))))
        return Note(pitch, duration)

    def copy(self):
        return Note(self.pitch, self.duration, self.volume, self.next_note)
