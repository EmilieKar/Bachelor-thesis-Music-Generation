import os
import json
from datetime import datetime

from EvolutionaryMusic.midi import Midi
from EvolutionaryMusic.train import NNFitnessModel, prepare_sequences


# Wants the music to only contain the note C
# Note: Music rep must have a list of notes
def calculate_fitness_only_C(individual):
    score = 0.0
    notes = [note for note in individual.get_notes() if note.pitch is not None]
    for note in notes:
        if note.pitch % 12 == 0:
            score += 1
    individual.fitness_score = int((score / len(individual)) * 10)
    # gives a rating between 1 and 10 just for debugging.
    # prob not the most efficient way to grade but floats are hard to look at


def calculate_set_note_fitness(individual):
    score = 0.0
    notes = [note for note in individual.get_notes() if note.pitch is not None]
    for note in notes:
        if note.pitch % 12 == 0:
            score += 1 / 3
        if note.duration == 0.5:
            score += 1 / 3
        if note.volume == 88:
            score += 1 / 3
    individual.fitness_score = score / max(1, len(individual)) * 10
    # gives a rating between 1 and 10 just for debugging.
    # prob not the most efficient way to grade but floats are hard to look at


class Fitness:
    def __init__(
        self,
        population_size,
        sequence_length,
        epochs,
        training_cycles,
        save_interactive_samples,
        train_existing_songs,
        load_neural_model,
        subrater_path,
    ):
        self.subrater = FitnessSubrater(subrater_path)
        self.neural = FitnessNeural(
            population_size,
            sequence_length,
            epochs,
            [],
            save_interactive_samples,
            train_existing_songs,
            load_neural_model,
            True,
        )
        self.fitness = self.subrater
        self.population_size = population_size
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.training_cycles = training_cycles
        self.current_individual = 0
        self.generations = 0

        # change the fitness function depending on training_cycles
        # a0:s3:i3 would change the fitness function to automatic (a0)
        if len(self.training_cycles) > 0:
            if self.parse_training_cycle(self.training_cycles[0], self.generations):
                self.training_cycles.pop(0)

    def parse_training_cycle(self, cycle, generation):
        value = int(cycle[1:])
        if value != generation:
            return False

        if cycle[0] == "s":
            print("Changing fitness to <subrater>")
            self.fitness = self.subrater
        elif cycle[0] == "i":
            print("Changing fitness to <interactive>")
            self.fitness = self.neural.to_interactive()
        elif cycle[0] == "a":
            print("Changing fitness to <automatic>")
            self.fitness = self.neural.to_automatic()

        return True

    def __call__(self, individual):
        self.fitness(individual)

        self.current_individual += 1
        if self.current_individual >= self.population_size:
            self.current_individual = 0

            if len(self.training_cycles) > 0:
                if self.parse_training_cycle(self.training_cycles[0], self.generations):
                    self.training_cycles.pop(0)
                    self.generations = -1

            self.generations += 1


class FitnessSubrater:
    """
    Contains calibration data which is used later in the fitness function.
    """

    subraters = {
        "neighbouring_pitch_range": 0,
        "direction_of_melody": 0,
        "stability_of_melody": 0,
        "pitch_range": 0,
        "unique_notes": 0,
        "equal_consecutive_notes": 0,
        "unique_rhythms": 0,
        "total_pitches": 0,
        "duration_of_melody": 0,
        "number_of_chords": 0,
        "limit_chord_length": 0,
        "stability_of_velocity": 0,
        "note_density": 0,
    }

    def __init__(self, filepath):
        """
        If there is no input files (names), the names list will be
        built of files in the /songs/training directory.
        """
        subrater_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "cache/subrater.json")
        )

        if filepath is None:
            # try to load subrater cache
            try:
                with open(subrater_path) as f:
                    self.subraters = json.load(f)
                    print("Loaded subrater cache")
                    return
            except Exception:
                print(
                    "\nERROR: You have to specify a 'subrater_path=...' "
                    "if 'cache/subrater.json' does not exist"
                )
                exit()
        else:
            names = [
                os.path.join(filepath, f)
                for f in os.listdir(filepath)
                if os.path.isfile(os.path.join(filepath, f))
            ]

        for i, name in enumerate(names):
            print(f"[{i+1}/{len(names)}] {name}")
            individual = Midi.convert_midi_to_list(name)

            note_list = individual.get_notes()

            self.subraters["neighbouring_pitch_range"] += neighbouring_pitch_range(note_list)
            self.subraters["direction_of_melody"] += direction_of_melody(note_list)
            self.subraters["stability_of_melody"] += stability_of_melody(note_list)
            self.subraters["pitch_range"] += pitch_range(note_list)
            self.subraters["unique_notes"] += unique_notes(note_list)
            self.subraters["equal_consecutive_notes"] += equal_consecutive_notes(note_list)
            self.subraters["unique_rhythms"] += unique_rhythms(note_list)
            self.subraters["total_pitches"] += total_pitches(note_list)
            self.subraters["duration_of_melody"] += duration_of_melody(note_list)
            self.subraters["number_of_chords"] += number_of_chords(note_list)
            self.subraters["limit_chord_length"] += limit_chord_length(note_list)
            self.subraters["stability_of_velocity"] += stability_of_velocity(note_list)
            self.subraters["note_density"] += note_density(note_list)

        if len(names) == 0:
            print("\nERROR: Your path contains no songs")
            exit()

        for key in self.subraters:
            self.subraters[key] /= len(names)

        # save calibration to cache
        cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        with open(subrater_path, "w", encoding="utf-8") as f:
            json.dump(self.subraters, f, ensure_ascii=False, indent=4)

    def __call__(self, individual):
        self.fitness(individual)

    def fitness(self, individual):
        """
        Will mutate the fitness_score of the given individual.

        Will compare the score between the calibration and the individual.
        If the individual has closer values to that of the calibration a
        greater fitness_score will be acquired.
        """

        note_list = individual.get_notes()

        fitness = 0
        fitness += abs(
            neighbouring_pitch_range(note_list) - self.subraters["neighbouring_pitch_range"]
        )
        fitness += abs(direction_of_melody(note_list) - self.subraters["direction_of_melody"])
        fitness += abs(stability_of_melody(note_list) - self.subraters["stability_of_melody"])
        fitness += abs(pitch_range(note_list) - self.subraters["pitch_range"])
        fitness += abs(unique_notes(note_list) - self.subraters["unique_notes"])
        fitness += abs(
            equal_consecutive_notes(note_list) - self.subraters["equal_consecutive_notes"]
        )
        fitness += abs(unique_rhythms(note_list) - self.subraters["unique_rhythms"])

        max_length = max(total_pitches(note_list), self.subraters["total_pitches"])
        fitness += (
            0
            if max_length == 0
            else abs(total_pitches(note_list) - self.subraters["total_pitches"]) / max_length
        )

        max_length = max(duration_of_melody(note_list), self.subraters["duration_of_melody"])
        fitness += (
            0
            if max_length == 0
            else abs(duration_of_melody(note_list) - self.subraters["duration_of_melody"])
            / max_length
        )

        fitness += abs(limit_chord_length(note_list) - self.subraters["limit_chord_length"])

        fitness += abs(number_of_chords(note_list) - self.subraters["number_of_chords"])

        fitness += abs(limit_chord_length(note_list) - self.subraters["limit_chord_length"])

        fitness += abs(number_of_chords(note_list) - self.subraters["number_of_chords"])

        fitness += abs(stability_of_velocity(note_list) - self.subraters["stability_of_velocity"])

        fitness += abs(note_density(note_list) - self.subraters["note_density"])

        individual.fitness_score = 1.0 - fitness / len(self.subraters)


def neighbouring_pitch_range(individual) -> float:
    """
    Returns the ratio between the number of times neighbouring pitch values are greater
    than 24 (2 octaves), and the total number of notes.
    """
    individual = [note.pitch for note in individual if note.pitch is not None]

    crazy_notes = 0
    for i in range(len(individual) - 1):
        if abs(individual[i] - individual[i + 1]) > 24:
            crazy_notes += 1

    return crazy_notes / max(1, len(individual))


def direction_of_melody(individual) -> float:
    """
    Returns the number of notes in the same direction, incremental or decremental flow,
    and the total number of notes.
    """

    individual = [note.pitch for note in individual if note.pitch is not None]

    required_notes = 2
    notes_in_direction = 0
    cur_dir = 1
    prev_dir = 1
    current_notes_in_direction = 0
    for i in range(len(individual) - 1):
        if individual[i] == 0:
            continue
        if individual[i] > individual[i + 1]:
            cur_dir = -1
        elif individual[i] < individual[i + 1]:
            cur_dir = 1

        if cur_dir == prev_dir:
            current_notes_in_direction += 1
        else:
            current_notes_in_direction = 0

        if current_notes_in_direction >= required_notes:
            notes_in_direction += 1

        prev_dir = cur_dir
    return notes_in_direction / max(1, len(individual))


def stability_of_melody(individual) -> float:
    """
    Returns the number of changes in direction, from incremental to decremental or vice versa,
    and the total number of notes.
    """
    individual = [note.pitch for note in individual if note.pitch is not None]

    notes_in_direction = 0
    direction = 1
    for i in range(len(individual) - 1):
        if individual[i] == 0:
            continue
        if individual[i] > individual[i + 1]:
            if direction == 1:
                notes_in_direction += 1
            direction = -1
        elif individual[i] < individual[i + 1]:
            if direction == -1:
                notes_in_direction += 1
            direction = 1

    return notes_in_direction / max(1, len(individual))


def pitch_range(individual) -> float:
    """
    Returns the ratio between the minimum and maximum pitch value.
    """

    individual = [note.pitch for note in individual if note.pitch is not None]

    if len(individual) == 0:
        return 0
    min_value = min(list(map(lambda note: note, individual)))
    max_value = max(list(map(lambda note: note, individual)))

    return min_value / max_value


def unique_notes(individual) -> float:
    """
    Returns the ratio between the number of unique notes and the total number of notes.
    """
    individual = [note.pitch for note in individual if note.pitch is not None]

    seen_notes = []

    for note in individual:
        if note not in seen_notes:
            seen_notes.append(note)

    return len(seen_notes) / max(1, len(individual))


def equal_consecutive_notes(individual) -> float:
    """
    Returns the ratio between the number of equal consective pitch values
    and the total number of notes.
    """
    individual = [note.pitch for note in individual if note.pitch is not None]

    equal_notes = 0

    for i in range(len(individual) - 1):
        if individual[i] == individual[i + 1]:
            equal_notes += 1

    return equal_notes / max(1, len(individual))


def unique_rhythms(individual) -> float:
    """
    Return the ratio between the number of unique durations and the total number of notes.
    """
    individual = [note.duration for note in individual]

    seen_rhythms = []

    for duration in individual:
        if duration not in seen_rhythms:
            seen_rhythms.append(duration)

    return len(seen_rhythms) / max(1, len(individual))


def total_pitches(individual) -> float:
    """
    Return the length of a melody
    """

    return len(individual)


def duration_of_melody(individual) -> float:
    """
    Return total duration of melody
    Does not take the duration of the last note (or notes) into account
    """
    individual = [note.next_note for note in individual]

    duration = 0
    for offset in individual:
        duration += offset

    return duration


def limit_chord_length(individual) -> float:
    """
    Return penalty if chord length is too long (5 notes or longer)
    """

    individual = [note.next_note for note in individual]

    limit = 5
    counter = 0
    total_chords = 0
    penalty = 0

    for offset in individual:
        if offset == 0:
            counter += 1
        else:
            # check for limit
            if counter + 1 >= limit:
                penalty += 1
            counter = 0
            total_chords += 1

    if individual[-1] == 0:
        if counter + 1 >= limit:
            penalty += 1
        total_chords += 1

    if total_chords == 0:
        return 0

    return penalty / total_chords


def number_of_chords(individual) -> float:
    """
    Returns the ratio between "events" consisting of two or more notes
    and the total number of "events".
    """

    if len(individual) <= 1:
        return 0

    offsets = [note.next_note for note in individual]
    total_events = 0
    chord_events = 0
    i = 0
    while i < len(offsets) - 1:
        if offsets[i] == 0:
            total_events += 1
            chord_events += 1
            i += 1
            while (i < len(offsets) - 1) and (offsets[i] == 0):
                i += 1
        total_events += 1
        i += 1

    return chord_events / total_events


def stability_of_velocity(individual) -> float:
    """
    Returns some measurement of how much the velocity changes.
    """

    if len(individual) == 0:
        return 0

    total_change = 0

    for i in range(len(individual) - 1):
        total_change += abs(individual[i].volume - individual[i + 1].volume)

    # what this means:
    # total_change / (highest possible change between two notes * number of notes)
    return total_change / (127 * len(individual))


def note_density(individual) -> float:
    return len(individual) / duration_of_melody(individual)


class FitnessNeural:
    def __init__(
        self,
        population_size,
        sequence_length,
        epochs,
        training_cycles,
        save_interactive_samples,
        train_existing_songs,
        load_neural_model,
        is_combination,
    ):
        self.total_notes = []
        self.total_offsets = []
        self.total_durations = []
        self.total_volumes = []

        self.midi_player = Midi()
        self.population_size = population_size
        self.current_individual = 0
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.generations = 0
        self.is_combination = is_combination
        self.save_interactive_samples = save_interactive_samples
        if self.save_interactive_samples:
            now = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
            self.training_samples_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), f"../songs/training/{now}")
            )

            if not os.path.exists(self.training_samples_path):
                os.makedirs(self.training_samples_path)

        self.training_cycles = [int(g[1:]) for g in training_cycles if g[0] == "i"]

        if len(self.training_cycles) > 0:
            self.fitness = (
                self.interactive_fitness if self.training_cycles[0] == 0 else self.automatic_fitness
            )
            if self.training_cycles[0] == 0:
                self.training_cycles.pop(0)
        else:
            self.fitness = self.automatic_fitness

        # create neural network model
        self.model = NNFitnessModel((sequence_length, 1))

        if load_neural_model:
            print("Loading saved neural model")
            self.model.load()

        # train the model with existing data if train_existing_songs is set
        if train_existing_songs:
            print(
                "Training model with songs from",
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../songs/training/")),
            )
            self.train_from_directory()

    def __call__(self, individual):
        return self.fitness(individual)

    def to_interactive(self):
        self.fitness = self.interactive_fitness
        return self

    def to_automatic(self):
        self.fitness = self.automatic_fitness
        return self

    # Interactive fitness
    def interactive_fitness(self, individual):
        notes, offsets, durations, volumes = Midi.convert_midi_to_NNinput(
            Midi.convert_to_midi(individual)
        )

        print(notes, "\n", offsets, "\n", durations, "\n", volumes)

        fitness_value = self.midi_player.play(individual, msg="Rate 0-100\n")

        try:
            score = int(fitness_value)
        except ValueError:
            score = 0
        score = max(min(score, 100), 0)

        # if flag save_interactive_samples=True then we want to save
        # every scored midi song to the training directory for later use
        if self.save_interactive_samples:
            filepath = f"{self.training_samples_path}/{self.current_individual+1}.mid"
            Midi.save(individual, filepath)
            neural_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../songs/training/data.json")
            )
            data = {}
            if os.path.isfile(neural_path):
                with open(neural_path, encoding="utf-8") as f:
                    data = json.load(f)

            data.update({filepath: score})

            with open(neural_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        self.total_notes.append((notes, score))
        self.total_offsets.append((offsets, score))
        self.total_durations.append((durations, score))
        self.total_volumes.append((volumes, score))

        individual.fitness_score = score

        self.current_individual += 1
        if self.current_individual >= self.population_size:
            if self.save_interactive_samples:
                now = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
                self.training_samples_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), f"../songs/training/{now}")
                )

                if not os.path.exists(self.training_samples_path):
                    os.makedirs(self.training_samples_path)

            if len(self.training_cycles) == 0 or self.training_cycles[0] != 0:
                if not self.is_combination:
                    print("Changing fitness to <automatic>")
                    self.fitness = self.automatic_fitness

            self.current_individual = 0
            self.generations = 0
            self.fit()

    def automatic_fitness(self, individual):
        notes, offsets, durations, volumes = Midi.convert_midi_to_NNinput(
            Midi.convert_to_midi(individual)
        )

        self.total_notes.append((notes, 0))
        self.total_offsets.append((offsets, 0))
        self.total_durations.append((durations, 0))
        self.total_volumes.append((volumes, 0))

        score = self.predict()
        individual.fitness_score = int(score.mean() * 100)

        self.current_individual += 1
        if self.current_individual >= self.population_size:
            self.current_individual = 0
            self.generations += 1

            if len(self.training_cycles) > 0 and self.generations == self.training_cycles[0]:
                if not self.is_combination:
                    print("Changing fitness to <interactive>")
                    self.fitness = self.interactive_fitness
                    self.training_cycles.pop(0)

    def train_from_directory(self):
        self.total_notes = []
        self.total_offsets = []
        self.total_durations = []
        self.total_volumes = []

        neural_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../songs/training/data.json")
        )
        data = {}
        if os.path.isfile(neural_path):
            with open(neural_path, encoding="utf-8") as f:
                data = json.load(f)

        for key in data:
            print(key, data[key])

            notes, offsets, durations, volumes = Midi.convert_midi_to_NNinput(key)

            self.total_notes.append((notes, data[key]))
            self.total_offsets.append((offsets, data[key]))
            self.total_durations.append((durations, data[key]))
            self.total_volumes.append((volumes, data[key]))

        self.fit()

    def fit(self):
        input_notes, output = prepare_sequences(
            self.total_notes, sequence_length=self.sequence_length, type="note"
        )
        input_offsets, _ = prepare_sequences(
            self.total_offsets, sequence_length=self.sequence_length, type="offset"
        )
        input_durations, _ = prepare_sequences(
            self.total_durations, sequence_length=self.sequence_length, type="duration"
        )
        input_volumes, _ = prepare_sequences(
            self.total_volumes, sequence_length=self.sequence_length, type="volume"
        )

        self.total_notes = []
        self.total_offsets = []
        self.total_durations = []
        self.total_volumes = []
        self.model.fit(
            input_notes,
            input_offsets,
            input_durations,
            input_volumes,
            output,
            epochs=self.epochs,
        )

    def predict(self):
        input_notes, _ = prepare_sequences(
            self.total_notes, sequence_length=self.sequence_length, type="note"
        )
        input_offsets, _ = prepare_sequences(
            self.total_offsets, sequence_length=self.sequence_length, type="offset"
        )
        input_durations, _ = prepare_sequences(
            self.total_durations, sequence_length=self.sequence_length, type="duration"
        )
        input_volumes, _ = prepare_sequences(
            self.total_volumes, sequence_length=self.sequence_length, type="volume"
        )

        self.total_notes = []
        self.total_offsets = []
        self.total_durations = []
        self.total_volumes = []

        return self.model.predict(input_notes, input_offsets, input_durations, input_volumes)
