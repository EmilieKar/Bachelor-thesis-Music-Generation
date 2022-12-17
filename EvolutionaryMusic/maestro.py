import os
import sys
import pandas as pd
import numpy as np
import json
import time
import random

from EvolutionaryMusic.midi import Midi
from EvolutionaryMusic.train import NNFitnessModel, prepare_sequences

np.set_printoptions(threshold=sys.maxsize, formatter={"float": lambda x: "{0:0.3f}".format(x)})


def generate_csv_with_json_and_maestro(output_path="", name="Johann Sebastian Bach"):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../songs/training/data.json"))
    df = pd.DataFrame(columns=["midi_filename", "score"])

    maestro_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../maestro-v3.0.0/maestro-v3.0.0.csv")
    )
    maestro_df = pd.read_csv(maestro_path)

    data = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    index = 0
    for i, row in maestro_df.iterrows():
        if row["canonical_composer"] == name:
            key = f"maestro-v3.0.0/{row['midi_filename']}"
            df.loc[i] = [key] + [100]
            index += 1

    for i, key in enumerate(data):
        df.loc[i + index] = [key] + [data[key]]

    df = df.sample(frac=1)

    train_df = df[: int(len(df) * 0.8)]
    test_df = df[int(len(df) * 0.8) :]

    train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)


def generate_csv(name="Johann Sebastian Bach", frac_set=0.1, rand_max=60, output_path=""):
    """
    Generate a csv file similar to that found in the maestro set, but with an additional column (score)
    Used for test purposes, setting scores to certain melodies to a value. Can be modified and
    used for other purposes
    """
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../maestro-v3.0.0/maestro-v3.0.0.csv")
    )

    train_df = pd.read_csv(path)
    composers = dict(map(lambda x: (x, 0), train_df["canonical_composer"].unique()))

    for key in composers:
        composers[key] = random.randint(0, rand_max)

    train_df["score"] = (np.random.random(train_df.shape[0]) * rand_max).astype(int)
    test_df = train_df.copy()

    train_df = train_df[train_df.split == "train"]
    test_df = test_df[test_df.split != "train"]

    train_df = train_df.sample(frac=frac_set)
    test_df = test_df.sample(frac=frac_set)

    for key in composers:
        train_df.loc[train_df["canonical_composer"] == key, "score"] = composers[key]
        test_df.loc[test_df["canonical_composer"] == key, "score"] = composers[key]

    train_df.loc[train_df["canonical_composer"] == name, "score"] = 100
    test_df.loc[test_df["canonical_composer"] == name, "score"] = 100

    train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)


def train_maestro(model, path, sequence_length, epochs=100, output_path="", use_json=False):
    """
    Trains a model according to the data specified in path.
    Has 4 arrays for the parameters in a song, and extracts information from a file path.
    Uses model.fit() to train
    """
    df = pd.read_csv(os.path.join(output_path, path))

    total_notes = []
    total_offsets = []
    total_durations = []
    total_volumes = []

    for index, row in df.iterrows():
        p = row["midi_filename"] if use_json else "maestro-v3.0.0/" + row["midi_filename"]
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../{p}"))

        try:
            notes, offsets, durations, volumes = Midi.convert_midi_to_NNinput(path)
        except Exception:
            continue
        total_notes.append((notes, df["score"][index]))
        total_offsets.append((offsets, df["score"][index]))
        total_durations.append((durations, df["score"][index]))
        total_volumes.append((volumes, df["score"][index]))

    input_notes, output = prepare_sequences(
        total_notes, sequence_length=sequence_length, type="note"
    )
    input_offsets, _ = prepare_sequences(
        total_offsets, sequence_length=sequence_length, type="offset"
    )
    input_durations, _ = prepare_sequences(
        total_durations, sequence_length=sequence_length, type="duration"
    )
    input_volumes, _ = prepare_sequences(
        total_volumes, sequence_length=sequence_length, type="volume"
    )

    # return
    model.fit(
        input_notes,
        input_offsets,
        input_durations,
        input_volumes,
        output,
        epochs=epochs,
        filepath=output_path,
    )


def predict_maestro(
    model, path, sequence_length, output_path="", threshold=0.5, use_baseline=None, use_json=False
):
    """
    Predicts a known paths's fitness scores. Has been used for test purposes, can be used
    in combination with other methods
    """
    model.load(filepath=output_path)

    df = pd.read_csv(os.path.join(output_path, path))

    predictions = []

    for index, row in df.iterrows():
        p = row["midi_filename"] if use_json else "maestro-v3.0.0/" + row["midi_filename"]
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../{p}"))

        try:
            notes, offsets, durations, volumes = Midi.convert_midi_to_NNinput(path)
        except Exception:
            continue

        input_notes, output = prepare_sequences(
            [(notes, df["score"][index])], sequence_length=sequence_length, type="note"
        )
        input_offsets, _ = prepare_sequences(
            [(offsets, df["score"][index])], sequence_length=sequence_length, type="offset"
        )
        input_durations, _ = prepare_sequences(
            [(durations, df["score"][index])], sequence_length=sequence_length, type="duration"
        )
        input_volumes, _ = prepare_sequences(
            [(volumes, df["score"][index])], sequence_length=sequence_length, type="volume"
        )

        prediction = model.predict(input_notes, input_offsets, input_durations, input_volumes)
        predictions.append((df["score"][index] / 100, prediction.mean()))

    print(np.array(predictions))
    scores = 0
    for o, p in predictions:
        pred = p if use_baseline is None else use_baseline
        scores = scores + 1 if abs(o - pred) < threshold else scores

    score_ones, total_ones = 0, 0
    for o, p in predictions:
        if not o == 1.0:
            continue
        score_ones = score_ones + 1 if abs(o - p) < threshold else score_ones
        total_ones += 1
    print("Accuracy: ", (scores / len(predictions)) * 100, "%")
    print("Accuracy only ones: ", (score_ones / total_ones) * 100, "%")


def evaluate_maestro(
    model, train_path, test_path, sequence_length, epochs=100, output_path="", use_json=False
):
    """
    Is the same as the combination of train and predict
    except that this evaluates the model after each epoch
    and saves the resulting loss to {output_path}/history.json
    """
    start_preprocess = time.time()

    train_df = pd.read_csv(os.path.join(output_path, train_path))
    test_df = pd.read_csv(os.path.join(output_path, test_path))

    total_train_notes = []
    total_train_offsets = []
    total_train_durations = []
    total_train_volumes = []

    total_test_notes = []
    total_test_offsets = []
    total_test_durations = []
    total_test_volumes = []

    for index, row in train_df.iterrows():
        p = row["midi_filename"] if use_json else "maestro-v3.0.0/" + row["midi_filename"]
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../{p}"))

        try:
            notes, offsets, durations, volumes = Midi.convert_midi_to_NNinput(path)
        except Exception:
            continue
        total_train_notes.append((notes, train_df["score"][index]))
        total_train_offsets.append((offsets, train_df["score"][index]))
        total_train_durations.append((durations, train_df["score"][index]))
        total_train_volumes.append((volumes, train_df["score"][index]))

    input_notes, output = prepare_sequences(
        total_train_notes, sequence_length=sequence_length, type="note"
    )
    input_offsets, _ = prepare_sequences(
        total_train_offsets, sequence_length=sequence_length, type="offset"
    )
    input_durations, _ = prepare_sequences(
        total_train_durations, sequence_length=sequence_length, type="duration"
    )
    input_volumes, _ = prepare_sequences(
        total_train_volumes, sequence_length=sequence_length, type="volume"
    )

    train_input = [input_notes, input_offsets, input_durations, input_volumes]
    train_output = output

    for index, row in test_df.iterrows():
        p = row["midi_filename"] if use_json else "maestro-v3.0.0/" + row["midi_filename"]
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../{p}"))

        try:
            notes, offsets, durations, volumes = Midi.convert_midi_to_NNinput(path)
        except Exception:
            continue
        total_test_notes.append((notes, test_df["score"][index]))
        total_test_offsets.append((offsets, test_df["score"][index]))
        total_test_durations.append((durations, test_df["score"][index]))
        total_test_volumes.append((volumes, test_df["score"][index]))

    input_notes, output = prepare_sequences(
        total_test_notes, sequence_length=sequence_length, type="note"
    )
    input_offsets, _ = prepare_sequences(
        total_test_offsets, sequence_length=sequence_length, type="offset"
    )
    input_durations, _ = prepare_sequences(
        total_test_durations, sequence_length=sequence_length, type="duration"
    )
    input_volumes, _ = prepare_sequences(
        total_test_volumes, sequence_length=sequence_length, type="volume"
    )

    test_input = [input_notes, input_offsets, input_durations, input_volumes]
    test_output = output

    print(f"Training shape: {train_output.shape}\tTest shape: {test_output.shape}")
    start_training = time.time()
    # return
    history = model.fit(
        *train_input,
        train_output,
        validation_data=(test_input, test_output),
        filepath=output_path,
        epochs=epochs,
    )

    end_training = time.time()

    history.history["training_data_points"] = str(train_output.shape)
    history.history["test_data_points"] = str(test_output.shape)
    history.history["training_time"] = end_training - start_training
    history.history["preprocess_time"] = start_training - start_preprocess
    history.history["epochs"] = epochs

    with open(os.path.join(output_path, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=4)


def output_directory(number=None):
    """
    Generates a new folder at {root/models}/{len(directories in that folder)}
    and return it if number is not specified else return the absolute path of
    the specified number
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/"))

    if number is not None:
        path = os.path.join(path, str(number))
    else:
        path = os.path.join(path, str(len(os.listdir(path)) + 1))
        if not os.path.exists(path):
            os.makedirs(path)

    return path


if __name__ == "__main__":
    output_path = output_directory(number=None)
    generate_csv_with_json_and_maestro(output_path=output_path)

    # generate path.csv
    # generate_csv(rand_max=60, frac_set=0.1, output_path=output_path)

    sequence_length = 20

    model = NNFitnessModel((sequence_length, 1))

    # train_maestro(model, "train.csv", sequence_length, epochs=10)

    # model.summary()

    # predict_maestro(
    #     model,
    #     "test.csv",
    #     sequence_length,
    #     output_path=output_path,
    #     threshold=0.2,
    #     use_baseline=None,
    #     use_json=True,
    # )

    evaluate_maestro(
        model,
        "train.csv",
        "test.csv",
        sequence_length,
        epochs=100,
        output_path=output_path,
        use_json=True,
    )

    # import tensorflow

    # tensorflow.keras.utils.plot_model(model.get_model(), show_shapes=True)
