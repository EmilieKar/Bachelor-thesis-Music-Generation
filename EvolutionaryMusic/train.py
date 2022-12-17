import numpy as np
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    BatchNormalization,
    concatenate,
)
from tensorflow.keras.models import load_model, Model

NOTE_BANDWIDTH = 88


# Class for the neural network model
class NNFitnessModel:
    def __init__(self, input_shape, model=None):
        self.model = self.new(input_shape) if model is None else model

    def new(self, input_shape):
        """
        Create a new model with layers for notes, offsets, durations. Connects them and
        the network itself has an LSTM, 2 Batch Normalization and 2 Dense layers.
        Uses the 'mean squared error' loss and 'adam' as optimizer
        """

        inputNotesLayer = Input(shape=(input_shape[0], input_shape[1]))
        inputNotes = LSTM(
            16,
            input_shape=(input_shape[0], input_shape[1]),
            return_sequences=True,
        )(inputNotesLayer)

        inputOffsetsLayer = Input(shape=(input_shape[0], input_shape[1]))
        inputOffsets = LSTM(
            16,
            input_shape=(input_shape[0], input_shape[1]),
            return_sequences=True,
        )(inputOffsetsLayer)

        inputDurationsLayer = Input(shape=(input_shape[0], input_shape[1]))
        inputDurations = LSTM(
            16,
            input_shape=(input_shape[0], input_shape[1]),
            return_sequences=True,
        )(inputDurationsLayer)

        inputVolumesLayer = Input(shape=(input_shape[0], input_shape[1]))
        inputVolumes = LSTM(
            16,
            input_shape=(input_shape[0], input_shape[1]),
            return_sequences=True,
        )(inputVolumesLayer)

        inputs = concatenate([inputNotes, inputOffsets, inputDurations, inputVolumes])

        x = LSTM(100)(inputs)
        x = BatchNormalization()(x)
        x = Dense(100, activation="relu")(x)

        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)

        outputNotes = Dense(1, activation="sigmoid", name="Note")(x)

        model = Model(
            inputs=[inputNotesLayer, inputOffsetsLayer, inputDurationsLayer, inputVolumesLayer],
            outputs=[outputNotes],
        )
        model.compile(loss="mean_squared_error", optimizer="adam")

        return model

    def get_model(self):
        return self.model

    def summary(self):
        self.model.summary()

    def load(self, filepath="cache"):
        self.model = load_model(filepath)

    def fit(
        self,
        input_notes,
        input_offsets,
        input_durations,
        input_volumes,
        output,
        validation_data=None,
        filepath="cache",
        epochs=100,
        batch_size=128,
    ):
        """uses the input parameters as X and output parameters as y in model.fit"""
        history = self.model.fit(
            [input_notes, input_offsets, input_durations, input_volumes],
            output,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
        )
        self.model.save(filepath)
        return history

    def predict(self, input_notes, input_offsets, input_durations, input_volumes):
        return self.model.predict([input_notes, input_offsets, input_durations, input_volumes])


def prepare_sequences(values, n_vocab=1, sequence_length=20, type="note"):
    """
    Prepares the sequences used by the Neural Network

    NOTE: this method normalizes the input according to a note (/88) at the moment
    """

    network_input = []
    network_output = []

    # adds "padding" to arrays shorter than sequence length
    for i in range(len(values)):
        if len(values[i][0]) < sequence_length:
            values[i] = (
                values[i][0] + [0] * (sequence_length - len(values[i][0])),
                max(0, (values[i][1] - (sequence_length - len(values[i][0])) * 4)),
            )

    # create input sequences and the corresponding outputs
    for ns in values:
        for i in range(0, len(ns[0]), sequence_length):
            if i + sequence_length > len(ns[0]):
                network_input.append(ns[0][-sequence_length:])
            else:
                network_input.append(ns[0][i : i + sequence_length])
            # interactive score
            network_output.append(ns[1])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize input (is made for notes, will not work for durations / volumes / offsets)
    if type == "note":
        network_input = (
            np.clip(network_input - (128 - NOTE_BANDWIDTH) / 2, 0, NOTE_BANDWIDTH) / NOTE_BANDWIDTH
        )
    elif type == "volume":
        network_input = network_input / 128
    elif type == "offset":
        pass
    elif type == "duration":
        pass
    else:
        raise Exception("Type for prepare sequence is invalid")

    network_output = np.reshape(network_output, (len(network_output), 1)) / 100

    return (network_input, network_output)
