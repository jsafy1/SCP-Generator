import random
import sys
import argparse
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential, save_model, load_model

try:
    from keras.layers import CuDNNLSTM
    GPU = True
except ModuleNotFoundError:
    from keras.layers import LSTM
    GPU = False

parser = argparse.ArgumentParser()
parser.add_argument("-output", dest="output", help="Location and name of output model", default="./model.h5")
parser.add_argument("-input", dest="input", help="Location of model you wish to continue training or test")
parser.add_argument("-data_dir", dest="data_dir", help="Location of training data", required=True)
parser.add_argument("-epochs", dest="epochs", help="Number of training epochs", default=20, type=int)
parser.add_argument("-batch_size", dest="batch_size", help="Batch size", default=512, type=int)
parser.add_argument("-nodes", dest="nodes", help="Number of nodes per layer", default=256, type=int)
parser.add_argument("-layers", dest="layers", help="Number of layers", default=2, type=int)
parser.add_argument("-step", dest="step", help="Number of steps to overlap", default=3, type=int)
parser.add_argument("-maxlen", dest="maxlen", help="How long each training sequence should be", default=50, type=int)
parser.add_argument("-test", dest="test", help="Do you want to test a model or train?", default=False)

args = vars(parser.parse_args())


def load_text(filepath):
    with open(filepath, encoding="utf8") as _input:
        text = _input.read()

    text = text[: len(text) // 5]  # Lower memory requirements.
    print("Length of text", len(text))
    chars = sorted(list(set(text)))
    print("Total unique chars:", len(chars))

    # We want to be able to convert chars to numbers and back.
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    return text, chars, char_indices, indices_char


def process_text(text, chars, char_indices, maxlen, step):
    # Generate overlapping sequences of length maxlen. 
    sentences = []
    next_chars = []

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    print("Sequences:", len(sentences))
    print("Vectorization...")

    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return x, y


def run_model(x, y, chars, model_path, layers, nodes, batch_size, epochs, maxlen, GPU=GPU):
    checkpoint = ModelCheckpoint(args["output"], monitor="loss", verbose=1, save_best_only=True, mode="min")
    if model_path is None:
        print("Building model")
        model = Sequential()

        for layer in range(layers - 1):
            model.add(
                CuDNNLSTM(
                    nodes,
                    input_shape=(maxlen, len(chars)),
                    return_sequences=True)
                ) if GPU else model.add(
                    LSTM(
                        nodes,
                        input_shape=(maxlen, len(chars)))
                    )
            model.add(Dropout(0.2))

        model.add(
            CuDNNLSTM(
                nodes,
                input_shape=(maxlen, len(chars)),
                return_sequences=False
            ) if GPU else model.add(
                    LSTM(
                        nodes,
                        input_shape=(maxlen, len(chars)))
                    )
        )

        model.add(Dense(len(chars), activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam")
    else:
        print("Continuing training")
        model = load_model(model_path)

    model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    return model


def sample(preds, temperature=1.0):
    # Helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def write_text(model_path, text, chars, indices_char, maxlen, length=1000, start_index=0):
    model = load_model(model_path)
    for diversity in [0.5, .75, 1.0]:
        print("----- Diversity:", diversity)
        generated = ""
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print("----- Generating with seed: " + sentence)
        sys.stdout.write(generated)

        for i in range(length):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    text, chars, char_indices, indices_char = load_text(args["data_dir"])
    if args["test"]:
        write_text(args["input"], text, chars, indices_char, args["maxlen"])
    else:
        x, y = process_text(text, chars, char_indices, args["maxlen"], args["step"])
        run_model(x, y, chars, args["input"], args["layers"], args["nodes"], args["batch_size"], args["epochs"], args["maxlen"])
