"""CNN on MNIST with a proper validation split (companion to
CNN_Identify_the_Digits_(MNIST)_validation_split.ipynb).

Same architecture, optimiser, batch size (10) and epochs (5) as
CNN_Identify_the_Digits_(MNIST).ipynb, but the validation data is the LAST 10,000 of the
60,000 training images. The model trains on the first 50,000. The 10,000 test images are
used once, after training, to report the final test accuracy.

Usage (one seed per process, about 6 minutes each on an 8-core CPU):
    python scripts/mnist_validation_split.py --seed 0 --out runs_seed0.json
"""
import argparse
import json
import os
import platform
import random
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

N_VAL = 10_000          # last 10,000 of the 60,000 training images
BATCH_SIZE = 10
EPOCHS = 5


def set_seeds(seed):
    """Fix the python, numpy and TensorFlow random generators (keras.utils.set_random_seed does all three)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self._t)


def build_model(input_shape=(28, 28, 1), num_classes=10):
    # identical to the architecture in CNN_Identify_the_Digits_(MNIST).ipynb
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
    return model


def run(seed):
    t_start = time.time()
    set_seeds(seed)
    (x_all, y_all), (x_test, y_test) = mnist.load_data()
    x_all = x_all.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255
    y_all_cat = tf.keras.utils.to_categorical(y_all, 10)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

    # validation split taken from the TRAINING data only
    x_train, y_train = x_all[:-N_VAL], y_all_cat[:-N_VAL]
    x_val, y_val = x_all[-N_VAL:], y_all_cat[-N_VAL:]
    assert len(x_train) == 50_000 and len(x_val) == 10_000

    model = build_model()
    timer = EpochTimer()
    hist = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
                     validation_data=(x_val, y_val), callbacks=[timer])

    # The test set is used exactly once, after training. Loss and accuracy come from this single pass.
    probs = model.predict(x_test, verbose=0)
    test_loss = float(np.mean(keras.losses.categorical_crossentropy(y_test_cat, probs)))
    correct = int(np.sum(np.argmax(probs, axis=1) == y_test))
    h = hist.history
    return {
        "seed": seed,
        "split": "train[:50000] fit, train[50000:] validation, test untouched until the end",
        "epochs": [
            {"epoch": i + 1, "seconds": round(timer.times[i], 1),
             "train_loss": h["loss"][i], "train_acc": h["accuracy"][i],
             "val_loss": h["val_loss"][i], "val_acc": h["val_accuracy"][i]}
            for i in range(EPOCHS)
        ],
        "test_loss": test_loss,
        "test_acc": correct / len(y_test),
        "test_correct": correct,
        "wall_seconds": round(time.time() - t_start, 1),
        "fit_seconds": round(sum(timer.times), 1),
        "env": {
            "python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
            "numpy": np.__version__, "os": platform.platform(), "cpu": platform.processor(),
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="JSON file to write the result to")
    args = ap.parse_args()
    result = run(args.seed)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"seed {args.seed}: test accuracy {result['test_acc']:.4f} "
          f"(final validation accuracy {result['epochs'][-1]['val_acc']:.4f}), "
          f"wall {result['wall_seconds']} s")
