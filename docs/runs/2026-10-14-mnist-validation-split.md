# CNN on MNIST with a separate validation split, 2026-10-14

**Status: Verified (re-run 2026-10-14, seeds 0, 1, 2, validation split = last 10,000 of the 60,000 training images).**

The original notebook, `CNN_Identify_the_Digits_(MNIST).ipynb`, passes the 10,000 **test** images to `model.fit` as `validation_data`. This run trains the same network with a validation set taken from the training data instead, evaluates the test set once at the end, and compares the result with the original 0.9905 (log: [2026-10-14-cnn-mnist.md](2026-10-14-cnn-mnist.md)). The original notebook and its stored outputs were not changed.

## Settings

| Item | Original notebook | This run |
|---|---|---|
| Architecture | Conv2D 32 (3x3) - Conv2D 64 (3x3) - MaxPool 2x2 - Dropout 0.25 - Flatten - Dense 256 - Dropout 0.5 - Dense 10 softmax, 2,380,938 parameters | identical |
| Optimiser, loss | Adam (defaults), categorical cross-entropy | identical |
| Batch size, epochs | 10, 5 | identical |
| Training images | 60,000 | **50,000** (the first 50,000 of the training set; 5,000 steps per epoch, 25,000 in total) |
| `validation_data` | the 10,000 test images | **the last 10,000 training images** (a fixed block, not shuffled; label counts 991, 1064, 990, 1030, 983, 915, 967, 1090, 1009, 961 for digits 0 to 9) |
| Test set | used as validation data during training, then evaluated | not touched until after training, then **one** prediction pass gives the loss, accuracy and report |
| Seeds | none | `random.seed`, `numpy.random.seed`, `tf.random.set_seed` and `keras.utils.set_random_seed` all set to the seed; `PYTHONHASHSEED=0`. Seeds 0, 1 and 2 |
| Early stopping, checkpoints, model selection | none | none (the model evaluated is the epoch-5 model) |

Two files hold the code: `scripts/mnist_validation_split.py` (one seed per process, writes a JSON result; used for seeds 0, 1 and 2) and `CNN_Identify_the_Digits_(MNIST)_validation_split.ipynb` (single-seed demonstration, seed 0, outputs stored).

## Hardware and software

AMD Ryzen 7 5800X (8 cores, 16 threads), 32 GB RAM, Windows 11, no GPU used (TensorFlow reported no GPU devices). Python 3.11.9, TensorFlow 2.15.1, Keras 2.15.0, NumPy 1.26.4 (the earlier environment, versions as in `requirements.txt`).

## Commands

```
KERAS_HOME=<local cache> PYTHONHASHSEED=0 timeout 1200 python scripts/mnist_validation_split.py --seed 0 --out seed0.json
KERAS_HOME=<local cache> PYTHONHASHSEED=0 timeout 1200 python scripts/mnist_validation_split.py --seed 1 --out seed1.json
KERAS_HOME=<local cache> PYTHONHASHSEED=0 timeout 1200 python scripts/mnist_validation_split.py --seed 2 --out seed2.json
KERAS_HOME=<local cache> PYTHONHASHSEED=0 timeout 1200 python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=python3 CNN_Identify_the_Digits_(MNIST)_validation_split.ipynb
```

The four runs were made one after another (never at the same time) on a copy in a scratch folder. All four exited with code 0, each well inside the 20-minute limit.

## Results per seed

| Seed | Final train acc | Final validation acc | **Test accuracy** | Test loss | Test correct | Wall time | Fit time |
|---|---|---|---|---|---|---|---|
| 0 | 0.9889 | 0.9901 | **0.9907** | 0.03185 | 9,907 / 10,000 | 364 s | 360 s |
| 1 | 0.9889 | 0.9901 | **0.9890** | 0.03728 | 9,890 / 10,000 | 355 s | 352 s |
| 2 | 0.9893 | 0.9892 | **0.9887** | 0.03436 | 9,887 / 10,000 | 321 s | 318 s |
| **Mean (sample SD)** | 0.9890 (0.0002) | 0.9898 (0.0009) | **0.9895 (0.0011)** | 0.03450 | | | |

Wall time includes loading the data and the test pass. The notebook (seed 0) took 346 s from start to finish (fit 318 s) and printed test accuracy 0.9907, test loss 0.031851209700107574. These match the script's seed-0 run in every printed per-epoch number and in the test loss to full precision, so a fixed seed reproduced the result on this machine and software.

### Per-epoch training and validation accuracy

| Epoch | Seed 0 train / val | Seed 1 train / val | Seed 2 train / val |
|---|---|---|---|
| 1 | 0.9510 / 0.9863 | 0.9515 / 0.9859 | 0.9510 / 0.9860 |
| 2 | 0.9777 / 0.9878 | 0.9793 / 0.9873 | 0.9789 / 0.9884 |
| 3 | 0.9833 / 0.9897 | 0.9839 / 0.9889 | 0.9850 / 0.9903 |
| 4 | 0.9860 / 0.9898 | 0.9882 / 0.9903 | 0.9879 / 0.9902 |
| 5 | 0.9889 / 0.9901 | 0.9889 / 0.9901 | 0.9893 / 0.9892 |

Per-epoch time was 62 to 75 s (13 to 15 ms per step).

## Comparison with the original 0.9905

| | Test accuracy |
|---|---|
| Original (test set as validation data, 60,000 training images, no seed, one run) | 0.9905 (9,905 / 10,000) |
| This run, mean of seeds 0, 1, 2 (validation split, 50,000 training images) | 0.9895 (range 0.9887 to 0.9907) |
| Difference, new mean minus original | **-0.0010** (about 10 test images out of 10,000) |

What the numbers allow us to say:

- **No sign of a large inflation.** The original 0.9905 lies inside the range of the three new runs (0.9887 to 0.9907), and the mean is 0.1 percentage points lower.
- **No conclusion on a small effect.** The seed-to-seed spread is 0.1 percentage points (SD 0.0011), the same size as the difference, and a test accuracy measured on 10,000 images carries a sampling error of about 0.1 percentage points on its own. Three seeds cannot separate a difference of this size from noise.
- **The two protocols differ in more than the leak.** The new runs train on 50,000 images, the original on 60,000. A slightly lower score is what less training data alone would be expected to give, so even a real gap could not be put down to leakage. A clean comparison would be seeded runs of the original protocol (60,000 training images, test set as validation data) next to these; that was not run.
- The final validation accuracy (mean 0.9898) and the test accuracy (mean 0.9895) agree, which is what is expected when neither set was used to make decisions.

## Could anything in the original run have leaked?

Read cell by cell (cells 15 to 41 of the committed notebook):

- The test set is used in `model.fit(..., validation_data=(x_test, y_test))` only to print and plot `val_loss` and `val_accuracy` per epoch. Keras does not train on validation data, so the weights are not updated from it.
- There is no `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau` or other callback, no hyperparameter loop and no model comparison. `model.save` stores the epoch-5 model, and the final `model.evaluate` scores that same model. The last validation accuracy (0.9905) and the reported test accuracy (0.9905) are the same number because they are the same images and the same weights.
- Pixel scaling is a fixed division by 255; nothing (scaler, PCA, encoder) is fitted on the test images.
- So **within this notebook the test set influenced nothing**, and the expected effect on the reported accuracy is zero.
- What the notebook cannot show: whether the architecture, epoch count or batch size were chosen earlier by looking at test results. Cell 24 still defines `batch_size = 128` and `epochs = 10`, which the training cell does not use (it passes `epochs=5, batch_size=10`), so an earlier configuration was changed at some point, and the reason is not recorded. The earlier stored output (test accuracy 0.8815) came from a different run. If those choices were made while watching the test curves, the test set would no longer be a perfectly independent measure; that cannot be checked from the repository.
- The plot legend in the original calls the test curve "cross validation". It is the test set, not a cross-validation fold.

## What the validation split changes, and what it does not (plain language)

- **It changes who is judged by what.** The curves printed while training now describe images the network never trained on and that are not the test images. The 10,000 test images are kept sealed until the end and looked at once, so the final number is a clean exam score.
- **It changes the training data size.** The network learns from 50,000 images instead of 60,000, because 10,000 are set aside for checking. A final model for use could be retrained on all 60,000 once the settings are fixed.
- **It does not change the network, the optimiser, the batch size or the number of epochs.** It does not make the model better; if anything it can make it a hair worse because of less training data.
- **It does not by itself remove optimism if the test set is used repeatedly.** The protection only holds while the test set is evaluated once and no choices are made on it afterwards.
- **It does not make one run decisive.** Different seeds give test accuracies that differ by about 0.1 percentage points.

## Caveats

- The validation block is the last 10,000 training images, not a random or stratified draw. Its digit counts are close to uniform (915 to 1,090); another split would give slightly different numbers.
- Three seeds with one validation split give a rough spread only.
- Hardware timing is not directly comparable with the original log (708 s wall for 60,000 images at about 19 ms per step; here 13 to 15 ms per step for 50,000 images). Machine load was not controlled in either session.
- The same seed reproduced the same numbers here (notebook vs script, seed 0); results on other CPUs or TensorFlow builds may differ.
