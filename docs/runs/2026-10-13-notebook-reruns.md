# Notebook re-runs, 2026-10-13

Four notebooks were executed top to bottom after the fixes in this branch, and their outputs were kept in the committed files. Nothing here uses a GPU.

## Environment

Windows 11, CPU only, Python 3.11.9 in a fresh virtual environment. Package versions are the ones in `requirements.txt`: tensorflow 2.15.1 (Keras 2.15.0), numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, matplotlib 3.9.2, seaborn 0.13.2, nbconvert 7.16.4, ipykernel 6.29.5, nbclient 0.10.0.

## Command

Run from the repository root, once per notebook, each with a 20-minute limit:

```
timeout 1200 python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=python3 <notebook>.ipynb
```

`KERAS_HOME` pointed at a local cache and `PYTHONHASHSEED=0` was set. The notebooks have no random seeds, so the numbers below can change a little from run to run.

## Wall time and result

| Notebook | Wall time | Exit | Errors in outputs | Cells with output / code cells |
|---|---|---|---|---|
| `banknote_randomforest.ipynb` | 35 s | 0 | 0 | 3 / 11 |
| `california_regression.ipynb` | 305 s | 0 | 0 | 17 / 27 |
| `plot_kmeans_silhouette_analysis.ipynb` | 11 s | 0 | 0 | 1 / 2 (5 figures; it had no outputs before) |
| `mnist_opencv.ipynb` | 181 s | 0 | 0 | 14 / 20 |

## Numbers produced by these runs

| Notebook | Value | Notes |
|---|---|---|
| `banknote_randomforest` | Random forest accuracy **0.9912536443148688**, confusion matrix `[[184, 3], [0, 156]]`, 343 test rows | Equal to the value in the previously committed output |
| `banknote_randomforest` | Decision tree confusion matrix `[[182, 5], [3, 153]]` (accuracy 335/343 = 0.9767, derived from the matrix) | The decision tree has no random seed, so this differs between runs |
| `california_regression` | Random forest R² **0.8094088629585355**, MSE 2616.38 | The committed output was R² 0.8094, MSE 2617.17. Unseeded, so the fourth decimal moves |
| `mnist_opencv` | 1 epoch, batch size 1: final training accuracy **0.8985**, loss 0.3709. The prediction for the first test image is `[7]` | Training accuracy, not test accuracy. The committed output had 0.7887. Unseeded |

## What changed to make them run

- `california_regression`: `plot_the_model` raised `ValueError` on NumPy 1.24 or newer because it plotted 1-element arrays (`trained_bias`, `trained_weight`). Now it uses `.item()`. `RMSprop(lr=...)` is now `RMSprop(learning_rate=...)`. The data is downloaded to `data/` instead of read from `/content/`.
- `mnist_opencv`: `model.predict_classes(...)` no longer exists in current Keras. Now `np.argmax(model.predict(test), axis=-1)`. The `!pip install keras` cell became a comment pointing to `requirements.txt`.
- `banknote_randomforest`: the data is downloaded from UCI to `data/`. The UCI file has no header row, so the column names used by the notebook are given in code.
- `plot_kmeans_silhouette_analysis`: no code change; executed so it has outputs.

## Data downloaded by the notebooks (SHA-256 of the files used)

| File | Source | SHA-256 |
|---|---|---|
| `data/data_banknote_authentication.txt` (1,372 rows) | https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt | `d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9` |
| `data/california_housing_train.csv` (17,000 rows) | https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv | `bb9ca4041a66e0ed691cb825188f16fdab518435c4950a080951be2fdfe30d4d` |
| MNIST | `keras.datasets.mnist` (downloaded and cached by Keras) | not recorded |

The `data/` folder is git-ignored; the files are third-party and are not redistributed here.

## Edits made to the executed files after the run

TensorFlow's deprecation warnings print the full path of the Python environment, and one NumPy warning prints the kernel's temporary folder. Those path prefixes (a user home directory) were replaced by `...\site-packages\` and `...\Temp\` in the stored outputs of `california_regression.ipynb` and `mnist_opencv.ipynb`, so that no local path is published. Nothing else in the outputs was edited.

## Not re-run

`ANN_using_pytorch`, `ANN_using_tensorflow` (data cell updated; stored outputs unchanged), `CNN_Identify_the_Digits_(MNIST)` (about 30,000 CPU steps, over the 20-minute limit, and it needs a personal image), `Natural_Language_Processing_with_Disaster_Tweets`, `titanic-tutorial` (Kaggle data, login needed), `diabetes`, `tip_amount_tensorflow`, `world_alcohol_pandas_revision` (data not in the repository), `SigmoidNeuron` (self-contained; outputs as committed).
