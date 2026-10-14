# Machine Learning Course Notebooks: teaching notebooks from linear regression to CNNs

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#notebooks)

## Overview

Thirteen Jupyter notebooks that walk through classic machine-learning topics: random forests and decision trees, regression on the California housing data, neural networks in TensorFlow and PyTorch, a CNN on MNIST, k-means with silhouette analysis, text classification and pandas basics. Most notebooks keep the outputs they produced, so students can read them without running anything. Six were re-run on 2026-10-13 and 2026-10-14 (four of them fixed to download their own data), including the CNN on MNIST. The other seven were re-run on 2026-10-15: two use data they already downloaded, and five whose original data files are not available now download a public substitute dataset instead, so their results differ from the stored outputs they had before (each says so in a note at the top).

These notebooks are used in the machine learning lab sessions I teach.

## Demo

No hosted demo. Open any notebook in Google Colab with the badges in the [Notebooks table](#notebooks).

## Problem statement

Students learning ML need small, runnable examples on public data, each showing one idea end to end. These notebooks are that material. Before these updates most of them could not run from a clean copy: they read data from Colab or Kaggle paths that were not in the repository, and several broke on current library versions.

## Approach

Each notebook is self-contained and follows the usual flow: load data, explore it, fit a model, evaluate it. All thirteen notebooks have been re-run and fetch their data in their first cells (a download from the dataset's public source, or a Keras loader), so they work in a clean checkout and in Colab. Where the original data file was lost, the notebook downloads a public substitute and says so in its first cell.

```mermaid
flowchart LR
  A[Download or load public dataset] --> B[Explore and prepare]
  B --> C[Fit model: forest, regression, CNN, k-means]
  C --> D[Evaluate and plot]
```

## Data

No data is stored in this repository. Public datasets are downloaded by the notebooks into a git-ignored `data/` folder; the sources and licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

| Dataset | Rows | Source | License / terms |
|---|---|---|---|
| Banknote Authentication (Lohweg, 2012) | 1,372 | UCI Machine Learning Repository | CC BY 4.0 |
| California housing (training set) | 17,000 | Google Machine Learning Crash Course | Per the course page |
| MNIST | 60,000 train / 10,000 test | Keras loader | Per the MNIST authors |
| Pima Indians Diabetes Database (substitute for the lost `diabetes.csv`) | 768 | OpenML data set 37 (copy of the UCI data); falls back to the identical table in `plotly/datasets` | OpenML licence field "Public"; MIT for the `plotly/datasets` repository |
| Titanic passenger list (substitute for the Kaggle files) | 1,309 (split 891 / 418) | OpenML data set 40945 (Harrell and Cason, Vanderbilt Biostatistics) | OpenML licence field "Public" |
| SMS Spam Collection (substitute for the Kaggle disaster tweets) | 5,574 (split 4,459 / 1,115) | UCI Machine Learning Repository | CC BY 4.0 |
| Restaurant tips (substitute for the lost `tip-amount.xlsx`) | 244 | `plotly/datasets` GitHub repository | MIT for the repository; no separate licence stated for the table |
| WHO alcohol consumption by beverage type (substitute for the lost `world_alcohol.csv`) | 41,521 | WHO Global Health Observatory OData API, indicator SA_0000001400 | CC BY 4.0 (WHO data terms) |

The original files behind the last five notebooks (`diabetes.csv`, the Kaggle Titanic and Disaster Tweets files, `tip-amount.xlsx`, `world_alcohol.csv`) are not in this repository; the notebooks no longer use them. The source of `tip-amount.xlsx` and `world_alcohol.csv` was not recorded. Each of those five notebooks has a note at the top that names the substitute, so results differ from the earlier stored outputs.

## Notebooks

| Notebook | Topic | Dataset | Outputs (cells with output / code cells) | Open in Colab | Status |
|---|---|---|---|---|---|
| `ANN_using_pytorch.ipynb` | Feed-forward neural network in PyTorch (banknote forgery detection) | Banknote Authentication (UCI), downloaded by the notebook | yes (14/23), no errors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/ANN_using_pytorch.ipynb) | Verified (re-run 2026-10-15, UCI banknote data): 17 s on CPU with PyTorch 2.5.1, which is now pinned in requirements.txt |
| `ANN_using_tensorflow.ipynb` | Neural network in TensorFlow/Keras, with a network diagram (drawn with matplotlib) | Banknote Authentication (UCI), downloaded by the notebook | yes (61/78), no errors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/ANN_using_tensorflow.ipynb) | Verified (re-run 2026-10-15, UCI banknote data): 26 s. Fixed an empty `Dense()`, a missing `Input` import and undefined `y_pred2` / `y_class2`; `ann_visualizer` (needs Graphviz) replaced by a matplotlib function |
| `CNN_Identify_the_Digits_(MNIST).ipynb` | CNN digit classifier | MNIST (Keras loader); the optional personal-image cells (2.jpeg) are skipped when the file is missing | yes (35/41), no errors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/CNN_Identify_the_Digits_%28MNIST%29.ipynb) | Verified (re-run 2026-10-14, full settings: batch size 10, 5 epochs, CPU, 12 min): test accuracy 0.9905. Three cells adjusted so it runs outside Colab |
| `CNN_Identify_the_Digits_(MNIST)_validation_split.ipynb` | Companion of the CNN above with a held-out validation split and fixed seeds | MNIST (Keras loader) | yes (10/10), no errors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/CNN_Identify_the_Digits_%28MNIST%29_validation_split.ipynb) | Verified (re-run 2026-10-14, seed 0; seeds 0, 1, 2 in the log): see [the section below](#cnn-on-mnist-with-a-separate-validation-split) |
| `Natural_Language_Processing_with_Disaster_Tweets.ipynb` | **Uses the SMS Spam Collection; file name kept for link stability.** Text preprocessing, word2vec and Naive Bayes on short texts (now SMS spam, formerly tweets) | SMS Spam Collection (UCI, CC BY 4.0), downloaded by the notebook; substitute for the Kaggle "Disaster Tweets" data, which is not available here | yes (39/52), no errors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/Natural_Language_Processing_with_Disaster_Tweets.ipynb) | Verified (re-run 2026-10-15, SMS Spam Collection substitute data): 11 s on gensim 4. Results differ from the earlier tweet outputs |
| `SigmoidNeuron.ipynb` | Sigmoid neuron from scratch | Synthetic, generated in the notebook | yes (13/26) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/SigmoidNeuron.ipynb) | Verified (re-run 2026-10-14, 56 s): runs clean after replacing `tqdm_notebook` with `tqdm` |
| `banknote_randomforest.ipynb` | Random forest vs decision tree on banknote data | Banknote Authentication (UCI), downloaded by the notebook | yes (3/11) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/banknote_randomforest.ipynb) | Verified (re-run 2026-10-13, 35 s): runs clean. Data is downloaded in code |
| `california_regression.ipynb` | Linear regression in TensorFlow, then a random forest | California housing (Google MLCC), downloaded by the notebook | yes (17/27) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/california_regression.ipynb) | Verified (re-run 2026-10-13, 5 min): runs clean after fixing a plot error on current NumPy |
| `diabetes.ipynb` | Decision tree on diabetes data, tree plot | Pima Indians Diabetes (OpenML data set 37), downloaded by the notebook; substitute for the lost `diabetes.csv` | yes (10/19), no errors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/diabetes.ipynb) | Verified (re-run 2026-10-15, Pima Indians Diabetes substitute data): 10 s. `sklearn.externals.six` removed and the pydotplus/graphviz plot replaced by `plot_tree`. Results differ from the earlier outputs |
| `mnist_opencv.ipynb` | Small CNN on MNIST | MNIST (Keras loader) | yes (14/20) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/mnist_opencv.ipynb) | Verified (re-run 2026-10-13, 3 min): runs clean after fixing a removed Keras method |
| `plot_kmeans_silhouette_analysis.ipynb` | K-means and silhouette analysis | Synthetic, make_blobs | yes (1/2) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/plot_kmeans_silhouette_analysis.ipynb) | Executed 2026-10-13 (11 s). It had no outputs before |
| `tip_amount_tensorflow.ipynb` | Linear regression in TensorFlow | Restaurant tips (`plotly/datasets`), downloaded by the notebook; substitute for the lost `tip-amount.xlsx` | yes (9/13), no errors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/tip_amount_tensorflow.ipynb) | Verified (re-run 2026-10-15, restaurant tips substitute data): 15 s. The stored `KeyError` is fixed. Results differ from the earlier outputs |
| `titanic-tutorial.ipynb` | Titanic survival: feature engineering and models | Titanic passenger list (OpenML data set 40945), downloaded by the notebook; substitute for the Kaggle files, which need a login | yes (33/45), no errors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/titanic-tutorial.ipynb) | Verified (re-run 2026-10-15, OpenML Titanic substitute data): 36 s. One cell added that scores the models on the 418 held-out rows. Results differ from the earlier outputs |
| `world_alcohol_pandas_revision.ipynb` | pandas selection and filtering | WHO Global Health Observatory alcohol consumption by beverage type (CC BY 4.0), downloaded by the notebook; substitute for the lost `world_alcohol.csv` | yes (31/32), no errors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janeferdinant/Machine-learning-Course/blob/master/world_alcohol_pandas_revision.ipynb) | Verified (re-run 2026-10-15, WHO GHO substitute data): 23 s. Has about 41,500 rows instead of 100, so printed rows and counts differ |

Status labels: **Verified (re-run date)** means the notebook was executed top to bottom and its outputs were kept. **Verified (re-run date, substitute data)** means the same, but the notebook's original data file is not available, so it ran on the named public dataset instead and its results are not the original results. **Not re-run (reason)** means it was only edited as described, or not touched, and its stored outputs come from earlier sessions. The Colab links use the `master` branch, which is this repository's default branch.

## Results

Numbers come from the re-runs on 2026-10-13 (log: [docs/runs/2026-10-13-notebook-reruns.md](docs/runs/2026-10-13-notebook-reruns.md)), 2026-10-14 (log: [docs/runs/2026-10-14-cnn-mnist.md](docs/runs/2026-10-14-cnn-mnist.md)) and 2026-10-15 (log: [docs/runs/2026-10-15-selfcontained-notebooks.md](docs/runs/2026-10-15-selfcontained-notebooks.md)). Most of the notebooks do not set a random seed, so repeated runs differ slightly. The rows dated 2026-10-15 for the five substituted datasets are results on the substitute data; they are not the results of the original data files and must not be compared with the earlier stored outputs.

| Metric | Value | Status | How measured | Date |
|---|---|---|---|---|
| Banknote, random forest accuracy | 0.99125 (0.9912536443148688) | Verified (re-run 2026-10-13) | `banknote_randomforest.ipynb`, 343 held-out rows, confusion matrix [[184, 3], [0, 156]] | 2026-10-13 |
| California housing, random forest R² | 0.8094 (0.8094088629585355), MSE 2616.38 | Verified (re-run 2026-10-13) | `california_regression.ipynb`, as printed by the notebook | 2026-10-13 |
| MNIST, small CNN, 1 epoch at batch size 1 | training accuracy 0.8985 | Verified (re-run 2026-10-13) | `mnist_opencv.ipynb`. Training accuracy, not a test score | 2026-10-13 |
| MNIST, CNN (2 conv + dense), 5 epochs at batch size 10, full 60,000 training images | test accuracy 0.9905 (0.9904999732971191), test loss 0.0297 | Verified (re-run 2026-10-14, full settings, CPU) | `CNN_Identify_the_Digits_(MNIST).ipynb`, 10,000 MNIST test images (also used as the validation set; no tuning was done on it). Log: [docs/runs/2026-10-14-cnn-mnist.md](docs/runs/2026-10-14-cnn-mnist.md) | 2026-10-14 |
| Banknote, PyTorch network (4-8-4-1, 3,500 full-batch epochs), test accuracy | 1.000 (343 of 343), final training loss 0.0106 | Verified (re-run 2026-10-15, UCI banknote data) | `ANN_using_pytorch.ipynb`, 25% held-out rows (343), confusion matrix [[191, 0], [0, 152]] | 2026-10-15 |
| Banknote, Keras networks, test accuracy | 0.9621 (6 hidden units, SGD, 10 epochs) to 1.000 (four hidden layers with dropout, 20 epochs of Adam) | Verified (re-run 2026-10-15, UCI banknote data) | `ANN_using_tensorflow.ipynb`, 343 held-out rows. Unseeded | 2026-10-15 |
| SMS spam, Multinomial Naive Bayes on TF-IDF (1-2 grams), validation accuracy | 0.9426 (spam recall 0.48, precision 1.00) | Verified (re-run 2026-10-15, SMS Spam Collection substitute data) | `Natural_Language_Processing_with_Disaster_Tweets.ipynb`, 836 validation messages | 2026-10-15 |
| Pima diabetes, random forest (20 trees), test accuracy | 0.779 (180 of 231; confusion matrix [[129, 17], [34, 51]]) | Verified (re-run 2026-10-15, Pima substitute data) | `diabetes.ipynb`, 231 held-out rows. Unseeded trees: the single decision trees vary by a few points between runs | 2026-10-15 |
| Titanic, logistic regression, held-out accuracy | 0.780 (best of nine models; random forest 0.773) | Verified (re-run 2026-10-15, OpenML Titanic substitute data) | `titanic-tutorial.ipynb`, 418 held-out rows. The notebook's own scores are training accuracy | 2026-10-15 |
| Restaurant tips, one-feature linear regression, training RMSE after 10 epochs | 1.20 US dollars | Verified (re-run 2026-10-15, restaurant tips substitute data) | `tip_amount_tensorflow.ipynb`, 244 rows, as printed by the notebook. Unseeded; training error, not a test score | 2026-10-15 |
| WHO alcohol table | 41,521 rows, 5 columns | Verified (re-run 2026-10-15, WHO GHO substitute data) | `world_alcohol_pandas_revision.ipynb`, shape printed by the notebook. A pandas exercise, so no model metric | 2026-10-15 |

### CNN on MNIST with a separate validation split

`CNN_Identify_the_Digits_(MNIST)` above passes the test images to `model.fit` as validation data. The companion notebook [`CNN_Identify_the_Digits_(MNIST)_validation_split.ipynb`](CNN_Identify_the_Digits_(MNIST)_validation_split.ipynb) trains the same network (batch size 10, 5 epochs) on the first 50,000 training images, uses the last 10,000 training images as validation data, fixes the random seeds, and evaluates the test set once at the end. Over seeds 0, 1 and 2 the test accuracy was 0.9907, 0.9890 and 0.9887 (mean 0.9895): **Verified (re-run 2026-10-14, seeds 0, 1, 2, validation split = last 10,000 of the 60,000 training images)**. The difference from 0.9905 is about 0.1 percentage points, the same size as the spread between seeds, so these runs show no large inflation but cannot settle a small one; the new runs also use 50,000 rather than 60,000 training images. Details, per-epoch numbers and what the split changes: [docs/runs/2026-10-14-mnist-validation-split.md](docs/runs/2026-10-14-mnist-validation-split.md). The three seeds were run with [`scripts/mnist_validation_split.py`](scripts/mnist_validation_split.py) (`python scripts/mnist_validation_split.py --seed 0 --out seed0.json`, about 6 minutes each on an 8-core CPU).

## Tech stack

Python 3.11, Jupyter, TensorFlow 2.15.1 with Keras 2.15.0, scikit-learn 1.5.2, NumPy 1.26.4, pandas 2.2.3, matplotlib 3.9.2, seaborn 0.13.2. `ANN_using_pytorch.ipynb` uses PyTorch 2.5.1 (CPU build). The NLP notebook uses gensim 4.3.3, NLTK 3.9.1 and SciPy 1.13.1. All pins are in `requirements.txt`.

## Getting started

Prerequisites: Python 3.11 (TensorFlow 2.15 supports 3.9 to 3.11), about 3 GB of disk space for the packages (TensorFlow and the CPU build of PyTorch), and an internet connection for the dataset downloads. The commands below were tested on Windows; on Linux or macOS use `python3.11 -m venv .venv` and `source .venv/bin/activate` instead of the two Windows lines.

```bash
git clone https://github.com/Janeferdinant/Machine-learning-Course.git
cd Machine-learning-Course
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --output-dir run-output banknote_randomforest.ipynb
```

Expected: the last command writes `run-output/banknote_randomforest.ipynb` in well under a minute, and the cell that prints the random forest accuracy shows `0.9912536443148688`. To work interactively, install JupyterLab (`pip install jupyterlab`), run `python -m jupyter lab` and open any notebook. TensorFlow is a large install and can take a long time on a slow connection.

## Project structure

```
*.ipynb                    The 13 course notebooks plus the validation-split companion of the MNIST CNN
requirements.txt           Package versions used for the re-runs (2026-10-13 to 2026-10-15)
docs/runs/                 Logs of the re-runs
scripts/                   mnist_validation_split.py: the seeded validation-split run (three seeds)
finance.py                 A Colab export of a Streamlit stock demo; not valid Python (see Limitations)
THIRD_PARTY_NOTICES.md     Dataset and code credits and licenses
LICENSE                    CC BY-NC-ND 4.0
data/                      Created by the notebooks; git-ignored
```

## Real-world applications

Fraud and forgery detection (banknotes), property price estimation (California housing), digit and image recognition (MNIST), segmentation (k-means), and text classification (short messages).

## Limitations

- All thirteen notebooks have been re-run: four on 2026-10-13, `CNN_Identify_the_Digits_(MNIST)` and `SigmoidNeuron` on 2026-10-14, and the other seven on 2026-10-15. Five of those seven (`diabetes`, `titanic-tutorial`, `Natural_Language_Processing_with_Disaster_Tweets`, `tip_amount_tensorflow`, `world_alcohol_pandas_revision`) now run on public substitute data because their original files are not available. Their results are not the results of the original data, and the NLP notebook now classifies SMS messages although its file name still says "Disaster Tweets" (the name is kept so the Colab links keep working).
- The substitute datasets are downloaded from third-party servers. OpenML in particular sometimes answers with an error under load (seen during the 2026-10-15 runs); the Pima notebook then falls back to a GitHub copy of the same table, and the Titanic notebook retries a few times. If a download fails, run the cell again later.
- `CNN_Identify_the_Digits_(MNIST)` was re-run on 2026-10-14 with its original training settings (30,000 training steps, about 10 minutes of training on an 8-core CPU). The network diagram cell needs Graphviz and prints a note without it, and the personal-image cells (`2.jpeg`) are skipped because that file is not in the repository.
- The fixes to `tip_amount_tensorflow` (undefined variables, `feature[-1]`) and `ANN_using_tensorflow` (undefined predictions, empty `Dense()`) are listed in the note at the top of each notebook. The network diagrams in `ANN_using_tensorflow` are now drawn with matplotlib and show at most 10 nodes per layer.
- `diabetes` draws its trees with scikit-learn's `plot_tree` instead of pydotplus and Graphviz, and no longer writes `diabetes.png`.
- Results vary between runs because most notebooks set no seed. The `mnist_opencv` accuracy of 0.8985 is a training accuracy, not a test score; the CNN notebook's 0.9905 is measured on the 10,000 test images.
- `mnist_opencv` is named after OpenCV but does not use it.
- The versions in `requirements.txt` were tested locally, not in Colab. Colab ships newer TensorFlow, Keras and NumPy, so a notebook may behave differently there.
- `finance.py` contains `!pip install` lines, so it cannot run as a Python file; it is left as uploaded.
- The versions in `requirements.txt` were tested on Windows with Python 3.11 only. On Linux the extra index line in `requirements.txt` selects the CPU build of PyTorch; that was not tested here.

## Future scope

Set random seeds, add a Colab smoke test for the self-contained notebooks, rename the NLP notebook now that it no longer uses tweets, and map notebooks to lecture topics.

## Credits & citation

- **Origin.** This repository was created from the GitHub Learning Lab course "Get Started with TensorFlow" (everydeveloper) under the name `TensorFlow-Pokemon-Course`, and later renamed. No course content remains. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Code.** `plot_kmeans_silhouette_analysis` follows the scikit-learn example gallery (BSD-3-Clause). `california_regression` and `tip_amount_tensorflow` build on Google's Machine Learning Crash Course exercise "Linear regression with a real dataset" (Apache-2.0). Others follow public tutorials, listed with their confidence level in THIRD_PARTY_NOTICES.md.
- **Data.** Lohweg, V. (2012). Banknote Authentication [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55P57. California housing data via Google's Machine Learning Crash Course. MNIST by LeCun, Cortes and Burges. Substitute data used from 2026-10-15: Pima Indians Diabetes Database (Sigillito; original owner NIDDK) via OpenML and UCI; Titanic passenger list (Harrell and Cason, Vanderbilt Biostatistics) via OpenML; Almeida, T. and Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84 (CC BY 4.0); restaurant tips table (Bryant and Smith, 1995) via the `plotly/datasets` repository; World Health Organization, Global Health Observatory, recorded alcohol per capita consumption by beverage type (CC BY 4.0). Licenses and URLs are in THIRD_PARTY_NOTICES.md.

## License

Original work © Jane Rose Ferdinant, licensed CC BY-NC-ND 4.0. Third-party components retain their own licenses; see THIRD_PARTY_NOTICES.md.

## Author

Jane Rose Ferdinant: [Portfolio](https://janeferdinant.vercel.app) · [LinkedIn](https://www.linkedin.com/in/janerosenf/)
