# Run log, 2026-10-15: seven notebooks made self-contained and re-run

Seven notebooks could not be re-run in the earlier sessions because they needed data files or packages that are not in the repository. This log records what was missing, what each notebook uses now, the settings, the runtime, the metrics and the status. Numbers on substitute data are results on that substitute data. They are not the results of the original data files and must not be compared with the outputs the notebooks had before.

## Environment

- Windows 11, AMD Ryzen 7 5800X (8 cores, 16 threads), 32 GB RAM, CPU only (no GPU used, no CUDA installed).
- Python 3.11.9, fresh virtual environment, `pip install -r requirements.txt` (the file in this commit): TensorFlow 2.15.1, Keras 2.15.0, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, SciPy 1.13.1, matplotlib 3.9.2, seaborn 0.13.2, PyTorch 2.5.1 (CPU wheel, `2.5.1+cpu`), gensim 4.3.3, NLTK 3.9.1, nbconvert 7.16.4. `pip check` reported no broken requirements.
- Command for each notebook, run from the repository root with the git-ignored `data/` folder absent, so every download ran:
  `python -m jupyter nbconvert --to notebook --execute --output-dir run-output --ExecutePreprocessor.timeout=1200 <notebook>.ipynb`
  The executed copy was then copied over the notebook (outputs kept) after local paths in the stored warnings were replaced with `...`.
- Runtime is the wall-clock time of that command (kernel start, downloads and execution). All seven are far under the 20-minute limit. No epoch or data-size reduction was needed.
- No random seeds were added (the notebooks that already set seeds or `random_state` keep them), so unseeded numbers change slightly from run to run.

## Failed attempts before the final runs (kept for honesty)

- The first execution of `tip_amount_tensorflow` failed in `plot_the_model` (a ragged array from `trained_bias`); fixed, then re-run.
- OpenML often refuses requests under load. Over six execution sequences of these notebooks `diabetes` failed four times and `titanic-tutorial` twice, with "Dataset with data_id 37 not found", "HTTP Error 503" or "Database connection error". The `diabetes` data cell now downloads the ARFF file directly, retries, and falls back to the identical table in `plotly/datasets`; `titanic-tutorial` retries up to four times (it is the one place where the fallback is not available). In the committed `titanic-tutorial` run one request got HTTP 503 and the retry succeeded; the stored output shows that message.
- Order of the final executions: the six other notebooks in one sequence, then `diabetes` on its own (10 s; OpenML answered, so the plotly fallback was not used in the committed outputs).

## Per notebook

### `ANN_using_pytorch.ipynb`

- **Missing:** the `torch` package (not in `requirements.txt`). The data cell already downloaded the UCI banknote file.
- **Now uses:** same data, Banknote Authentication, Lohweg, V. (2012), UCI Machine Learning Repository, https://doi.org/10.24432/C55P57, CC BY 4.0, https://archive.ics.uci.edu/dataset/267/banknote+authentication. PyTorch 2.5.1 CPU wheel is pinned.
- **Settings:** 1,372 rows, 75% / 25% split (`random_state=42`), network 4-8-4-1 with sigmoid output, SGD learning rate 0.01, 3,500 full-batch epochs, `torch.manual_seed(20)`. Unchanged from the notebook.
- **Runtime:** 17 s.
- **Metrics:** final training loss 0.0106; test confusion matrix [[191, 0], [0, 152]] (343 of 343 correct).
- **Code changes:** none. A note cell was added at the top.
- **Status:** Verified (re-run 2026-10-15, UCI banknote data).

### `ANN_using_tensorflow.ipynb`

- **Missing:** `ann_visualizer` (needs the Graphviz program, not installed), plus four code errors that stopped the notebook: `Dense()` without units, `Input` not imported, `y_pred2` and `y_class2` undefined.
- **Now uses:** same UCI banknote data as above. The network diagrams are drawn by a short matplotlib function named `ann_viz` (at most 10 nodes per layer).
- **Settings:** unchanged: Keras Sequential models, 75% / 25% split (`random_state=42`), SGD for 10 epochs and Adam for 5 to 20 epochs. Not seeded.
- **Runtime:** 26 s.
- **Metrics (test, 343 rows):** `model1` (6 hidden units, SGD, 10 epochs) accuracy 0.9621, confusion matrix [[185, 6], [7, 145]]; `model21` (same plus dropout 0.2) 0.9242; `model3` (64-32-16-8 with dropout, Adam, 20 epochs in total) confusion matrix [[191, 0], [0, 152]]; `model01` (256 units, 2 epochs) [[191, 0], [3, 149]].
- **Status:** Verified (re-run 2026-10-15, UCI banknote data).

### `Natural_Language_Processing_with_Disaster_Tweets.ipynb`

- **Missing:** the Kaggle competition files `train.csv` and `test.csv` (Kaggle login required, and no permissively licensed copy of those tweets was found), plus the gensim 3 API (`size=`, `wv.vocab`) and `get_feature_names`.
- **Now uses:** SMS Spam Collection, Almeida, T. and Hidalgo, J. (2011), UCI Machine Learning Repository, https://doi.org/10.24432/C5CC84, CC BY 4.0 (stated on the UCI page, read 2026-10-15), https://archive.ics.uci.edu/dataset/228/sms+spam+collection. 5,574 messages (747 spam, 4,827 ham). It is a short-text classification task of the same kind; the preprocessing, bag-of-words, TF-IDF, word2vec and Naive Bayes steps are unchanged. The keyword and location cells were replaced by a message-length cell, because the SMS data has no such columns. NLTK `stopwords` and `wordnet` are downloaded at run time.
- **Settings:** shuffle with seed 42, 80% train (4,459 rows, 4,180 after dropping duplicate texts) / 20% test (1,115 rows, no labels), validation split 80% / 20% (`random_state=42`), word2vec `vector_size=50`, `min_count=2`, `seed=42` (workers=4, so not exactly reproducible), Multinomial Naive Bayes on TF-IDF with 1 to 2 grams (30,990 features, dense). The dense arrays use about 2 to 3 GB of RAM at peak (estimated from the array sizes, not measured).
- **Runtime:** 11 s.
- **Metrics:** validation accuracy 0.9426 on 836 messages, confusion matrix [[744, 0], [48, 44]]; spam recall 0.48, spam precision 1.00. The accuracy is high mainly because 89% of the validation messages (744 of 836) are ham.
- **Status:** Verified (re-run 2026-10-15, SMS Spam Collection substitute data).

### `diabetes.ipynb`

- **Missing:** `diabetes.csv`; `sklearn.externals.six` (removed from scikit-learn); the pydotplus and Graphviz picture (Graphviz program not installed).
- **Now uses:** the Pima Indians Diabetes Database (768 rows; original owner NIDDK, donor Vincent Sigillito, via UCI), OpenML data set 37, licence field "Public" (read 2026-10-15), https://api.openml.org/data/v1/download/37/diabetes.arff. Fallback if OpenML is overloaded: the same table in https://github.com/plotly/datasets (MIT licence for the repository, no separate licence for the table); the two copies were compared and hold identical numbers. Trees are drawn with `sklearn.tree.plot_tree`.
- **Settings:** 70% / 30% split (`random_state=1`), standardised features, `DecisionTreeClassifier()` (unseeded), then entropy tree with `max_depth=3`, then `RandomForestClassifier(n_estimators=20, random_state=0)`.
- **Runtime:** 10 s (run on its own, see above).
- **Metrics (231 test rows):** unrestricted decision tree 0.6926, entropy tree of depth 3 0.7706, random forest 0.7792 (confusion matrix [[129, 17], [34, 51]]). The single trees are unseeded and vary by a few points between runs (an earlier run of the same code gave 0.6753 and 0.7706).
- **Status:** Verified (re-run 2026-10-15, Pima Indians Diabetes substitute data).

### `tip_amount_tensorflow.ipynb`

- **Missing:** `tip-amount.xlsx` (source not recorded), and three code problems: `RMSprop(lr=...)`, `feature[-1]` in the plot function (the stored `KeyError`), and cells using undefined `x` and `y`.
- **Now uses:** the restaurant `tips` table (244 rows; total bill, tip, sex, smoker, day, time, size; Bryant and Smith, 1995) from the `plotly/datasets` repository, pinned to commit 79f2104, https://raw.githubusercontent.com/plotly/datasets/79f21048146faece1dd7e9ebc815612d6662c7bc/tips.csv. The repository is under the MIT licence; no separate licence is stated for the table. The notebook uses `total_bill` as the single feature and `tip` as the label.
- **Settings:** unchanged: one Dense unit, RMSprop learning rate 0.01, 10 epochs, batch size 12. Not seeded.
- **Runtime:** 15 s.
- **Metrics:** training RMSE after epoch 10 = 1.2027 US dollars (the loss was still falling; training error, not a test score; another unseeded run gave 1.1210).
- **Status:** Verified (re-run 2026-10-15, restaurant tips substitute data).

### `titanic-tutorial.ipynb`

- **Missing:** the Kaggle `train.csv` and `test.csv` (Kaggle login required).
- **Now uses:** the Titanic passenger list, OpenML data set 40945 (Frank E. Harrell Jr. and Thomas Cason, Vanderbilt Biostatistics), licence field "Public" (read 2026-10-15), https://www.openml.org/d/40945, loaded with `fetch_openml`. 1,309 rows, shuffled with seed 42 and split 891 train / 418 test with the Kaggle column names (`PassengerId` is a new row number; `boat`, `body` and `home.dest` are dropped because boat and body reveal survival). The test rows keep their labels in a separate file so one added cell can score the models on them.
- **Settings:** unchanged model list and settings (random forest with 100 trees and the other eight models, mostly unseeded).
- **Runtime:** 36 s.
- **Metrics:** the notebook's own scores are training-set accuracy (random forest and decision tree 95.96%, logistic regression 81.93%). On the 418 held-out rows (added cell): logistic regression 77.99%, random forest 77.27%, linear SVC 76.56%, naive Bayes 75.84%, decision tree 74.16%, KNN 70.57%, SVM 67.70%, SGD 66.27%, perceptron 63.64%.
- **Status:** Verified (re-run 2026-10-15, OpenML Titanic substitute data).

### `world_alcohol_pandas_revision.ipynb`

- **Missing:** `world_alcohol.csv` (source not recorded; the file held 100 rows for 1985 to 1989), and a typo (`w_a_con` for `df`) that stopped the first cell.
- **Now uses:** WHO Global Health Observatory indicator SA_0000001400, "Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol), by beverage type", 1960 to 2024, from the GHO OData API (https://ghoapi.azureedge.net/api/SA_0000001400, country names from the COUNTRY dimension). WHO data terms: CC BY 4.0 unless specifically indicated otherwise (https://data.who.int/about/data/terms-and-conditions, read 2026-10-15). The notebook rebuilds the five columns it uses (`Year`, `WHO region`, `Country`, `Beverage Types`, `Display Value`) and drops the "total" rows: 41,521 rows, 48 of them without a value.
- **Settings:** `random_state=0` added to the two `df.sample` calls so the printed samples repeat. Nothing else changed.
- **Runtime:** 23 s.
- **Metrics:** none (a pandas exercise). Shape printed by the notebook: (41521, 5).
- **Status:** Verified (re-run 2026-10-15, WHO GHO substitute data).

## Result of the run

All seven notebooks ran top to bottom with exit code 0 and 0 error outputs; their executed outputs are stored in the notebooks. Code cells with output / code cells: PyTorch ANN 14/23, TensorFlow ANN 61/78, NLP 39/52, diabetes 10/19, tip 9/13, titanic 33/45, world alcohol 31/32 (the rest are empty cells or cells that print nothing). No stored output contains a local path (checked by searching the notebooks for the user name, `AppData`, `portfolio-audit` and drive-letter paths).
