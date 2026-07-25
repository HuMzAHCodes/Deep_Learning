# Lab Report: Hyperparameter Tuning of a Neural Network with Keras Tuner

**Dataset:** Pima Indians Diabetes (`diabetes.csv`)
**Task:** Binary classification — predict whether a patient has diabetes (`Outcome` = 0 or 1)
**Tools:** TensorFlow / Keras, Keras Tuner (`keras_tuner`), scikit-learn, pandas, NumPy

---

## 1. Theoretical Overview — What Is Being Practiced

When you build a neural network, there are two kinds of "knobs":

- **Parameters** — the weights and biases the network *learns on its own* during training (via backpropagation and gradient descent). You never set these by hand.
- **Hyperparameters** — the design choices *you* make **before** training begins. The network cannot learn these itself. Examples: which optimizer to use, how many hidden layers, how many neurons per layer, the activation function, the dropout rate.

The core problem this lab addresses: **hyperparameters are not learned, they are chosen — and choosing them well is hard.** A network with a good architecture and a bad choice of hyperparameters will underperform. Traditionally people guess these values (`32 neurons`, `adam`, `2 layers`) out of habit, but there is no guarantee those defaults are optimal for *this specific dataset*.

**Hyperparameter tuning** is the systematic search for the best combination of these choices. Instead of guessing, you define a *search space* (a menu of possible values for each hyperparameter), and a tuner automatically builds many models, trains each one briefly, evaluates them on validation data, and reports which combination scored best.

The specific technique used here is **Random Search** via the **Keras Tuner** library. Random Search works by:
1. Randomly sampling a combination of hyperparameters from the defined search space.
2. Building and training a model with that combination (a "trial").
3. Recording the objective metric (here, **validation accuracy**).
4. Repeating for a fixed number of trials (`max_trials`).
5. Returning the combination that scored highest.

The mechanism that makes this work in Keras Tuner is the **`hp` (HyperParameters) object**. Instead of hardcoding a value like `Dense(32)`, you write `Dense(hp.Int('units', 8, 128, step=8))`. This tells the tuner: *"don't fix this — treat it as something to search over."* Three helper methods define the search space:
- `hp.Choice(name, [list of options])` — pick one value from a discrete list (e.g. which optimizer).
- `hp.Int(name, min, max, step)` — pick an integer in a range (e.g. number of neurons or layers).
- `hp.Float(...)` — pick a floating-point value in a range (not used here, but the same idea).

**Why validation accuracy is the objective:** we tune against `val_accuracy` (performance on held-out test data), **not** training accuracy. If we tuned on training accuracy, the tuner would happily pick a model that memorizes the training set and generalizes poorly. Validation accuracy is our proxy for real-world performance.

**Supporting concepts also practiced:**
- **Feature scaling (`StandardScaler`)** — neural networks train faster and more stably when all input features are on a similar scale (mean 0, standard deviation 1). The diabetes features range wildly (Glucose in the 100s, DiabetesPedigreeFunction below 1), so scaling matters.
- **Dropout** — a regularization technique that randomly "switches off" a fraction of neurons during each training step, forcing the network not to over-rely on any single path. This combats overfitting. The dropout rate itself is a hyperparameter worth tuning.

---

## 2. What We Did — Quick Recall Summary

*(Read this section first to test yourself; the deep dive below fills in the details.)*

1. **Loaded and prepped the data** — read `diabetes.csv`, checked correlation of each feature with the outcome, split features (`x`) from label (`y`), standardized `x` with `StandardScaler`, and did a 70/30 train/test split.
2. **Built a baseline model by hand** — a simple `32 → 1` network with `adam`, trained for 10 epochs, just to establish a "guessed" starting point.
3. **Then ran the *same* Keras Tuner technique three separate times**, each time searching for **one** hyperparameter while holding the rest fixed:
   - **Search 1 → Best optimizer?** → answer: **`rmsprop`**
   - **Search 2 → Best number of neurons (units) in the hidden layer?** → answer: **`80`**
   - **Search 3 → Best number of hidden layers?** → answer: **`6`**
4. **Finally, combined everything into one big search** — tuned number of layers, units per layer, activation per layer, dropout per layer, *and* optimizer all at once, then trained the winning model for 200 epochs.

Your one-line intuition was correct: **we used the same tuning technique three times to isolate the best value of one parameter at a time, then did everything together at the end.**

---

## 3. Deep Dive — Section by Section

### Section A — Data Loading and Exploration

```python
import numpy as np
import pandas as pd
df = pd.read_csv("diabetes.csv")
df.head(3)
```

The dataset has 8 feature columns (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) and 1 target column (`Outcome`).

```python
df.corr()['Outcome']
```

This checks how strongly each feature linearly correlates with the outcome:

| Feature | Correlation with Outcome |
|---|---|
| Glucose | 0.467 |
| BMI | 0.293 |
| Age | 0.238 |
| Pregnancies | 0.222 |
| DiabetesPedigreeFunction | 0.174 |
| Insulin | 0.131 |
| SkinThickness | 0.075 |
| BloodPressure | 0.065 |

**Interpretation:** Glucose is by far the most predictive single feature, followed by BMI and Age. BloodPressure and SkinThickness barely correlate. This is exploratory — no features are dropped — but it confirms the target is genuinely learnable from these inputs.

### Section B — Feature/Label Split and Preprocessing

```python
x = df.iloc[:, :-1].values     # all columns except the last → features
y = df.iloc[:, -1]             # last column → label (Outcome)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)    # standardize each feature to mean 0, std 1
```

`iloc[:, :-1]` slices every row and every column *except the last*; `iloc[:, -1]` grabs only the last column. `StandardScaler.fit_transform` computes the mean and standard deviation of each feature and rescales it. This is essential before feeding data to a neural network so that a large-magnitude feature (like Glucose) doesn't dominate the gradient updates simply because of its scale.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1)
```

30% of the data is held out as a test/validation set. `random_state=1` fixes the split so results are reproducible across runs.

### Section C — The Baseline (Hand-Guessed) Model

```python
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(32, activation="relu", input_dim=8))   # hidden layer: 32 neurons, 8 inputs
model.add(Dense(1, activation="sigmoid"))              # output: 1 neuron for binary prob

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test))
```

This is a **manually chosen** architecture — nothing is tuned. Key design points:
- `input_dim=8` because there are 8 features.
- Hidden layer uses **ReLU** activation (standard for hidden layers).
- Output layer uses **sigmoid**, which squashes the output to a probability between 0 and 1 — correct for binary classification.
- Loss is **binary crossentropy**, the standard loss for two-class problems.

This model exists purely as a reference point. The rest of the lab asks: *can we do better than these guesses by searching systematically?* The markdown cells make the plan explicit:

```
#1. How to select the optimizer
#2. How to decide on hidden layers number
#3. How to decide the node numbers
#4. All in one model
```

### Section D — Setting Up Keras Tuner

```python
!pip install -q keras-tuner
import keras_tuner as kt
```

Keras Tuner is the library that automates the search. Everything below reuses the *same pattern*, changing only which hyperparameter is left free.

---

### Section E — Search 1: Finding the Best Optimizer

```python
def build_model(hp):
  model = Sequential()
  model.add(Dense(32, activation="relu", input_dim=8))
  model.add(Dense(1, activation="sigmoid"))

  model.compile(
      optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adadelta']),
      loss="binary_crossentropy", metrics=["accuracy"])
  return model
```

Note what changed vs. the baseline: the architecture is **frozen** (32 neurons, 1 hidden layer), but the optimizer is now `hp.Choice(...)` — the tuner will try `adam`, `sgd`, `rmsprop`, and `adadelta` and see which wins.

```python
tuner = kt.RandomSearch(build_model, objective="val_accuracy", max_trials=5)
tuner.search(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))
tuner.get_best_hyperparameters()[0].values
```

- `objective="val_accuracy"` — optimize for validation accuracy.
- `max_trials=5` — try up to 5 random combinations (here, that covers the small optimizer menu).
- `tuner.search(...)` runs the whole experiment.
- `get_best_hyperparameters()[0].values` reports the winner.

**Result:**
```python
{'optimizer': 'rmsprop'}
```

So `rmsprop` beat the default `adam` for this dataset.

```python
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
model.summary()
model.fit(X_train, Y_train, batch_size=32, epochs=100, initial_epoch=6)
```

`tuner.hypermodel.build(best_hps)` rebuilds a fresh model using the winning hyperparameters, and `model.fit(... initial_epoch=6)` continues training it further (starting the epoch counter at 6, since the search already trained a few epochs). `model.summary()` confirmed the tiny `32 → 1` architecture (321 total trainable parameters).

---

### Section F — Search 2: Finding the Best Number of Neurons (Units)

```python
def build_model(hp):
  model = Sequential()
  # try unit counts 8, 16, 24, ..., 128 and search for the best one
  units = hp.Int('units', min_value=8, max_value=128, step=8)
  model.add(Dense(units=units, activation="relu", input_dim=8))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
  return model
```

Now the **optimizer is frozen to `rmsprop`** (the winner from Search 1), and the free hyperparameter is the **number of neurons** in the hidden layer. `hp.Int('units', 8, 128, step=8)` defines the search space as {8, 16, 24, …, 128}.

```python
tuner = kt.RandomSearch(build_model, objective="val_accuracy", max_trials=5,
                        directory="mydir", project_name="hamza")
tuner.search(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))
tuner.get_best_hyperparameters()[0].values
```

New arguments `directory` and `project_name` tell the tuner where to save its logs and results on disk, so multiple searches don't collide. (This is why each subsequent search uses a different `project_name`.)

**Result:**
```python
{'units': 80}
```

So 80 neurons in the hidden layer outperformed the original guess of 32.

---

### Section G — Search 3: Finding the Best Number of Hidden Layers

```python
def build_model(hp):
  model = Sequential()
  # input layer — fixed
  model.add(Dense(80, activation="relu", input_dim=8))

  # search how many hidden layers to add: 1 to 10
  for i in range(hp.Int("num_layers", min_value=1, max_value=10)):
    model.add(Dense(80, activation="relu"))

  model.add(Dense(1, activation="sigmoid"))
  model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
  return model
```

This is the clever bit: `hp.Int("num_layers", 1, 10)` is placed *inside* `range(...)`, so the tuner decides **how many times the loop runs** — i.e. how many hidden layers get stacked. The units (80, from Search 2) and optimizer (`rmsprop`, from Search 1) are now frozen; only **depth** is being searched.

```python
tuner = kt.RandomSearch(build_model, objective="val_accuracy", max_trials=4,
                        directory="mydir", project_name="num_layers")
tuner.search(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))
tuner.get_best_hyperparameters()[0].values
```

**Result:**
```python
{'num_layers': 6}
```

So 6 hidden layers was the best-scoring depth in this search.

```python
model = tuner.get_best_models(num_models=1)[0]
model.fit(X_train, Y_train, epochs=100, initial_epoch=6, validation_data=(X_test, Y_test))
```

The best model is retrieved directly and trained further to 100 epochs.

> **The pattern across Searches 1–3:** identical Random Search technique, one free hyperparameter at a time, each search's winner "locked in" before the next. This is a controlled, one-variable-at-a-time approach — easy to reason about, but it assumes the hyperparameters are independent of each other (which isn't strictly true).

---

### Section H — The Final "All-in-One" Search

The last section drops the one-at-a-time restriction and searches **everything simultaneously**:

```python
def build_model(hp):
  model = Sequential()
  counter = 0

  # search number of hidden layers: 1 to 10
  for i in range(hp.Int('num_layers', min_value=1, max_value=10)):

    if counter == 0:
      # first hidden layer needs input_dim
      model.add(Dense(
          hp.Int('units' + str(i), min_value=8, max_value=128, step=8),
          activation=hp.Choice('activation' + str(i), values=['relu', 'tanh', 'sigmoid']),
          input_dim=8))
      model.add(Dropout(hp.Choice('dropout' + str(i), values=[.1,.2,.3,.4,.5,.6,.7])))
    else:
      # later layers — Keras infers input shape
      model.add(Dense(
          hp.Int('units' + str(i), min_value=8, max_value=128, step=8),
          activation=hp.Choice('activation' + str(i), values=['relu', 'tanh', 'sigmoid'])))
      model.add(Dropout(hp.Choice('dropout' + str(i), values=[.1,.2,.3,.4,.5,.6,.7])))
    counter += 1

  model.add(Dense(1, activation='sigmoid'))

  model.compile(
      optimizer=hp.Choice('optimizer', values=['rmsprop','adam','sgd','nadam','adadelta']),
      loss='binary_crossentropy', metrics=['accuracy'])
  return model
```

What makes this different and more powerful:
- **Per-layer hyperparameters.** Because the layer names are built dynamically with `'units' + str(i)`, `'activation' + str(i)`, `'dropout' + str(i)`, **each hidden layer gets its own independently-searched** unit count, activation function, and dropout rate. Layer 0 might end up with 32 relu neurons and 0.6 dropout while layer 1 has 80 tanh neurons and 0.2 dropout.
- **Dropout is now tunable** — each layer's dropout rate is chosen from {0.1 … 0.7}.
- **Activation is now tunable** — each layer independently picks relu, tanh, or sigmoid.
- **Optimizer is tunable again** (5 options this time, adding `nadam`).
- The `counter == 0` branch exists only because the **first** hidden layer must declare `input_dim=8`, while later layers don't need it — Keras infers their input shape automatically.

```python
tuner = kt.RandomSearch(build_model, objective="val_accuracy", max_trials=3,
                        directory="mydir", project_name="final1")
tuner.search(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))
tuner.get_best_hyperparameters()[0].values
```

**Result (the full winning configuration):**
```python
{'num_layers': 1,
 'units0': 32, 'activation0': 'relu',  'dropout0': 0.6,
 'optimizer': 'adam',
 # (units1..6 etc. were also sampled but only layer 0 is used since num_layers=1)
 'units1': 80, 'activation1': 'tanh', 'dropout1': 0.2,
 'units2': 24, 'activation2': 'tanh', 'dropout2': 0.2,
 ... }
```

Note an important subtlety: the winner here has **`num_layers: 1`** and **`optimizer: adam`**, which *differs* from the earlier isolated searches (which favored 6 layers and rmsprop). This is expected — with only `max_trials=3` the search barely samples a huge combined space, and because all hyperparameters interact, the best combination is not simply the concatenation of the individually-best values. It illustrates the tradeoff: the all-in-one search is more realistic (it explores interactions) but needs far more trials to search reliably.

```python
model = tuner.get_best_models(num_models=1)[0]
model.fit(X_train, Y_train, epochs=200, initial_epoch=6, validation_data=(X_test, Y_test))
```

The winning model is extracted and trained to 200 epochs. The training log settled around **~78–79% validation accuracy**, a reasonable score for this dataset.

---

## 4. Key Takeaways

1. **Hyperparameters are chosen, not learned** — and searching for them beats guessing.
2. **Keras Tuner + `hp.Choice`/`hp.Int`** turns any hardcoded value into a searchable dimension; `RandomSearch` samples combinations and keeps the best by `val_accuracy`.
3. **Isolating one hyperparameter at a time** (Searches 1–3) is clean and interpretable — best optimizer `rmsprop`, best units `80`, best depth `6` — but it ignores interactions between hyperparameters.
4. **Searching everything at once** (Section H) is more realistic because hyperparameters influence each other, but it needs many more trials to search the vastly larger space well — which is why its "best" result differed from the one-at-a-time findings.
5. Practical scaffolding matters: **`StandardScaler`** for stable training, **`dropout`** for regularization, **`sigmoid` + `binary_crossentropy`** for binary classification, and **`val_accuracy`** (not training accuracy) as the honest objective.

---

## Appendix — Best Hyperparameters at a Glance

| Search | Free hyperparameter | Everything else fixed to | Winner |
|---|---|---|---|
| 1 | optimizer | 32 units, 1 layer | **rmsprop** |
| 2 | units (neurons) | rmsprop, 1 layer | **80** |
| 3 | num_layers | rmsprop, 80 units | **6** |
| 4 (all-in-one) | layers + units + activation + dropout + optimizer (per layer) | nothing | 1 layer, 32 units, relu, dropout 0.6, adam |
