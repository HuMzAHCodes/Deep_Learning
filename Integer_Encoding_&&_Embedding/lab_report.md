# Lab Report

## Text Vectorization: Integer Encoding vs. Word Embedding
*Sentiment classification with a SimpleRNN on the IMDB dataset (Keras)*

| Field | Detail |
|---|---|
| Name | Naini |
| Roll No. | ________________ |
| Course | ________________ |
| Date | ________________ |

---

## Contents

1. [Objective](#1-objective)
2. [Introduction](#2-introduction)
3. [Theoretical Background](#3-theoretical-background)
4. [Experiment 1 — Integer Encoding + SimpleRNN (Baseline)](#4-experiment-1--integer-encoding--simplernn-baseline)
5. [Experiment 2 — Word Embedding](#5-experiment-2--word-embedding)
6. [Difference Between the Two Concepts](#6-difference-between-the-two-concepts)
7. [Conclusion](#7-conclusion)

---

## 1. Objective

To numerically represent text for a neural network using two techniques —
**integer encoding** and **word embedding** — and to demonstrate, with a
SimpleRNN sentiment classifier on the IMDB dataset, the **shortcomings of
integer encoding** and **how word embedding solves them**.

---

## 2. Introduction

Neural networks work on numbers, not on raw words, so text must first be
converted into a numeric form (vectorized). This lab studies two ways of doing
that:

- **Integer encoding** — replace every word with a single integer index.
- **Word embedding** — replace every word with a learned vector of many numbers.

The lab is built in three parts. **Part A** demonstrates integer encoding on a
tiny 10-sentence corpus so the mechanics are visible. **Part B** trains a
SimpleRNN on the IMDB reviews using the raw integers — and fails. **Part C**
introduces a word embedding, first demonstrating what it does, then applying it
to IMDB, where accuracy improves. Because the same RNN is used throughout, any
difference in results is due to the **input representation** alone.

---

## 3. Theoretical Background

### 3.1 Why text must be vectorized

A neural network performs arithmetic on numeric tensors. Words are symbolic, so
a mapping from words to numbers is required before any learning can happen. The
quality of this mapping controls what information about the words the network is
even able to see — so it matters as much as the model itself.

### 3.2 Integer Encoding

Integer encoding is the simplest numeric representation of text:

1. **Tokenization** — split the corpus into words and build a vocabulary of
   unique tokens.
2. **Indexing** — assign each unique word a unique integer, usually ranked by
   frequency (index 1 = most frequent). Index 0 is reserved for padding.
3. **Encoding** — replace each word in a sentence by its integer, producing a
   sequence of integers.
4. **Padding** — pad or truncate all sequences to a fixed length so they form a
   rectangular tensor the model can accept.

Each document then becomes a fixed-length vector of integers. The IMDB dataset
used in this lab already ships in this integer-encoded form, so its reviews
arrive as lists of word-IDs.

### 3.3 Shortcomings of Integer Encoding

Feeding raw integer indices to a model has serious problems:

- **False magnitude (the core flaw):** integers imply order and size
  (`5000 > 5`), but word indices are arbitrary labels. Word-ID `2071` is not
  "bigger" or "more" than word-ID `56`. When the model reads these IDs as
  numeric features it infers relationships that do not exist. In the baseline,
  each word is handed to the RNN as a **single number** (see `input_shape=(50, 1)`
  below), which carries almost no usable information — so the model cannot learn
  and settles at ~50% accuracy (random guessing on two classes).
- **No semantic similarity:** related words such as "good" and "great" get
  unrelated integers, so the encoding captures nothing about meaning.
- **Sparsity of the one-hot fix:** one way to remove the false magnitude is
  one-hot encoding, but that produces very high-dimensional, sparse vectors
  (length = vocabulary size) that are memory-heavy and still carry no similarity
  information.
- **Not learnable:** the integer mapping is fixed in advance and never adapts to
  the task during training.

#### Understanding `input_shape=(50, 1)`

An RNN reads a sequence one step at a time, so its input is described as
`(timesteps, features_per_timestep)`:

- **50** = timesteps = how many tokens are in each (padded) review.
- **1** = features per timestep = how many numbers describe **one** token.

So `(50, 1)` means *"a sequence of 50 steps, and at each step I give you just 1
number."* That single number is the raw word-ID — one meaningless value per
word. This `1` is precisely the flaw that embedding fixes by making it *many*
numbers per word.

### 3.4 Word Embeddings

A word embedding maps each integer index to a **dense, low-dimensional vector**
of real numbers (for example 8, 32, or 100 dimensions). In Keras this is the
`Embedding` layer: a **trainable lookup table** of shape
`(vocabulary size × embedding dimension)`, where each integer simply selects one
**row** of the table (its vector). These vectors start random and are then
adjusted by backpropagation during training, so words used in similar contexts
drift towards similar vectors.

### 3.5 How Embedding solves the shortcomings

- **Removes false magnitude:** integers are used only as lookup indices, never
  as numeric quantities, so no fake ordinal relationship is introduced.
- **Captures similarity:** the learned vectors place related words close
  together in vector space, giving the network real linguistic structure.
- **Many numbers per word:** each token becomes a vector (e.g. 32 numbers)
  instead of one — which is why the embedding model *removes* the flawed
  `input_shape=(50, 1)` entirely; the `Embedding` layer now decides the number
  of features per token.
- **Dense and learnable:** a small dense vector replaces the huge sparse one-hot
  vector, and it is trained jointly with the model, so it is tuned to the task.

---

## 4. Experiment 1 — Integer Encoding + SimpleRNN (Baseline)

### 4.1 Integer encoding on a toy corpus

A small corpus is tokenized and integer-encoded so the mechanics are visible.

```python
import numpy as np
# NOTE: on current Colab (Keras 3), keras.preprocessing.text was removed.
# Use the tensorflow.keras path, which still ships the legacy Tokenizer.
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

docs = ['go india', 'india india', 'hip hip hurray',
        'jeetega bhai jeetega india jeetega', 'bharat mata ki jai',
        'kohli kohli', 'sachin sachin', 'dhoni dhoni',
        'modi ji ki jai', 'inquilab zindabad']

# Build a word -> integer vocabulary (oov_token handles unseen words)
tokenizer = Tokenizer(oov_token='<nothing>')
tokenizer.fit_on_texts(docs)

tokenizer.word_index      # the learned word -> integer mapping
tokenizer.word_counts     # frequency of each word
tokenizer.document_count  # number of documents (10)

# INTEGER ENCODING: each sentence -> a list of integer IDs
sequences = tokenizer.texts_to_sequences(docs)

# Pad to equal length (zeros added at the end)
sequences = pad_sequences(sequences, padding='post')
```

These 10 sentences are only a demonstration of *how* integer encoding works;
they are not used to train any model.

### 4.2 Switching to the pre-encoded IMDB dataset

IMDB is a large, real dataset of movie reviews that has **already been
integer-encoded**, so we skip the tokenizing step and just load it. Each review
arrives as a list of word-IDs.

```python
from keras.datasets import imdb
from keras import Sequential
from keras.layers import Dense, SimpleRNN

# Reviews arrive PRE-encoded as integer sequences
(X_train, y_train), (X_test, y_test) = imdb.load_data()

# Reviews have DIFFERENT lengths. X_train[2] is the 3rd review (index starts
# at 0); len() shows its word count (e.g. 141). A network needs every input to
# be the SAME size, which the next step enforces.
print(X_train[0])
len(X_train[2])
```

```python
# Force every review to exactly 50 tokens:
#   longer than 50  -> truncated (extra words cut off)
#   shorter than 50 -> padded with 0s at the end (padding='post')
# '50 tokens' is a CHOICE, not a rule. 50 is quite short -- a 141-word review
# loses most of its words -- which is one reason the baseline learns poorly.
X_train = pad_sequences(X_train, padding='post', maxlen=50)
X_test  = pad_sequences(X_test,  padding='post', maxlen=50)
```

### 4.3 The flawed baseline model

```python
# input_shape=(50, 1) = (timesteps, features_per_timestep):
#   50 timesteps      -> 50 tokens per review
#   1 feature/token   -> each token is just ONE raw word-ID
# One meaningless number per word gives the RNN nothing to learn from.
model = Sequential()
model.add(SimpleRNN(32, input_shape=(50, 1), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

### 4.4 Observations

Training and validation accuracy stay around **0.50** across all epochs — no
better than random guessing. The raw integer IDs carry a false magnitude and no
meaning, so the model has nothing useful to learn from.

---

## 5. Experiment 2 — Word Embedding

### 5.1 Demonstration: watching integers become vectors

Before applying embedding to IMDB, we demonstrate what it does on the same
10 sentences, using fresh self-contained variables.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import Sequential
from keras.layers import Embedding

# A self-contained copy of the sentences and its own tokenizer
demo_docs = ['go india', 'india india', 'hip hip hurray',
             'jeetega bhai jeetega india jeetega', 'bharat mata ki jai',
             'kohli kohli', 'sachin sachin', 'dhoni dhoni',
             'modi ji ki jai', 'inquilab zindabad']

demo_tokenizer = Tokenizer(oov_token='<nothing>')
demo_tokenizer.fit_on_texts(demo_docs)

# Integer-encode and pad (embedding always starts from integers)
demo_seqs = demo_tokenizer.texts_to_sequences(demo_docs)
demo_seqs = pad_sequences(demo_seqs, padding='post')

# The Embedding layer needs two numbers:
#   demo_vocab = number of rows in the lookup table (+1 for the padding index 0)
#   demo_len   = tokens per padded sentence
demo_vocab = len(demo_tokenizer.word_index) + 1
demo_len   = demo_seqs.shape[1]
```

```python
# Embedding = trainable lookup table of shape (demo_vocab x 8).
# Each integer ID selects ONE row = one 8-number vector.
demo_model = Sequential()
demo_model.add(Embedding(input_dim=demo_vocab, output_dim=8, input_length=demo_len))
demo_model.compile('adam', 'mse')   # compiled only so we can run predict()

# Pass the integer sentences through the layer and compare shapes:
demo_vectors = demo_model.predict(demo_seqs)
print("Input (integers) shape:", demo_seqs.shape)     # (10, demo_len)
print("Output (vectors) shape:", demo_vectors.shape)  # (10, demo_len, 8)

demo_vectors[0]   # first sentence: each integer is now an 8-number VECTOR
```

```python
# The layer's weights ARE the word vectors: one ROW per vocabulary index.
# Shape (demo_vocab, 8) -> every word (plus padding row 0) has its own vector.
embedding_matrix = demo_model.layers[0].get_weights()[0]
print("Embedding matrix shape:", embedding_matrix.shape)
embedding_matrix[2]   # the vector for word index 2 ('india')
```

This makes the concept concrete: the input of shape `(10, demo_len)` becomes
`(10, demo_len, 8)` — every single integer is expanded into an 8-dimensional
vector, and the embedding matrix is the lookup table holding one vector per word.

### 5.2 Applying embedding to IMDB

```python
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

# Cap the vocabulary to the 10,000 most frequent words (drop rare noise)
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Keep more context than the baseline (200 tokens instead of 50)
maxlen = 200
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test  = pad_sequences(X_test,  padding='post', maxlen=maxlen)
```

```python
# KEY CHANGE: there is NO input_shape=(50, 1) here.
# The Embedding layer turns each integer ID into a learned 32-dim vector,
# so the RNN receives 32 meaningful numbers per word instead of 1 useless one.
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen))
model.add(SimpleRNN(64))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=128,
          validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")
```

### 5.3 Observations

With the embedding layer the validation accuracy rises well above the 0.50
baseline. The RNN now receives dense, learnable vectors that encode word meaning
rather than arbitrary integer labels. Since the recurrent model is essentially
unchanged, this confirms that the **input representation**, not the RNN, was the
bottleneck.

---

## 6. Difference Between the Two Concepts

| Aspect | Integer Encoding | Word Embedding |
|---|---|---|
| Representation of a word | One integer index | Dense vector of many numbers |
| Numbers per word (to the RNN) | 1 | e.g. 32 |
| Dimensionality | Scalar (or huge sparse one-hot) | Small and dense |
| Captures word similarity | No | Yes |
| Introduces false magnitude | Yes (the core problem) | No |
| Learned during training | No (fixed mapping) | Yes (trainable) |
| Role in this lab | The input that failed | The fix that worked |
| IMDB SimpleRNN accuracy | ~0.50 (random) | Well above baseline |

**In short:** integer encoding gives each word a single, meaningless label,
which the RNN cannot learn from. Word embedding replaces that single label with
a vector of learnable numbers that captures meaning — directly removing integer
encoding's false-magnitude and no-similarity problems.

---

## 7. Conclusion

Integer encoding is a necessary first step because it converts words into
indices, but its raw indices are unsuitable as direct model input: they imply a
false numeric magnitude, carry no semantic similarity, and are not learnable.
Feeding a single such number per word into a SimpleRNN (`input_shape=(50, 1)`)
left the IMDB classifier stuck at chance-level accuracy. A word embedding solves
this by turning each index into a dense, learnable vector of many numbers that
captures meaning; demonstrated first on 10 sentences and then applied to IMDB,
it raised accuracy substantially without changing the recurrent model. The lab
therefore shows that a good numeric representation of text is as important as
the model architecture itself.
