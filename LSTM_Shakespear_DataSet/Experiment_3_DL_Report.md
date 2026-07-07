# Experiment 3 — Lab Report
## Character-Level Text Generation using LSTM (Recurrent Neural Networks)
**Course:** Deep Learning Lab (CS-405)
**Dataset:** Shakespeare Text (~1 Million Characters)
**Model:** Character-Level LSTM Language Model

---

## 1. Objective

To build a Recurrent Neural Network based on LSTM architecture in PyTorch that learns the sequential structure of text at the character level and generates new, coherent text sequences. The experiment demonstrates how LSTMs handle long-range dependencies in sequential data, how embedding layers learn character representations, and how temperature-controlled sampling controls the creativity of generated output.

---

## 2. Why This Approach — Why LSTM for Sequential Data?

### The Core Problem with Previous Architectures

Every model built in the previous experiments — ANN, DNN, CNN, and the binary classifier from Experiment 2 — operates on **fixed-size, independent inputs**. Feed in a vector of features, get back a prediction. Feed in an image, get back a class label. The model has no memory of what it processed before the current input.

This works perfectly for tabular data and images. It completely fails for sequential data.

Consider the sentence: *"The clouds were dark and it started to ____"*

To predict the next word ("rain"), you need to remember what was said several words ago — "clouds", "dark". The relevant context is not just the last word, it stretches back across the entire sentence. A Dense or CNN layer reading one character at a time has no mechanism to carry that context forward. It sees "\_\_\_\_" and has no idea what came before.

**Sequential data requires memory across time.** This is exactly what RNNs and LSTMs provide.

### Why Not a Simple RNN?

A basic RNN does maintain a hidden state that carries information forward — but it suffers from the **vanishing gradient problem** in a more severe form than feedforward networks. When backpropagating through hundreds of time steps, gradients shrink exponentially. The network effectively forgets anything that happened more than 10–20 steps ago. For character-level text, where a closing quotation mark must match an opening one from 200 characters earlier, this is catastrophic.

### Why LSTM?

Long Short-Term Memory networks solve the vanishing gradient problem through a gating mechanism — three learned gates that control what information to keep, what to forget, and what to output:

**Forget Gate:** decides what portion of the previous cell state to discard
```
f = σ(Wf · [h(t-1), x(t)] + bf)
```

**Input Gate:** decides what new information to store in the cell state
```
i = σ(Wi · [h(t-1), x(t)] + bi)
C̃ = tanh(Wc · [h(t-1), x(t)] + bc)
```

**Cell State Update:** combines forget and input gates
```
C(t) = f ⊙ C(t-1) + i ⊙ C̃
```

**Output Gate:** decides what to output based on the cell state
```
o = σ(Wo · [h(t-1), x(t)] + bo)
h(t) = o ⊙ tanh(C(t))
```

The cell state `C(t)` is the LSTM's long-term memory — it flows through time with only element-wise operations (no matrix multiplications), allowing gradients to flow back unchanged over hundreds of steps. This is what makes LSTM capable of learning dependencies across long sequences.

### Why Character-Level (Not Word-Level)?

The model operates on individual characters rather than words. This choice has significant advantages:

- **No vocabulary explosion:** English words number in the hundreds of thousands. Characters are only 65. A word-level model would need a 100,000-neuron output layer; ours needs 65.
- **Handles unknown words naturally:** the model can generate any word, including words it never saw during training, by composing characters.
- **Learns spelling, punctuation, and structure simultaneously:** the model must learn that 'q' is almost always followed by 'u', that sentences end with periods, that Shakespeare uses colons after character names — all from raw characters.
- **Transferable to any language or notation system:** the same architecture works on code, music notation (ABC format), DNA sequences, or any other character-based system.

### Why Shakespeare?

The lab manual used ABC music notation via the MIT `mitdeeplearning` package — a black box that hid the data loading, preprocessing, and audio conversion pipeline. We replaced it with Shakespeare text for three reasons:

1. **Full visibility:** every step from raw text to training batch is written by us — no hidden helper functions
2. **No system dependencies:** no `abcmidi`, `timidity`, GPU assertions, or Comet ML API keys required
3. **Richer structure:** Shakespeare has consistent patterns — character names in uppercase followed by colons, verse structure, dialogue — making it easier to visually evaluate generation quality

The LSTM architecture, training loop, and all core concepts are identical regardless of whether the dataset is music notation or Shakespeare text.

---

## 3. Advantages of This Approach

### Over Simple RNN
- Handles long-range dependencies — LSTM can remember context from hundreds of characters ago
- Gradient clipping (`max_norm=1.0`) combined with LSTM's cell state prevents both vanishing and exploding gradients
- Practical training convergence on sequences of length 100

### Over Feedforward Networks (ANN/DNN from previous experiments)
- Processes variable-length sequences naturally — the same model handles sequences of any length
- Shares parameters across time steps — the same LSTM weights process every character position, far more efficient than a fixed-size Dense network
- Learns temporal patterns — the order of characters matters; an ANN treats all positions as independent

### Over CNN (Experiment 15/16 from ML Lab)
- CNNs can detect local patterns in sequences (n-grams) but have fixed receptive fields
- LSTM's hidden state accumulates context across the entire sequence regardless of length
- For text generation specifically, LSTM's autoregressive generation (one character at a time, feeding output back as next input) is natural; CNN cannot do this

### The Embedding Layer Advantage
Rather than one-hot encoding characters (65-dimensional sparse vectors), the embedding layer maps each character to a dense 128-dimensional vector. These vectors are **learned** — after training, characters used in similar contexts have similar embeddings. The PCA visualization in Cell 13 confirms this: uppercase letters cluster together, lowercase cluster together, punctuation forms its own group. The embedding layer compresses sparse categorical data into rich, meaningful dense representations.

### Temperature Sampling — Industry-Grade Generation
Most educational implementations use argmax — always picking the most probable next character. This produces repetitive, deterministic output. Temperature sampling introduces controlled stochasticity:

```
P(character) = softmax(logits / temperature)
```

- **Low temperature (0.3):** probability distribution is sharper — confident, repetitive, more structured
- **High temperature (1.3):** distribution is flatter — more diverse, sometimes incoherent

This is exactly the technique used in production LLMs including GPT and Claude. Understanding it at the character level builds intuition for how sampling works in large language models.

---

## 4. Steps Taken (Overview)

1. Downloaded Shakespeare text dataset directly via URL — no package dependencies
2. Built vocabulary and character-to-index mappings — 65 unique characters
3. Visualized character frequency distribution — confirmed space as most frequent
4. Implemented `get_batch()` — creates input/target pairs shifted by one character
5. Defined `CharLSTM` — Embedding → 2-layer LSTM → Dropout → Linear
6. Demonstrated the embedding layer output before training
7. Set up CrossEntropyLoss, Adam optimizer, ReduceLROnPlateau scheduler
8. Verified untrained model loss equals `log(vocab_size)` — confirms random baseline
9. Trained for 3000 iterations with gradient clipping and checkpointing
10. Plotted smoothed training loss and perplexity curves
11. Implemented temperature-controlled `generate_text()` function
12. Generated text at temperatures 0.3, 0.7, 1.0, 1.3 — compared quality
13. Visualized the learned embedding space using PCA
14. Printed full summary of architecture, training, and results

---

## 5. Cell-by-Cell Walkthrough

---

### Cell 1 — Imports and Device Setup

Set random seeds for NumPy, PyTorch, and Python's `random` module — all three are necessary because TensorFlow/PyTorch maintain independent random number generators. Detected GPU availability. Running on GPU reduces training time from ~25 minutes to ~4 minutes for 3000 iterations.

---

### Cell 2 — Load Dataset

Downloaded ~1 million characters of Shakespeare directly from a GitHub URL using `requests`. No package installation, no API key, no file management — the entire dataset is in memory as a Python string after one line. Confirmed total character count and previewed the first 500 characters to verify the download.

---

### Cell 3 — Vocabulary and Encoding

**What was done:** Extracted all unique characters using `sorted(set(text))` — 65 characters including uppercase, lowercase, digits, punctuation, space, and newline. Built bidirectional mappings: `char2idx` (character → integer) and `idx2char` (integer → character). Encoded the entire text as a NumPy integer array of shape `(1,115,394,)`.

**Why integer encoding and not one-hot?**
One-hot would make each character a 65-dimensional sparse vector — 98.5% zeros. The embedding layer in the model takes integer indices directly and learns dense representations internally. Storing the text as integers is both memory-efficient and the correct input format for `nn.Embedding`.

---

### Cell 4 — Character Frequency Visualization

Plotted the 20 most frequent characters as a bar chart. Space was the most frequent — confirming the text is word-based with natural spacing. Newline appeared frequently — reflecting Shakespeare's verse and dialogue structure. This EDA step confirms the dataset loaded correctly and gives intuition for what patterns the LSTM must learn: spaces after every word, newlines after every line, uppercase only at sentence starts and character names.

---

### Cell 5 — Sequence Batch Creation

**The key concept:** Character-level language modeling is framed as next-character prediction. For every input sequence of 100 characters, the target is the same sequence shifted one position to the right.

```
Input  : "To be or not to be, that is the questio"
Target : "o be or not to be, that is the question"
```

The model must predict character `t+1` given all characters up to `t`. This transforms an unsupervised text dataset into a supervised learning problem — the labels are derived from the data itself. `get_batch()` samples 64 random starting positions, extracts 100-character windows, and returns tensors ready for the GPU.

---

### Cell 6 — LSTM Model Definition

**Architecture:**

```
Embedding(65, 128)     → converts character indices to 128-dim dense vectors
LSTM(128, 512, layers=2) → processes sequence, maintains hidden + cell state
Dropout(0.3)           → regularization between LSTM and output
Linear(512, 65)        → maps hidden state to vocabulary scores
```

**Total parameters: ~3.8 million**

The `forward()` method takes a batch of integer sequences and an optional hidden state. During training, hidden state is not passed between batches — each batch is treated as independent. During generation, hidden state is passed between steps — the model remembers what it generated previously.

**Why 2 LSTM layers?**
Stacking two LSTM layers allows the second layer to learn patterns in the first layer's output — higher-level temporal abstractions. The first layer might learn character-level patterns (common letter combinations, word endings). The second layer learns word-level and phrase-level patterns (sentence structure, dialogue conventions).

---

### Cell 7 — Embedding Layer Demonstration

Before training, the embedding vectors for 'H' and 'e' are random — they have no meaningful relationship. After training, characters used in similar contexts will have similar embedding vectors. This cell demonstrated the shape transformation: 5 integer inputs → `(1, 5, 128)` embedded output. Each character becomes a 128-dimensional point in a learned semantic space.

---

### Cell 8 — Loss Function, Optimizer, Baseline Verification

**Loss function — CrossEntropyLoss:**
The model outputs 65 scores (logits) for each character position. CrossEntropyLoss measures how far the predicted distribution is from the true one-hot target. Before training, the untrained model's loss should equal `log(vocab_size) = log(65) ≈ 4.17` — this is the loss of a perfectly random model assigning equal probability to all 65 characters. Verifying this confirms the model initialized correctly and the loss function is working.

**Perplexity:**
```
Perplexity = e^(loss)
```
Untrained perplexity ≈ 65 — the model is as confused as randomly picking from 65 characters. A well-trained model achieves perplexity < 10, meaning it is effectively choosing from fewer than 10 equally likely characters at each step — it has learned that most characters are very unlikely in any given context.

**ReduceLROnPlateau scheduler:** monitors validation loss and halves the learning rate if it does not improve for 3 consecutive log checkpoints. This allows the model to take large steps early in training and fine-tune with smaller steps later.

---

### Cell 9 — Training Loop

**The complete PyTorch training loop — 5 essential steps:**

```
1. optimizer.zero_grad()                    — clear stale gradients
2. logits, _ = model(x_batch)              — forward pass
3. loss = compute_loss(logits, y_batch)    — compute loss
4. loss.backward()                          — backpropagation through time (BPTT)
5. optimizer.step()                         — update all 3.8M parameters
```

**Gradient clipping — the most important RNN-specific addition:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Without this, RNNs suffer from **exploding gradients** — when gradients backpropagate through hundreds of time steps, they can multiply to astronomically large values and destabilize training in a single update. Clipping rescales the entire gradient vector if its norm exceeds 1.0, keeping updates bounded and training stable.

**Checkpointing:** saved the model weights whenever average loss improved. This ensures the best model is preserved even if the final iterations overfit or degrade.

Training for 3000 iterations produced a consistent loss decrease from ~4.17 (random) to approximately 1.3–1.5, corresponding to perplexity dropping from 65 to approximately 4–5.

---

### Cell 10 — Training Curves

Plotted two panels: smoothed training loss and perplexity over 3000 iterations.

**What the curves showed:**
- Loss dropped steeply in the first 500 iterations — the model quickly learned basic patterns (spaces between words, common letter combinations)
- Loss continued decreasing more gradually from iterations 500–2000 — learning word-level and phrase-level patterns
- Loss plateaued near the end — the model had extracted most learnable patterns from 3000 iterations

The smoothing function (`np.convolve` with window=50) removed the batch-to-batch noise to reveal the true learning trend. Raw loss is noisy because each batch samples different parts of the text; the smoothed curve shows the underlying convergence.

**Perplexity curve:** mirrored the loss curve (since perplexity = e^loss) but made the improvement more interpretable — from 65 (random) down to approximately 4–5 (the model is effectively choosing between 4–5 equally likely characters at each step).

---

### Cell 11 — Temperature Sampling Function

**The generation loop — autoregressive generation:**

```
1. Encode seed string → integer tensor
2. Prime hidden state by running seed through the model
3. Feed last character of seed as first input
4. For each generation step:
   a. Forward pass → logits for next character
   b. Scale logits by temperature
   c. Softmax → probability distribution
   d. Sample from distribution (multinomial, not argmax)
   e. Feed sampled character back as next input
```

The key design decision is **multinomial sampling** rather than argmax. Argmax always picks the single most probable character — deterministic and repetitive. Multinomial sampling picks randomly according to the probability distribution — maintains diversity while still being guided by what the model learned.

Loaded the best saved checkpoint to ensure generation uses the model from the optimal training point, not the final iteration which may have been slightly worse.

---

### Cell 12 — Generation at Different Temperatures

**Temperature = 0.3 (Conservative):**
Output was highly structured and repetitive. The model confidently reproduced common Shakespeare patterns — character name: dialogue format, common phrases. The same few words appeared repeatedly. Clearly recognizable as Shakespeare-like but monotonous.

**Temperature = 0.7 (Balanced — best quality):**
Output showed the best balance of structure and variety. Sentences were mostly grammatical, character names followed correct format, verse structure appeared. This is the recommended temperature for quality generation.

**Temperature = 1.0 (Standard):**
More variety in vocabulary but some grammatical breakdown. Less repetition but occasional nonsensical character combinations.

**Temperature = 1.3 (Creative):**
High diversity — many unusual word choices and constructions. Occasional real words mixed with invented ones. The model was taking more risks in its sampling, sometimes coherent and sometimes not.

**The key insight:** temperature is a trade-off dial between coherence and creativity. This is not a training parameter — it is a generation-time parameter. The same trained model produces completely different outputs at different temperatures without any retraining.

---

### Cell 13 — Embedding Space Visualization

Applied PCA to reduce the 128-dimensional embedding vectors to 2D and plotted all 65 characters colored by type (uppercase red triangles, lowercase blue circles, digits green squares, punctuation gray crosses).

**What the visualization revealed:**
- Uppercase letters formed a loose cluster — the model learned they appear in similar contexts (sentence starts, character names)
- Lowercase letters clustered separately — the most common character type, appearing in all word bodies
- Punctuation scattered in its own region — each punctuation mark has a distinctive context (period ends sentences, comma separates clauses)
- This spatial separation is entirely learned — no one told the model that 'A' and 'B' are similar; it discovered this from patterns in the training text

This directly connects to the PCA theory from Group 4 of the ML Lab — the same dimensionality reduction technique reveals structure in the LSTM's learned representations.

---

### Cell 14 — Final Summary

Printed complete experiment summary confirming architecture, parameter count, training configuration, and results. Final perplexity of ~4–5 means the model reduced its effective choice at each character step from 65 (random) to approximately 4–5 — a dramatic improvement demonstrating successful sequence learning.

---

## 6. Results

| Metric | Value |
|---|---|
| Vocabulary size | 65 unique characters |
| Model parameters | ~3.8 million |
| Training iterations | 3,000 |
| Starting perplexity | ~65 (random baseline) |
| Final perplexity | ~4–5 |
| Loss reduction | ~4.17 → ~1.4 |
| Best temperature | 0.7 |
| Generation quality | Recognizable Shakespeare structure and vocabulary |

---

## 7. What We Learned from This Lab

**1. Sequential memory is architecturally different from feedforward memory.**
An ANN or CNN has no concept of "before" and "after". An LSTM's hidden state and cell state give it genuine memory — it can learn that an open parenthesis from 50 characters ago needs a close parenthesis later. This is a fundamentally different kind of computation.

**2. The vanishing gradient problem has a structural solution.**
We did not solve vanishing gradients with a trick (like ReLU solved it for feedforward networks). LSTM solves it architecturally — the cell state flows through time with only addition and element-wise multiplication, providing a gradient highway that does not attenuate over time.

**3. Gradient clipping is mandatory for RNNs.**
Feedforward networks rarely need gradient clipping. RNNs almost always do. Backpropagation through time multiplies gradients across hundreds of steps — exploding gradients can destroy training in a single iteration. `clip_grad_norm_(max_norm=1.0)` is standard practice for every RNN/LSTM implementation.

**4. Perplexity is the correct evaluation metric for language models.**
Accuracy is meaningless here — there is no single right next character, just more or less probable ones. Perplexity measures how surprised the model is on average. A perplexity of 5 means the model is as uncertain as if it were choosing uniformly from 5 options — far better than the 65-option random baseline.

**5. Temperature sampling bridges the lab and production LLMs.**
The `temperature` parameter in `generate_text()` is exactly the same parameter exposed in the APIs of GPT-4, Claude, and every major LLM. Understanding it at the character level — seeing concretely how dividing logits by a small number sharpens the distribution — gives genuine intuition for what temperature means in large models.

**6. Embeddings learn meaningful representations without supervision.**
The PCA visualization showed that the embedding layer, trained only to predict next characters, spontaneously organized characters into meaningful groups — uppercase together, lowercase together, punctuation separate. This is the core insight of representation learning: optimizing for a prediction task forces the network to learn useful internal representations as a side effect.

**7. The full PyTorch training loop is transparent and controllable.**
Keras `.fit()` hides everything — loss computation, gradient flow, weight updates. The manual PyTorch loop (`zero_grad → forward → loss → backward → step`) exposes every step. This transparency is why researchers prefer PyTorch — you can intervene at any point, modify gradients, implement custom losses, or add unusual update rules that Keras makes impossible.

---

## 8. Connection to Real-World Applications

The CharLSTM built in this experiment is conceptually identical to the first generation of serious language models. GPT-1 (2018) used essentially this architecture — a deep transformer instead of LSTM, but the same character/token prediction objective, the same cross-entropy loss, the same temperature sampling. Understanding this experiment means understanding the foundation that all modern LLMs are built on.

Beyond text, the same LSTM architecture is used for:
- **Music generation:** replace Shakespeare characters with musical notes (the original lab manual's intent with ABC notation)
- **Code generation:** replace text with source code characters
- **DNA sequence modeling:** replace characters with nucleotides (A, T, G, C)
- **Time series forecasting:** replace characters with discretized sensor readings
- **Speech recognition:** replace characters with phoneme labels

The architecture is domain-agnostic — what changes is the vocabulary and the data, not the model.

---

## 9. Limitations and Possible Improvements

**Current limitations:**
- 3000 training iterations is relatively short — perplexity is still decreasing at the end. Training for 10,000+ iterations would further improve generation quality
- Fixed sequence length of 100 — the model cannot learn dependencies longer than 100 characters within a single training window
- No beam search — temperature sampling is simple but beam search (maintaining top-k candidate sequences) produces more coherent longer-range text
- Character-level is now superseded by subword tokenization (BPE, WordPiece) in modern LLMs — these provide a better trade-off between vocabulary size and sequence length

**Possible improvements:**
- **Increase hidden size to 1024** — more capacity to memorize Shakespeare patterns
- **Train for 10,000 iterations** — loss is still decreasing at 3000
- **Add beam search generation** — maintain top 5 candidate sequences, pick the most probable complete sequence
- **Implement teacher forcing ratio** — gradually reduce reliance on true labels during training to better match generation conditions
- **Subword tokenization** — use BPE to tokenize at the word-piece level for better vocabulary efficiency

---

## 10. Conclusion

Experiment 3 introduced the most architecturally distinct model in the Deep Learning Lab — the LSTM, which adds temporal memory to neural computation. Unlike every previous model (ANN, DNN, CNN) that processes fixed-size independent inputs, the LSTM processes sequences of arbitrary length while maintaining a hidden state that carries context across time.

The Shakespeare character-level language model demonstrated all core LSTM concepts: embedding layers that learn character representations, two-layer LSTM that captures both character-level and phrase-level patterns, gradient clipping that prevents exploding gradients, and temperature sampling that controls generation creativity.

Most importantly, the experiment made the connection from academic exercise to production systems explicit — the same `temperature` parameter, the same cross-entropy objective, the same autoregressive generation loop that powers GPT and Claude is demonstrated here at character scale. The difference between this experiment and a large language model is scale (3.8M vs 175B parameters) and tokenization (characters vs subwords) — not architecture or training philosophy.

---

*Experiment 3 Lab Report | Deep Learning Lab (CS-405)*
*Model: CharLSTM | Dataset: Shakespeare | Framework: PyTorch*
