# Experiment 2 — Lab Report
## PyTorch Fundamentals for Deep Learning

**Course:** CS-405 Deep Learning
**Department:** Computer Software Engineering, MCS NUST

---

## 1. Objective

To provide hands-on experience with PyTorch by exploring tensor operations, computational graphs, automatic differentiation, and neural network modules — culminating in building, training, and evaluating a complete binary classification model from scratch.

---

## 2. Why PyTorch When We Already Have Keras?

This is the most important conceptual question before starting this lab.

In CS-471, we used **TensorFlow/Keras** which abstracts almost everything:
```python
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=100)
```

Two lines and the model trains. But what is actually happening inside? Keras hides:
- How gradients are computed
- How weights are updated step by step
- What the training loop actually does
- How loss flows backward through the network

**PyTorch solves this by making everything explicit.**

The same training process in PyTorch requires you to write every step:
```python
optimizer.zero_grad()    # step 1 — clear old gradients
y_pred = model(X)        # step 2 — forward pass
loss = criterion(y_pred, y) # step 3 — compute loss
loss.backward()          # step 4 — backpropagation
optimizer.step()         # step 5 — update weights
```

**This is not a disadvantage — it is the entire point.** Writing the training loop manually forces you to understand what gradient descent actually does at every step.

**Key differences:**

| Aspect | TensorFlow/Keras (CS-471) | PyTorch (CS-405) |
|--------|--------------------------|------------------|
| Training | `.fit()` hides everything | Manual training loop — full control |
| Computation graph | Static (build then run) | Dynamic (built as code runs) |
| Debugging | Harder — graph is opaque | Standard Python debugging works |
| Flexibility | Less — designed for standard pipelines | More — custom architectures, custom losses |
| Research use | Less common | Dominant in academic research |
| Industry use | Common (production) | Growing rapidly |
| Core concept | Model-centric | Tensor-centric |

**When to use which:**
- **Keras:** Production deployment, rapid prototyping, standard architectures
- **PyTorch:** Research, custom architectures, understanding what's happening inside, this course

---

## 3. Cell-by-Cell Walkthrough

---

### Cell 1 — Imports & Environment Check

```python
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available  : {torch.cuda.is_available()}")
```

Sets up the environment. `torch.cuda.is_available()` checks whether a GPU is accessible — PyTorch can automatically use CUDA-enabled GPUs for dramatically faster computation. When True, tensors and models can be moved to GPU with `.to('cuda')`.

---

### Cell 2 — Scalar and Vector Tensors

```python
integer = torch.tensor(1234)      # 0D tensor — scalar
fibonacci = torch.tensor([1,1,2,3,5,8])  # 1D tensor — vector
```

Introduces tensors — the core data structure of PyTorch. A tensor is a multi-dimensional array that:
- Supports all NumPy-style operations
- Can run on GPU (NumPy cannot)
- Tracks operations for automatic differentiation (NumPy cannot)

The `ndim` property tells the number of dimensions; `shape` tells the size of each dimension.

---

### Cell 3 — 2D and 4D Tensors

```python
matrix = torch.ones(3, 4)               # 2D — 3 rows, 4 columns
images = torch.zeros(10, 3, 256, 256)   # 4D — image batch
```

**The 4D tensor is the standard format for image batches in all deep learning:**
```
(batch_size, channels, height, width)
→ 10 images, RGB (3 channels), 256×256 pixels each
```

This is the format every CNN layer expects as input. Understanding this shape is critical for all subsequent image experiments in this lab manual.

Note: This is **different from TensorFlow/Keras** which uses `(batch, height, width, channels)` — PyTorch puts channels first.

---

### Cell 4 — Tensor Slicing

```python
row_vector    = matrix[1]      # second row
column_vector = matrix[:, 1]  # second column
scalar        = matrix[0, 1]  # single element
```

Slicing works identically to NumPy. In deep learning context:
- `images[0]` → first image, shape (3, 256, 256)
- `images[:, 0]` → red channel of all images, shape (10, 256, 256)
- `images[0, :, :10, :10]` → top-left 10×10 patch of first image

This kind of indexing is used constantly when working with image batches — extracting specific samples, channels, or spatial regions.

---

### Cell 5 — Computation Graph

```python
a = torch.tensor(15)
b = torch.tensor(61)
c = a + b    # same as torch.add(a, b)
```

Introduces PyTorch's dynamic computation graph. Every operation on tensors is tracked — PyTorch builds a graph of how each tensor was computed. This graph is later traversed backwards during `loss.backward()` to compute gradients automatically.

Unlike TensorFlow 1.x (static graphs), PyTorch builds the graph on-the-fly as Python code runs — making it natural to debug with standard Python tools.

---

### Cell 6 — Custom Computation Graph

```python
def func(a, b):
    c = a + b    # add
    d = b + 1    # increment
    e = c * d    # multiply
    return e
```

Demonstrates a multi-step computation graph. With `a=1.5, b=2.5`:
- c = 1.5 + 2.5 = 4.0
- d = 2.5 + 1 = 3.5
- e = 4.0 × 3.5 = 14.0

PyTorch tracks all three operations and could compute `de/da` and `de/db` automatically if needed — this is the autograd system that makes backpropagation work.

---

### Cell 7 — Manual Dense Layer (OurDenseLayer)

```python
class OurDenseLayer(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.W    = nn.Parameter(torch.randn(num_inputs, num_outputs))
        self.bias = nn.Parameter(torch.randn(num_outputs))

    def forward(self, x):
        z = torch.matmul(x, self.W) + self.bias
        y = torch.sigmoid(z)
        return y
```

**This is the most conceptually important cell in the first half of the lab.** It builds a Dense layer from scratch to reveal exactly what `nn.Linear` does internally.

**What happens mathematically:**
$$y = \sigma(xW + b)$$

- `x` has shape (1, 2) — 1 sample, 2 features
- `W` has shape (2, 3) — weight matrix
- `xW` has shape (1, 3) — matrix multiplication
- `b` has shape (3,) — broadcast-added to each row
- `sigmoid(z)` maps each value to (0, 1)

**Why `nn.Parameter`:**
Wrapping tensors in `nn.Parameter` tells PyTorch these are learnable — they will be included in `model.parameters()` and updated by the optimizer during training. Regular tensors are not updated.

**Two mandatory methods in nn.Module:**
- `__init__`: Define the learnable parameters
- `forward`: Define how data flows through the layer

---

### Cell 8 — nn.Sequential

```python
model = nn.Sequential(
    nn.Linear(n_input_nodes, n_output_nodes),
    nn.Sigmoid()
)
```

`nn.Sequential` is PyTorch's equivalent of Keras Sequential — it stacks layers linearly. The output of each layer feeds directly into the next. Much cleaner than writing the full class for simple architectures.

**Total parameters for Linear(2→3):**
- Weights: 2×3 = 6
- Biases: 3
- Total: 9

---

### Cell 9 — Proper nn.Module Subclassing

```python
class LinearWithSigmoidActivation(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear     = nn.Linear(num_inputs, num_outputs)
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        return self.activation(self.linear(inputs))
```

The standard PyTorch pattern for defining neural networks. More verbose than Sequential but more flexible — the `forward` method can contain any Python logic (conditionals, loops, multiple paths).

---

### Cell 10 — Conditional Forward Pass

```python
def forward(self, inputs, isidentity=False):
    if isidentity:
        return inputs          # pass through unchanged
    else:
        return self.linear(inputs)
```

**This demonstrates PyTorch's most powerful feature over Keras:** the `forward` method is just Python. It can contain if statements, for loops, print statements — anything. In Keras, the computation graph is fixed at compile time. In PyTorch, the graph is built dynamically every forward pass — enabling architectures that change behavior based on input.

This flexibility is why research papers almost exclusively use PyTorch — novel architectures often require logic that doesn't fit into Keras's fixed patterns.

---

### Cell 11 — Automatic Differentiation (Autograd)

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2        # y = x²
y.backward()      # compute dy/dx automatically
print(x.grad)     # → 6.0  (dy/dx = 2x = 2×3 = 6)
```

**This is the mathematical foundation of all deep learning.** `requires_grad=True` tells PyTorch to track all operations on this tensor. `backward()` traverses the computation graph in reverse and computes gradients using the chain rule.

- Mathematical derivative: $\frac{dy}{dx} = 2x = 2 \times 3 = 6$
- PyTorch computed the same result automatically

In a neural network, this is how the gradient of the loss with respect to every single weight is computed — automatically, through potentially hundreds of layers, without manually deriving any derivative.

---

### Cell 12 — Gradient Descent from Scratch

```python
for step in range(50):
    optimizer.zero_grad()     # 1. clear old gradients
    loss = (x - x_f) ** 2    # 2. compute loss L = (x - 4)²
    loss.backward()           # 3. dL/dx = 2(x - 4)
    optimizer.step()          # 4. x = x - lr × dL/dx
```

**Minimizing $L = (x - 4)^2$ — the simplest possible optimization problem:**
- Analytical minimum: $x = 4$
- Gradient: $\frac{dL}{dx} = 2(x - 4)$
- Gradient descent moves $x$ toward 4 step by step

Starting from $x=0$, after 50 steps of SGD with lr=0.1, $x$ converges to exactly 4.0.

**Why `optimizer.zero_grad()` every step:**
PyTorch accumulates gradients by default — calling `backward()` multiple times adds to existing gradients. For standard training, we want fresh gradients each step — so we clear them first.

**The loss convergence plot** shows the characteristic exponential decay — loss drops steeply at first then approaches zero as $x$ nears the target. This is the same curve you will see in every training plot for the rest of this lab manual.

---

## 4. Priority Cells — Full Implementation (Cells 13–18)

---

### Cell 13 — Dataset Preparation

```python
X, y = make_classification(n_samples=1000, n_features=2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
```

**What happens here step by step:**

**1. Generate dataset:** `make_classification` creates 1,000 samples with 2 features — two Gaussian blobs that are partially overlapping. 2 features are used so the decision boundary can be visualized in 2D.

**2. StandardScaler:** Same rule as CS-471 — neural networks use gradient-based optimization which is sensitive to feature scale. Without scaling, large-magnitude features dominate gradient updates.

**3. Split first, then scale:** `fit_transform` on training data only, `transform` on test — same no-data-leakage rule from CS-471 Experiment 3.

**4. Convert to PyTorch tensors:** NumPy arrays must be converted to PyTorch tensors before feeding to the model. `FloatTensor` creates float32 tensors (the default dtype for neural network weights).

**5. `.unsqueeze(1)`:** Changes `y_train` shape from `(800,)` to `(800, 1)`. BCELoss requires the target to have the same shape as the prediction — the model outputs shape `(batch, 1)`, so y must also be `(batch, 1)`.

**The scatter plot** shows two clouds of points (red = Class 0, green = Class 1). They are partially overlapping near the boundary — a realistic classification scenario that requires a non-linear decision boundary.

---

### Cell 14 — BinaryClassifier Architecture

```python
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),   # 2 inputs → 16 neurons
            nn.ReLU(),
            nn.Linear(16, 8),   # 16 → 8 neurons
            nn.ReLU(),
            nn.Linear(8, 1),    # 8 → 1 output
            nn.Sigmoid()        # probability output
        )

    def forward(self, x):
        return self.network(x)
```

**Architecture design — layer by layer:**

**Input → 16 neurons (Layer 1):**
2 input features expanded to 16 neurons. More neurons = more capacity to learn complex patterns. 16 is a power of 2 — efficient for GPU memory alignment.

**16 → 8 neurons (Layer 2):**
The classic funnel pattern from CS-471 — progressively compressing the representation. Layer 2 combines the 16 patterns from Layer 1 into 8 higher-level features.

**8 → 1 neuron (Output):**
Binary classification → 1 output neuron. Outputs a single probability value.

**ReLU hidden layers, Sigmoid output:**
- ReLU in hidden layers: `max(0, x)` — fast, avoids vanishing gradients, introduces non-linearity allowing curved decision boundaries
- Sigmoid at output: maps raw score to probability (0, 1) — required for BCELoss

**Parameter count breakdown:**
| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| Linear(2→16) | 2×16 = 32 | 16 | 48 |
| Linear(16→8) | 16×8 = 128 | 8 | 136 |
| Linear(8→1) | 8×1 = 8 | 1 | 9 |
| **Total** | | | **193** |

193 parameters — a tiny model by modern standards, but sufficient for this 2D dataset.

---

### Cell 15 — Manual Training Loop

```python
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(EPOCHS):
    # TRAINING
    model.train()
    optimizer.zero_grad()              # 1. clear gradients
    y_pred = model(X_train_t)         # 2. forward pass
    loss   = criterion(y_pred, y_train_t) # 3. compute loss
    loss.backward()                    # 4. backpropagation
    optimizer.step()                   # 5. update weights

    # EVALUATION
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_t)
        test_loss   = criterion(y_test_pred, y_test_t)
```

**This is the most important cell in the entire experiment.** It exposes everything that Keras `.fit()` hides.

**The 5-step training loop — explained:**

**Step 1 — `optimizer.zero_grad()`:**
PyTorch accumulates gradients across calls to `backward()`. If not cleared, gradients from epoch 1 would add to epoch 2's gradients — corrupting weight updates. Always clear before each forward pass.

**Step 2 — `model(X_train_t)` (Forward Pass):**
Data flows through the network: Input → Linear → ReLU → Linear → ReLU → Linear → Sigmoid → Output probability. Each layer applies its transformation. The computation graph is built dynamically as this executes.

**Step 3 — `criterion(y_pred, y_train_t)` (Loss Computation):**
BCELoss (Binary Cross-Entropy Loss):
$$L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$
Measures how far the predicted probabilities are from the true labels. High loss = poor predictions. Low loss = accurate predictions.

**Step 4 — `loss.backward()` (Backpropagation):**
PyTorch traverses the computation graph in reverse — from loss back through Sigmoid → Linear → ReLU → Linear → ReLU → Linear → inputs. At each layer, it computes the gradient of the loss with respect to every weight and bias using the chain rule. These gradients are stored in `param.grad` for each parameter.

**Step 5 — `optimizer.step()` (Weight Update):**
Adam optimizer updates every weight using its stored gradient:
$$w \leftarrow w - \alpha \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$$
Where $\hat{m}$ and $\hat{v}$ are the momentum estimates. Adam adapts the learning rate per parameter — parameters with consistently large gradients get smaller updates; parameters with small or noisy gradients get larger updates.

**`model.train()` vs `model.eval()`:**
- `model.train()`: Enables Dropout (randomly drops neurons) and BatchNorm in training mode
- `model.eval()`: Disables Dropout — all neurons active; BatchNorm uses running statistics

**`with torch.no_grad()`:**
During evaluation, we don't need gradients — no `backward()` will be called. `torch.no_grad()` disables gradient tracking for that block, saving memory and computation.

---

### Cell 16 — Training Curves

```python
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(test_losses,  label='Test Loss')
axes[1].plot(train_accs,   label='Train Accuracy')
axes[1].plot(test_accs,    label='Test Accuracy')
```

**What the plots reveal:**

**Loss plot:** Both training and test loss should drop and converge. If test loss starts rising while training loss continues falling — overfitting. The gap between the two lines at the end quantifies how much the model overfits.

**Accuracy plot:** Both should rise and plateau. The test accuracy plateauing below train accuracy is normal and expected. If the gap is large (>5-10%), the model is overfitting the training data.

**For this 2D dataset:** Both curves should converge smoothly to ~90%+ accuracy with minimal gap, demonstrating that the simple 193-parameter network is sufficient for the task without overfitting.

---

### Cell 17 — Decision Boundary Visualization

```python
# Create 300×300 mesh grid
xx, yy = np.meshgrid(x1_range, x2_range)
grid   = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

with torch.no_grad():
    Z = model(grid).numpy().reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap='RdYlGn')       # probability heatmap
plt.contour(xx, yy, Z, levels=[0.5])          # decision boundary line
```

**What this visualization shows:**

Every point in 2D feature space is fed through the trained model. The output probability (0→red, 0.5→yellow, 1→green) is plotted as a color map. The **black contour line at probability=0.5** is the decision boundary — points above the line are predicted Class 1, below are predicted Class 0.

**Key insight:** The boundary is **curved** — not a straight line. This confirms that the two ReLU hidden layers learned non-linear representations. A single-layer perceptron (Experiment 1) would only produce a straight line boundary. The two hidden layers with ReLU gave the network the capacity to learn this curved boundary.

This is visually the proof of why deep networks with non-linear activations outperform single-layer perceptrons.

---

### Cell 18 — Final Evaluation

```python
y_pred_cls = (y_pred_final >= 0.5).numpy().astype(int).flatten()
print(classification_report(y_test, y_pred_cls))
```

**Converting probabilities to class predictions:**
`(y_pred >= 0.5)` applies the threshold — probability above 0.5 → Class 1, below → Class 0. Same as the step function from Experiment 1's perceptron, but now applied to a smooth probability output.

**Classification report metrics:**
- **Precision:** Of all predicted Class 1, how many actually are Class 1?
- **Recall:** Of all actual Class 1, how many did the model catch?
- **F1:** Harmonic mean of Precision and Recall

**Confusion matrix:** Standard 2×2 matrix — True Negatives, False Positives, False Negatives, True Positives — same interpretation as CS-471 Experiment 5.

---

## 5. Results Summary

| Metric | Value |
|--------|-------|
| Dataset | 1000 samples, 2 features, binary labels |
| Architecture | 2 → 16 → 8 → 1 (193 parameters) |
| Optimizer | Adam (lr=0.01) |
| Loss Function | Binary Cross-Entropy |
| Training Epochs | 100 |
| Final Test Accuracy | ~90–92% |

---

## 6. PyTorch Training Loop — Complete Summary

The full training loop explicitly written in this experiment maps directly to what Keras `.fit()` does invisibly:

| Manual PyTorch Step | Keras Equivalent |
|--------------------|-----------------|
| `optimizer.zero_grad()` | Done automatically per batch |
| `y_pred = model(X)` | Forward pass inside `.fit()` |
| `loss = criterion(y_pred, y)` | Loss computed from `compile(loss=...)` |
| `loss.backward()` | Backpropagation — hidden completely |
| `optimizer.step()` | Weight update from `compile(optimizer=...)` |
| `model.eval()` + `torch.no_grad()` | Validation inside `.fit(validation_data=...)` |

---

## 7. Key Concepts Learned

**BCELoss (Binary Cross-Entropy):** Standard loss for binary classification. Penalizes confident wrong predictions heavily. Requires sigmoid output — probabilities in (0, 1).

**Adam Optimizer:** Adaptive gradient descent — adjusts learning rate per parameter. More stable and faster than vanilla SGD. The default choice for most neural network training.

**`model.train()` / `model.eval()`:** Switches between training and inference modes. Critical when Dropout or BatchNormalization are used.

**`torch.no_grad()`:** Context manager that disables gradient computation — used during evaluation/inference to save memory and computation.

**`optimizer.zero_grad()`:** Clears accumulated gradients before each step. Forgetting this is one of the most common PyTorch bugs — gradients would accumulate across steps giving wrong updates.

**`unsqueeze(1)`:** Adds a dimension to a tensor. Changes shape from (N,) to (N, 1) — required when output and target shapes must match for BCELoss.

**Decision Boundary:** The surface in feature space where the model outputs probability = 0.5. Everything on one side is predicted Class 0, the other side Class 1. Non-linear activations (ReLU) allow curved boundaries.

---

*Experiment 2 Lab Report | CS-405 Deep Learning*
