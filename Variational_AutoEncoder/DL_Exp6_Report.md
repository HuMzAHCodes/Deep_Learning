# Experiment 6 — Lab Report
## Generating Novel Fashion Images using Variational Autoencoders (VAE)

**Course:** CS-405 Deep Learning
**Dataset:** Fashion MNIST (via torchvision)
**Framework:** PyTorch

---

## 1. Objective

To design, implement, and evaluate a Convolutional Variational Autoencoder (CNN-VAE) that:
- Learns a continuous, structured latent space from Fashion MNIST images
- Generates new, realistic clothing images never seen during training
- Demonstrates class-conditional generation by sampling around class-specific latent regions
- Evaluates generation quality using pixel-level statistical comparison (FID-inspired metric)

---

## 2. Why This Architecture? — Motivation and Design Decisions

### 2.1 Why Not a Standard Autoencoder?

In Experiment 5, we implemented a standard Autoencoder for image denoising. It worked well for reconstruction but had a fundamental limitation: **it cannot generate new data**.

A standard AE encodes each image to a single fixed point in latent space. The latent space is completely unorganized — similar images may map to distant points, and random points sampled from the space may decode into meaningless noise. There is no structure that allows controlled generation.

**The VAE solves this** by encoding each image not as a point but as a **probability distribution** — a Gaussian centered at $\mu$ with spread $\sigma$. The decoder learns to decode any sample from this distribution into a valid image. Because distributions from different inputs overlap in the latent space, the space becomes continuous — you can move smoothly through it and generate coherent new images at every point.

### 2.2 Why CNN Instead of Dense Layers (Lab Manual's Approach)?

The lab manual uses Dense layers for both encoder and decoder:
```
Input(784) → Dense(512) → Dense(2) → mu, log_sigma
```

This approach has three critical problems:

**1. Flattening destroys spatial information:**
Fashion MNIST images are 28×28 spatial grids. Flattening to 784 loses all neighborhood relationships between pixels — the model treats pixel[0][0] and pixel[0][1] as completely unrelated. As established in CS-471 Experiment 14 (CNN), spatial relationships are exactly what makes image features meaningful.

**2. Parameter inefficiency:**
A Dense(784→512) layer has 784×512 = 400,000+ parameters just for the first layer. A Conv2D(1→32, 3×3) achieves similar feature extraction with only 288 parameters — because the same filter scans the entire image (weight sharing).

**3. Poor reconstruction quality:**
Dense decoders produce blurry, low-quality reconstructions because they must reconstruct spatial structure from a flat vector without any guidance about what pixels are neighbors.

**The CNN VAE solves all three:**
- Conv2D layers preserve spatial structure throughout
- Shared weights dramatically reduce parameters
- ConvTranspose2D decoder reconstructs spatial detail layer by layer

### 2.3 Why KL Annealing Instead of Fixed KL Weight?

The lab manual uses a fixed KL weight of `5e-4`:
```python
kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma))
```

This is arbitrary and causes a common VAE failure called **posterior collapse**.

**What is posterior collapse?**
If the KL weight is too large at the start of training, the network takes the path of least resistance — it sets all $\mu = 0$ and $\log\sigma = 0$ (making the posterior identical to the prior $\mathcal{N}(0,1)$). The KL loss is perfectly minimized but the latent space carries no information — the decoder ignores $z$ entirely and generates average-looking images regardless of input.

**KL Annealing solution:**
Start with KL weight = 0 (pure reconstruction training) and linearly increase to 1 over the first half of training:

```python
def get_kl_weight(epoch, total_epochs, max_weight=1.0):
    warmup_epochs = total_epochs // 2
    if epoch < warmup_epochs:
        return max_weight * (epoch / warmup_epochs)
    return max_weight
```

This gives the encoder time to first learn meaningful latent representations (when KL weight is near 0), then gradually regularize them toward $\mathcal{N}(0,1)$ without collapsing.

---

## 3. Architecture — Complete Design

### 3.1 Encoder

```
Input: (batch, 1, 28, 28)
Conv2d(1→32, 3×3, stride=2, padding=1)  → (batch, 32, 14, 14)  + ReLU
Conv2d(32→64, 3×3, stride=2, padding=1) → (batch, 64, 7, 7)    + ReLU
Conv2d(64→128, 3×3, stride=2, padding=1)→ (batch, 128, 4, 4)   + ReLU
Flatten()                                → (batch, 2048)
Linear(2048 → latent_dim)               → mu      (batch, 2)
Linear(2048 → latent_dim)               → log_var  (batch, 2)
```

**Design choices:**
- Stride=2 instead of MaxPooling — learnable downsampling preserves more information
- Three conv blocks give the encoder 3 levels of feature abstraction
- Two separate Linear heads for $\mu$ and $\log\sigma^2$ — these are independent outputs, each needing its own weights

### 3.2 Reparameterization

```python
def reparameterize(self, mu, log_var):
    if self.training:
        std     = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
    return mu   # deterministic at inference
```

This is the mathematical heart of the VAE. The trick separates the stochastic part ($\epsilon$) from the learnable part ($\mu$, $\sigma$) — gradients flow cleanly through $\mu$ and $\sigma$ while $\epsilon$ is treated as a constant.

During inference (`model.eval()`), we return $\mu$ directly — the most likely point in the learned distribution, giving deterministic reconstructions.

### 3.3 Decoder

```
Input: (batch, 2)
Linear(2 → 2048)                                    → (batch, 2048)
Reshape                                             → (batch, 128, 4, 4)
ConvTranspose2d(128→64, 3×3, stride=2, padding=1)  → (batch, 64, 7, 7)   + ReLU
ConvTranspose2d(64→32, 3×3, stride=2, padding=1)   → (batch, 32, 14, 14) + ReLU
ConvTranspose2d(32→1, 3×3, stride=2, padding=1)    → (batch, 1, 28, 28)  + Sigmoid
```

**ConvTranspose2d (Transposed Convolution):**
The exact reverse of Conv2d — it expands spatial dimensions. Each input pixel is multiplied by the filter weights and spread across a larger output region. This is also called "fractionally strided convolution" or informally "deconvolution".

**Sigmoid at output:**
Maps final pixel values to [0, 1] — matching the normalized [0, 1] range of the input images. Binary Cross-Entropy loss requires outputs in this range.

---

## 4. Loss Function — The Two-Part Objective

$$L_{VAE} = L_{reconstruction} + \beta \cdot L_{KL}$$

### 4.1 Reconstruction Loss

$$L_{recon} = -\sum_{i} [x_i \log\hat{x}_i + (1-x_i)\log(1-\hat{x}_i)]$$

Binary Cross-Entropy between the original image and reconstruction. Measures pixel-wise fidelity — how accurately does the decoder reproduce the input?

Divided by batch size to keep the loss scale independent of batch size.

### 4.2 KL Divergence Loss

$$L_{KL} = -\frac{1}{2}\sum_j\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

```python
kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
```

Measures how different the learned distribution $\mathcal{N}(\mu, \sigma^2)$ is from the standard normal $\mathcal{N}(0,1)$. Pushes the encoder to produce distributions centered near 0 with spread near 1.

**The tension between the two losses:**
- Minimizing $L_{recon}$ alone → encoder maps each image to a very specific, tight region (like a standard AE)
- Minimizing $L_{KL}$ alone → encoder maps everything to $\mathcal{N}(0,1)$, ignoring input
- Minimizing both → encoder maps images to overlapping Gaussian regions that are still distinguishable — the structured latent space

The KL vs Reconstruction loss curves plotted separately reveal this tension — as KL weight increases during annealing, reconstruction loss temporarily rises before both stabilize.

---

## 5. Training Configuration

| Hyperparameter | Value | Reasoning |
|----------------|-------|-----------|
| Latent dimensions | 2 | Enables 2D visualization of latent space |
| Batch size | 128 | Balance between speed and gradient stability |
| Learning rate | 1e-3 | Standard Adam starting point |
| Optimizer | Adam | Adaptive learning rate — fast convergence |
| Epochs | 30 | Sufficient for convergence on Fashion MNIST |
| KL warmup | 15 epochs | First 50% of training — allows encoder to stabilize |
| Max KL weight | 1.0 | Full KL regularization after warmup |

---

## 6. Cell-by-Cell Results Analysis

### Cell 8 — Training Curves

Three separate plots reveal the dynamics of VAE training:

**Total Loss:** Should decrease steadily and plateau — confirms overall convergence.

**Reconstruction Loss:** Drops steeply in early epochs when KL weight is near 0 (pure reconstruction training). May rise slightly as KL weight increases (the encoder sacrifices some reconstruction precision for better latent space structure). Stabilizes in the second half of training.

**KL Divergence Loss:** Starts near 0 (encoder not yet regularized), increases as KL weight grows, then stabilizes. The shape mirrors the annealing schedule — a linear rise followed by plateau.

**The Reconstruction vs KL tension plot** is the most informative — it shows exactly when the two objectives compete and when they reach equilibrium. A healthy VAE shows both curves decreasing together by the end of training.

### Cell 9 — Reconstruction Quality

Side-by-side comparison of original and reconstructed images. A well-trained CNN VAE should show:
- Clear clothing silhouettes preserved
- Texture slightly smoothed (VAE's inherent blurriness — a known limitation)
- Class identity clearly maintained (shoes remain shoes, bags remain bags)

The 2D latent space (only 2 dimensions) is intentionally restrictive — more latent dimensions would produce sharper reconstructions but cannot be visualized as a 2D grid.

### Cell 10 — 20×20 Latent Space Grid

The most visually striking output. Each cell in the grid corresponds to one point in the 2D latent space decoded into an image. Reading the grid reveals:

- **Smooth transitions:** Moving from one cell to an adjacent cell produces gradual morphing between clothing types — not abrupt jumps. This is the direct result of KL regularization making the latent space continuous.
- **Organized regions:** Different areas of the grid correspond to different clothing categories — shoes cluster in one region, tops in another.
- **Edge regions:** Points at the extreme edges of the grid (sampled from the tails of the Gaussian) may produce blurry or ambiguous images — the decoder has less training signal from these less-populated regions.

### Cell 11 — Latent Space Scatter Plot

All 60,000 training images encoded to their $\mu$ values and plotted in 2D, colored by class. A well-trained VAE shows:
- **Class clustering:** Same-class items group together
- **Smooth boundaries:** Neighboring clusters correspond to visually similar items (e.g., Shirt near T-shirt/top near Pullover — all upper-body clothing)
- **Centered distribution:** Most points near (0,0) — the KL regularization is working
- **No holes:** The space between clusters is filled — the decoder can generate coherent images even between clusters

### Cell 12 — Class-Conditional Generation

By encoding all images of each class, finding the mean latent position (class center), and sampling with noise_scale=0.5 around it, we generate 8 variations of each clothing type. This demonstrates:

- The VAE has learned class-specific regions of latent space
- Small perturbations around a class center produce recognizable variations of that class
- Larger noise_scale values produce more diverse but potentially less class-faithful variations

### Cell 13 — FID-Inspired Metric

Compares pixel statistics of real vs generated images across 4 statistical moments:

| Metric | Interpretation |
|--------|---------------|
| Mean difference | How bright generated images are vs real |
| Std deviation difference | How much contrast variation exists |
| Skewness | Asymmetry of pixel distribution |
| Kurtosis | Tail behavior — presence of very dark/bright pixels |

A well-trained VAE shows small differences across all four moments. The pixel distribution histogram visually confirms whether generated images approximate the overall brightness and contrast of real Fashion MNIST images.

---

## 7. Parameters That Change Results

These are the key knobs — changing them significantly affects output quality:

### 7.1 `latent_dim` — Most Impactful

| Value | Effect |
|-------|--------|
| 2 | Visualizable but poor reconstruction — not enough capacity |
| 8–16 | Good balance — sharp reconstructions, cannot directly visualize |
| 32–64 | Very sharp reconstructions — state of the art for Fashion MNIST |
| Too large | Latent space becomes sparse — generation quality drops |

**Recommendation:** Use 2 only for visualization experiments. Use 16–32 for quality generation.

### 7.2 `noise_scale` in Class-Conditional Generation

| Value | Effect |
|-------|--------|
| 0.1 | Very tight — almost identical variations |
| 0.5 | Good diversity while staying on-class |
| 1.0 | High diversity — may cross class boundaries |
| 2.0+ | Likely produces off-class or blurry images |

### 7.3 KL Annealing Schedule

| Schedule | Effect |
|----------|--------|
| No annealing (fixed weight=1.0) | Risk of posterior collapse, poor reconstructions |
| Linear warmup (our approach) | Stable training, good latent structure |
| Cyclical annealing | Often better — multiple KL peaks force encoder to use latent space |
| Fixed weight=5e-4 (manual's approach) | Effectively ignores KL — becomes near-standard AE |

### 7.4 `max_weight` in KL Annealing

| Value | Effect |
|-------|--------|
| 0.1 | Near-standard AE — sharp reconstructions, poor generation |
| 0.5 | Balanced — moderate reconstruction quality, decent generation |
| 1.0 | Standard VAE — good generation, slightly blurry reconstructions |
| >1.0 | Over-regularized — very blurry, posterior collapse risk |

### 7.5 Architecture Depth

| Encoder depth | Effect |
|--------------|--------|
| 2 Conv blocks | Fast, lower quality |
| 3 Conv blocks (our approach) | Good quality, manageable training time |
| 4+ Conv blocks | Better quality, needs BatchNorm to train stably |

### 7.6 Learning Rate

| Value | Effect |
|-------|--------|
| 1e-2 | Too large — loss oscillates, may diverge |
| 1e-3 | Standard — good convergence (our choice) |
| 1e-4 | Slow but stable — good for fine-tuning |
| 1e-5 | Too slow — underfits in 30 epochs |

---

## 8. Key Learnings

**1. The reparameterization trick is the enabling insight of VAEs:**
The separation of randomness ($\epsilon$) from learned parameters ($\mu$, $\sigma$) makes the entire VAE framework possible. Without it, gradients cannot flow through sampling operations and the model cannot be trained with standard backpropagation.

**2. KL divergence is not just a regularizer — it is the architect of the latent space:**
Without KL loss, the latent space is unstructured and generation fails completely. With KL loss, the latent space becomes a smooth, continuous manifold where interpolation produces meaningful new images.

**3. The reconstruction-KL tension is fundamental, not a bug:**
These two objectives genuinely compete. Better reconstruction requires specific, tight encodings. Better generation requires smooth, overlapping distributions. The balance between them determines the character of the model — more AE-like or more generative.

**4. CNN architecture is not optional for image generation:**
Dense VAEs produce blurry, low-quality results because they lose spatial structure. CNN encoders + ConvTranspose decoders are the minimum viable architecture for image generation on any real dataset.

**5. Latent space dimensionality controls a fundamental tradeoff:**
Lower dimensions → more compressed, smoother latent space, worse reconstruction. Higher dimensions → sharper reconstruction, less smooth generation. The choice depends on whether the priority is visualization, reconstruction quality, or generation quality.

**6. Generative models learn the data distribution, not just decision boundaries:**
Every discriminative model (LR, SVM, RF, CNN classifier) learns what separates classes. The VAE learns what makes each class look the way it does — a fundamentally richer representation.

---

## 9. Limitations

### 9.1 Inherent Blurriness
VAEs produce slightly blurry images — a direct consequence of minimizing pixel-wise reconstruction loss (BCE/MSE). These losses are minimized by producing average-looking images rather than sharp, specific ones. GANs (Experiment 7) address this through adversarial training which produces much sharper images.

### 9.2 2D Latent Space is Too Small
Using `latent_dim=2` for visualization severely limits reconstruction quality. Real-world VAE applications use 128–512 latent dimensions. The 2D constraint is a pedagogical choice that trades quality for interpretability.

### 9.3 FID Metric is Approximate
True FID (Fréchet Inception Distance) computes statistics in the feature space of a pre-trained Inception network — capturing perceptual similarity rather than pixel statistics. Our pixel-level statistics are a much simpler proxy that misses perceptual quality entirely. Two images can have identical pixel statistics but look completely different to a human.

### 9.4 No Disentanglement
The 2D latent space learned here mixes multiple attributes — a point might encode both "is a shoe" and "is dark-colored" simultaneously. Disentangled VAEs (β-VAE, FactorVAE) learn latent dimensions that each control a single interpretable attribute — but require careful hyperparameter tuning and more latent dimensions.

### 9.5 Fashion MNIST is Too Simple
Fashion MNIST (28×28 grayscale) is an easy benchmark. Real generative modeling challenges involve high-resolution color images (256×256 RGB) with complex backgrounds — where simple CNN VAEs fail and require architectures like VQ-VAE, VQ-VAE-2, or diffusion models.

### 9.6 No Conditional Generation Control
Our class-conditional generation samples around class centers — a post-hoc technique that does not guarantee class-faithful outputs, especially at high noise_scale values. Proper conditional VAEs (CVAE) condition the encoder and decoder directly on class labels during training, giving explicit control over what class is generated.

---

## 10. Comparison — Lab Manual vs Our Implementation

| Aspect | Lab Manual (Keras) | Our Implementation (PyTorch) |
|--------|-------------------|------------------------------|
| Framework | Old Keras 2 backend | PyTorch — consistent with Lab 2 |
| Encoder | Dense(784→512→2) | CNN (Conv2D × 3) |
| Decoder | Dense(2→512→784) | CNN (ConvTranspose2D × 3) |
| KL weight | Fixed 5e-4 | Annealed 0→1 over 15 epochs |
| KL in Part 1 | No KL (just reconstruction) | Full KL from start |
| Data loading | Manual CSV download | torchvision.datasets (automatic) |
| Generation | 20×20 grid only | Grid + class-conditional + FID metric |
| Loss visualization | Single total loss | Reconstruction + KL separately |
| Posterior collapse | Risk — fixed tiny KL weight | Prevented by annealing |

---

## 11. Results Summary

| Metric | Value |
|--------|-------|
| Architecture | CNN Encoder + ConvTranspose Decoder |
| Latent dimensions | 2 |
| Training epochs | 30 |
| KL warmup | 15 epochs (linear) |
| Latent space | Smooth, class-clustered, continuous |
| Generation quality | Recognizable clothing items |
| Class-conditional | 8 variations per class, noise_scale=0.5 |
| Pixel mean difference | Small (< 0.05 for well-trained model) |

---

*Experiment 6 Lab Report | CS-405 Deep Learning*
