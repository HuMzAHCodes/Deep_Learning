# Experiment 7 — Lab Report
## Generative Adversarial Networks (GANs) for Synthetic Data Generation
**Course:** Deep Learning Lab (CS-405)
**Dataset:** 1D Gaussian (Part 1) + MNIST (Part 2)
**Framework:** PyTorch

---

## 1. Objective

To understand and implement Generative Adversarial Networks by training a Generator and Discriminator in an adversarial framework. Part 1 establishes the core GAN concept on a simple 1D distribution matching task. Part 2 scales this to image generation using a Deep Convolutional GAN (DCGAN) that learns to produce realistic handwritten digits from random noise.

---

## 2. Why We Needed This Model — The Gap GANs Fill

### The Limitation of Every Previous Model

Every model built in this lab series so far — ANN, CNN, LSTM, Autoencoder, VAE — was either a classifier or a reconstructor. Given an input, produce an output. Given an image, classify it. Given a noisy image, reconstruct it. Even the VAE from Lab 5/6, which could generate new images, was fundamentally constrained by its reconstruction objective — it averaged over possible outputs, producing blurry generations because it was optimizing pixel-wise MSE or BCE loss.

None of these models could answer the question: **can a neural network learn to create something new from scratch, indistinguishable from real data, without being explicitly told what "real" looks like numerically?**

GANs answer this question. They do not minimize a fixed loss against a target — they learn what "real" looks like by competing against another network that is simultaneously learning to detect fakes.

### The Core Insight — Adversarial Training

Ian Goodfellow introduced GANs in 2014 with a simple but powerful idea: instead of defining a fixed loss function that measures how good the generator is, use another neural network (the discriminator) as the loss function — and train both simultaneously.

The generator and discriminator play a minimax game:

```
Generator goal     : fool the discriminator — make fake data look real
Discriminator goal : correctly identify real data as real, fake as fake

Generator   improves → Discriminator must improve to keep up
Discriminator improves → Generator must improve to keep fooling it
```

At equilibrium, the generator produces data so realistic that the discriminator cannot do better than random guessing (50/50). This equilibrium is theoretically guaranteed to produce samples from the true data distribution — something no fixed reconstruction loss can achieve.

### Why Not Just Use VAE?

The VAE from Lab 5 could generate new Fashion MNIST images. But the generated images were noticeably blurry. This is a fundamental limitation of reconstruction-based training:

- VAE optimizes BCE/MSE pixel by pixel — any pixel can be off by a small amount
- Averaging over many possible reconstructions produces smooth, blurry results
- The loss function does not care about perceptual sharpness — only per-pixel accuracy

GAN's discriminator is a learned perceptual judge. It does not say "pixel (14,7) is wrong by 0.03" — it says "this does not look like a real digit." This holistic judgment produces sharper, more realistic images. A well-trained DCGAN generates digits that are genuinely difficult to distinguish from real MNIST samples — something no VAE achieves on the same dataset.

---

## 3. Steps Taken (Overview)

1. Imported PyTorch, torchvision, scipy for statistical evaluation
2. **Part 1:** Generated real data from N(0,1) and uniform noise for generator input
3. Built 1D Generator (noise_dim=10 → 64 → 1, Tanh output) and Discriminator (1 → 64 → 1, Sigmoid output)
4. Trained 1D GAN for 1000 epochs with label smoothing (real=0.9)
5. Evaluated using statistical moments and Q-Q plot
6. **Part 2:** Loaded MNIST normalized to [-1,1] matching Tanh generator output
7. Built DCGAN Generator — Linear projection → ConvTranspose2D×2 → Tanh (noise → 28×28 image)
8. Built DCGAN Discriminator — Conv2D×2 with LeakyReLU → Sigmoid (image → probability)
9. Applied DCGAN weight initialization — N(0, 0.02)
10. Trained DCGAN for 50 epochs with Adam (lr=0.0002, β₁=0.5) and label smoothing
11. Monitored training dynamics — G/D loss ratio and equilibrium
12. Visualized generated digits at epochs 1, 10, 20, 30, 40, 50
13. Generated final 8×8 grid of 64 new digits
14. Ran mode collapse detection via pixel diversity metric
15. Compared real vs generated pixel statistics

---

## 4. Cell-by-Cell — What Happened

**Cell 1 — Imports:** Standard PyTorch imports plus `scipy.stats` for moment computation in Part 1. Set seeds for reproducibility.

**Cell 2 — 1D Data Generation:** Generated 10,000 samples from N(0,1) as "real data" and 10,000 uniform samples as generator noise. Visualized both distributions — the transformation from red (uniform) to blue (Gaussian) is the entire goal of Part 1. This visualization makes the generator's task concrete before writing a single line of model code.

**Cell 3 — 1D Generator and Discriminator:** Built two tiny networks. Generator takes 10 uniform values and outputs 1 number — the Tanh activation allows negative outputs matching the Gaussian's range. Discriminator takes 1 number and outputs a probability — Sigmoid constrains to (0,1). Both use configurable hidden dimensions and layer counts to enable easy experimentation.

**Cell 4 — Train 1D GAN:** The training loop demonstrates the alternating update structure that all GANs follow. Step 1 trains the discriminator on real data (label=0.9, label smoothing) and fake data (label=0.0). Step 2 freezes the discriminator and trains the generator to fool it — the generator's loss is BCE with real labels (1.0) applied to fake data. The `detach()` call in Step 1 is critical — it stops gradients from flowing into the generator during the discriminator update.

**Cell 5 — Evaluate 1D GAN:** Compared real and generated distributions via overlaid histograms, Q-Q plot, and four statistical moments. The histogram overlap confirmed the generator learned the approximate shape of N(0,1). The Q-Q plot R² near 1.0 confirmed Gaussian shape. Moment comparison showed GAN matches mean and variance well but struggles with higher-order moments (skewness, kurtosis) — a known GAN limitation.

**Cell 6 — Load MNIST and Build DCGAN Generator:** Normalized MNIST to [-1,1] using `Normalize(0.5, 0.5)` — this matches the Tanh generator output range. Built DCGAN generator with Linear projection (100→128×7×7), then two ConvTranspose2D blocks that upsample spatially: 7×7→14×14→28×28. BatchNorm after each ConvTranspose2D stabilizes training. Tanh in the final layer outputs in [-1,1].

**Cell 7 — Build DCGAN Discriminator:** Mirror of the generator using Conv2D to shrink: 28×28→14×14→7×7. LeakyReLU(0.2) instead of ReLU — passes 20% of negative gradients through, providing richer signal to the generator. No BatchNorm on the first layer — standard DCGAN guideline. Applied weight initialization N(0, 0.02) to all Conv and BatchNorm layers per the original DCGAN paper.

**Cell 8 — Training Loop:** Full DCGAN training for 50 epochs. Every batch performs the two-step alternating update. Monitored D(real) and D(fake) averages per epoch — at equilibrium both should converge toward 0.5. Saved generated images at epochs 1, 10, 20, 30, 40, 50 using fixed noise for consistent visual comparison. Saved generator checkpoints every 10 epochs.

**Cell 9 — Loss Curves:** Plotted G and D losses with smoothing. Added equilibrium reference line at ln(2)≈0.693 — the theoretical loss value when D cannot distinguish real from fake. Plotted G/D ratio — a healthy ratio oscillates near 1.0. Divergence in either direction signals training instability.

**Cell 10 — Visual Progression:** Plotted 8 generated digits at each saved checkpoint. Early epochs showed blurry noise — the generator had not yet learned digit structure. Later epochs showed recognizable, diverse handwritten digits. The mode collapse check computed pixel standard deviation across 64 generated images — values above 0.15 confirm the generator is producing diverse outputs rather than a single repeated image.

**Cell 11 — Final Summary:** Compared pixel statistics (mean, std, min, max) between 1000 real and 1000 generated images. Printed complete training summary including all architectural decisions, hyperparameters, and results.

---

## 5. What We Learned — Key Findings

### Label Smoothing Matters
Using real_label=0.9 instead of 1.0 prevented the discriminator from becoming overconfident early in training. An overconfident discriminator produces near-zero gradients for the generator — it gives the generator no useful signal to improve. Label smoothing keeps gradients flowing throughout training.

### β₁=0.5 in Adam is GAN-Specific
Standard Adam uses β₁=0.9 (high momentum). In GAN training, high momentum causes oscillations — the generator and discriminator overshoot each other's equilibrium. Reducing β₁ to 0.5 lowers momentum, making updates more responsive to current gradients. This is one of the most practically impactful hyperparameter changes specific to GAN training.

### ConvTranspose2D vs Dense for Image Generation
The manual's Dense generator (100→256→512→1024→784) treats all 784 pixels independently. DCGAN's ConvTranspose2D generator builds images spatially — 7×7 feature maps are upsampled to 14×14, then 28×28. Each generated pixel is influenced by its neighbors through the convolutional structure. Result: sharper, more coherent digit shapes with realistic local texture.

### LeakyReLU in Discriminator is Essential
Standard ReLU kills all negative activations. In the discriminator, dead neurons reduce the discriminator's ability to detect subtle fake artifacts. LeakyReLU(0.2) allows 20% of the negative signal through — the discriminator remains sensitive to both positive and negative evidence of fakeness. This gives the generator richer gradient signal to improve against.

### The Equilibrium Signal
The theoretical GAN equilibrium is when both G and D have loss = ln(2) ≈ 0.693. At this point the discriminator is outputting 0.5 for all images — it cannot tell real from fake. Monitoring how close the losses get to 0.693 and how stable they remain is the key training diagnostic — more reliable than visual inspection alone.

### Higher-Order Moments are Hard
Part 1's moment comparison revealed a consistent GAN weakness: mean and variance (1st and 2nd moments) were matched well. Skewness (3rd) and kurtosis (4th) were harder. This is because the GAN's binary real/fake signal cannot directly enforce higher-order statistical properties — the discriminator does not explicitly measure skewness. This motivates advanced variants like Wasserstein GAN (WGAN) which directly optimizes a distance metric between distributions.

---

## 6. Real-World Applications

### Image Generation and Data Augmentation
GANs generate realistic synthetic images that are indistinguishable from real ones. When labeled training data is scarce — a common problem in medical imaging, satellite imagery, or rare event detection — GANs augment the dataset with synthetic samples. A GAN trained on 1000 chest X-rays can generate 10,000 more, giving classifiers more training data without additional medical annotation.

### Face Generation and Editing
StyleGAN2 (NVIDIA, 2020) generates photorealistic human faces that do not exist. The website thispersondoesnotexist.com demonstrates this directly. Beyond generation, GANs edit specific attributes — age, expression, hairstyle, lighting — while leaving the rest of the face unchanged. This is used in film production for de-aging actors and in video game character creation.

### Image-to-Image Translation (Pix2Pix, CycleGAN)
Conditional GANs transform images from one domain to another. Pix2Pix turns architectural sketches into photorealistic building renders, satellite images into road maps, and black-and-white photos into color. CycleGAN performs this without paired training data — it learned to translate horse photos to zebra photos (and back) without ever seeing a horse next to a zebra.

**Real deployments:** Adobe Photoshop's Generative Fill uses GAN-based inpainting. Google Maps uses GAN-based satellite-to-map translation. Medical imaging companies use GANs to synthesize rare pathology examples for radiologist training.

### Video Game and Film Visual Effects
GANs generate textures, terrain, and character appearances in video games. NVIDIA's GAN-based upscaling (DLSS) reconstructs high-resolution frames from lower-resolution renders in real time — used in modern GPUs to improve gaming performance without sacrificing visual quality.

### Drug Discovery and Molecular Design
GANs generate novel molecular structures with desired properties — similar to VAE interpolation but with sharper, more realistic outputs. The discriminator learns what "valid molecule" looks like from known drug structures. The generator proposes new candidates. This accelerates early-stage drug discovery by generating millions of candidate molecules computationally rather than synthesizing them in a lab.

### Deepfakes and Synthetic Media
GANs power face-swapping technology — replacing one person's face in a video with another's. While this has legitimate uses (film dubbing, privacy protection), it is also the source of malicious deepfakes. Understanding GAN architecture is therefore important not just for building generative systems but for building detection systems that identify GAN-generated content.

---

## 7. Shortcomings — What GANs Get Wrong

### Training Instability
GAN training is notoriously fragile. The generator and discriminator must improve at roughly the same rate. If the discriminator becomes too strong too fast, the generator receives near-zero gradients and stops learning. If the generator improves too fast, the discriminator never learns to distinguish. This balance is difficult to achieve and sensitive to learning rate, architecture, and batch size. Our label smoothing and β₁=0.5 choices directly address this, but instability remains a fundamental challenge.

### Mode Collapse
The most common GAN failure. The generator discovers a small subset of outputs that consistently fool the discriminator — and produces only those outputs. In MNIST terms, a collapsed generator might produce only "1"s regardless of the input noise. Our pixel diversity metric detected this, but preventing it requires more sophisticated techniques (mini-batch discrimination, unrolled GANs, WGAN).

### No Explicit Density Estimation
A VAE can evaluate the probability of any given image — useful for anomaly detection. A GAN cannot. It can generate samples from the learned distribution but cannot tell you how likely any specific image is. This limits GANs for applications that need uncertainty quantification.

### Evaluation is Difficult
There is no single objective metric for GAN quality. Fréchet Inception Distance (FID) is the current standard but requires a pre-trained Inception network. Our pixel statistics comparison is simpler but less reliable. Visual inspection is subjective. This makes comparing GAN variants and tuning hyperparameters harder than for classifiers where accuracy is unambiguous.

### Higher-Order Moment Matching
As demonstrated in Part 1, GANs match the mean and variance of target distributions well but struggle with skewness and kurtosis. In image terms, this means overall color and contrast statistics match but subtle texture statistics may not. Wasserstein GAN (WGAN) addresses this by using Earth Mover Distance instead of BCE as the discriminator loss — it directly measures distribution distance rather than binary classification.

### Computationally Expensive
Training a GAN requires running two networks per batch — generator and discriminator — plus two separate backward passes. Compared to a classifier that runs one forward and one backward pass per batch, GAN training is roughly 4× more compute-intensive. State-of-the-art GANs (StyleGAN3) require hundreds of GPU hours on expensive hardware.

### No Control Over What is Generated
A standard GAN samples randomly — you cannot specify "generate a '7'" or "generate a boot". Conditional GAN (cGAN) addresses this by conditioning both generator and discriminator on class labels, but this requires labeled data during training — partially defeating the unsupervised advantage.

---

## 8. GAN vs VAE — When to Use Which

| Aspect | GAN | VAE |
|---|---|---|
| Image sharpness | High — discriminator enforces realism | Low — reconstruction loss causes blur |
| Training stability | Difficult — mode collapse, instability | Stable — single loss function |
| Density estimation | No — cannot evaluate P(x) | Yes — explicit probability model |
| Latent space | No structured interpolation | Smooth, continuous, interpolatable |
| Generation speed | Fast — one forward pass | Fast — one forward pass |
| Training speed | Slow — two networks, two passes | Faster — one network |
| Anomaly detection | No | Yes — high reconstruction error |
| Use when | Sharpest possible generation needed | Need structured latent space or anomaly detection |

---

## 9. Conclusion

Experiment 7 demonstrated the adversarial training paradigm — the most influential idea in generative AI since backpropagation. The progression from Part 1 (1D Gaussian matching) to Part 2 (DCGAN on MNIST) made the GAN concept concrete at two scales: simple enough to understand the math in Part 1, complex enough to produce realistic images in Part 2.

The DCGAN architecture — ConvTranspose2D generator, LeakyReLU discriminator, weight initialization, label smoothing, and β₁=0.5 — represents the distilled best practices from the original DCGAN paper. These choices are not arbitrary; each one addresses a specific failure mode of naive GAN training.

The shortcomings section is as important as the results. GANs are powerful but fragile. Mode collapse, training instability, and the absence of density estimation are real limitations that motivate the entire field of GAN research — WGAN, StyleGAN, Diffusion Models. Understanding why GANs fail is the foundation for understanding why diffusion models, the current state of the art in image generation, were developed to replace them.

The connection from this experiment to the real world is direct: Stable Diffusion, DALL-E, and Midjourney — the image generation tools reshaping creative industries — are the successors of the DCGAN implemented here. The architecture evolved, the scale grew by orders of magnitude, but the core problem — learning to generate realistic data from noise — is identical.

---

*Experiment 7 Lab Report | Deep Learning Lab (CS-405)*
*Models: 1D GAN + DCGAN | Datasets: Gaussian + MNIST | Framework: PyTorch*
