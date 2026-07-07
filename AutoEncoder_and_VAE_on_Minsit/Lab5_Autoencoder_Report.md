# Experiment 5 — Lab Report
## Image Denoising and Feature Learning using Autoencoders
**Course:** Deep Learning Lab (CS-405)
**Dataset:** Fashion MNIST (60,000 train / 10,000 test)
**Framework:** PyTorch

---

## 1. Objective

To implement and compare three autoencoder architectures — Standard Autoencoder (AE), Denoising Autoencoder (DAE), and Variational Autoencoder (VAE) — on the Fashion MNIST dataset. The experiment demonstrates unsupervised feature learning, image denoising, latent space organization, and generative modeling, with quantitative evaluation using the Structural Similarity Index (SSIM).

---

## 2. Why This Approach — Why Autoencoders?

### The Shift from Supervised to Unsupervised Learning

Every model built before this experiment — DNN, CNN, LSTM — was supervised. You gave the model inputs and correct labels, and it learned the mapping between them. Supervised learning requires large amounts of labeled data, and labeling data is expensive and time-consuming.

Autoencoders are **unsupervised** — they learn from data alone, with no labels. The training signal is the data itself: compress the input into a compact representation, then reconstruct it. If the reconstruction matches the original, the model learned something meaningful about the data's structure.

This is powerful because unlabeled data is abundant. A hospital has millions of X-ray images but only a fraction with expert annotations. An autoencoder can learn the visual structure of X-rays from all of them — labeled or not.

### What an Autoencoder Actually Learns

The bottleneck forces the network to discover the most important features. With 784 input pixels compressed to 32 latent dimensions, the encoder cannot memorize — it must generalize. The 32 numbers in the latent vector must encode everything needed to reconstruct the image: shape, texture, orientation, clothing type. These 32 numbers are a learned, compact description of the image — far more meaningful than raw pixels.

### Why Three Different Architectures?

Each architecture solves a different problem:

**Standard AE** — establishes the baseline. Can it compress and reconstruct?

**Denoising AE** — adds a practical task. Given a corrupted image, can it recover the clean original? This tests whether the latent representation captures true structure rather than surface-level pixel patterns.

**VAE** — replaces the fixed latent point with a probability distribution. This seemingly small change produces a fundamentally different latent space — smooth, continuous, and generative. The VAE can create new images that were never in the training set.

---

## 3. Steps Taken (Overview)

1. Loaded Fashion MNIST with `ToTensor()` only — no mean/std normalization, pixel range kept at [0,1] to match Sigmoid decoder output
2. Visualized one sample per class and pixel value distribution
3. Built Standard AE — Linear encoder (784→256→128→32), Linear decoder (32→128→256→784→Sigmoid)
4. Trained Standard AE with MSELoss and Adam for 20 epochs
5. Visualized original vs reconstructed images, computed per-class SSIM scores
6. Built Denoising AE — same architecture, different training loop
7. Trained DAE — added Gaussian noise (factor=0.4) to input, reconstructed clean image
8. Visualized Original → Noisy → Denoised pipeline, computed SSIM improvement
9. Built VAE — encoder outputs μ and log_var, reparameterization trick, decoder from sampled z
10. Defined VAE loss — reconstruction (BCE) + KL divergence
11. Trained VAE with latent_dim=2 for 30 epochs
12. Visualized VAE reconstructions, computed SSIM
13. Plotted 2D latent space scatter — all 10,000 test images colored by class
14. Generated new images by sampling from N(0,1)
15. Performed latent space interpolation — Sandal → Sneaker morphing
16. Side-by-side comparison of all three models
17. Final summary with SSIM scores for all models

---

## 4. Cell-by-Cell — What Happened

**Cell 1 — Imports:** Loaded PyTorch, torchvision, sklearn's SSIM metric. Set seeds. Confirmed GPU availability.

**Cell 2 — Dataset:** Downloaded Fashion MNIST. Applied only `ToTensor()` — no standardization. Unlike classification where normalization helps gradient descent, autoencoders use Sigmoid in the decoder which outputs [0,1]. The target pixel values must be in the same range — so we keep pixels in [0,1] not centered around zero.

**Cell 3 — Visualization:** Plotted one sample per class confirming 10 distinct clothing categories. Pixel histogram showed most pixels are dark (background) with a spike near 1.0 (clothing pixels) — a bimodal distribution that the encoder must learn to handle.

**Cell 4 — Standard AE:** Built symmetric encoder-decoder. Latent dim=32 gives 24.5× compression. Sigmoid in decoder output ensures values stay in [0,1] matching pixel range. `forward()` returns both the reconstruction and the latent vector z — z is used later for visualization.

**Cell 5 — Train Standard AE:** Training loop uses `_` to discard labels — autoencoders are unsupervised. MSELoss compares reconstructed pixels to original pixels. Loss decreased steadily over 20 epochs confirming the network learned to compress and reconstruct clothing images.

**Cell 6 — Reconstructions + SSIM:** Visualized 8 original/reconstructed pairs. SSIM measured perceptual similarity — values above 0.7 indicate good reconstruction. Per-class SSIM revealed which clothing types are hardest to reconstruct — typically Shirt and Pullover which have subtle texture differences.

**Cell 7 — Denoising AE:** Same architecture as standard AE. The key difference is in the training loop: `add_noise()` adds Gaussian noise (factor=0.4) to create corrupted images, the model receives the noisy image as input but is trained to output the clean original. `torch.clamp(noisy, 0, 1)` keeps corrupted pixels in valid range.

**Cell 8 — Denoising Results:** Plotted three rows — original, noisy, denoised. SSIM comparison showed noisy image SSIM vs clean, then denoised image SSIM vs clean. The improvement in SSIM confirmed the DAE successfully recovered structure that the noise destroyed.

**Cell 9 — VAE Architecture:** Two critical new components compared to standard AE. First, the encoder splits into two heads — `fc_mu` outputs the mean vector and `fc_log_var` outputs the log variance. Second, `reparameterize()` implements `z = μ + exp(0.5×log_var) × ε` where ε~N(0,1). Using log_var instead of var ensures numerical stability — variance must be positive, and exp() guarantees this. Latent dim=2 was chosen specifically to enable 2D visualization.

**Cell 10 — VAE Loss and Training:** VAE loss has two components. Reconstruction loss (BCE) measures how well the decoder reconstructed the image. KL divergence measures how far the learned distribution N(μ,σ²) is from the standard normal N(0,1). KL loss formula: `-0.5 × Σ(1 + log_var - μ² - exp(log_var))`. Without KL loss, the VAE collapses to a standard AE — each image maps to a unique region of latent space with gaps between them. KL loss forces all encodings toward N(0,1), making the space smooth and continuous. Training for 30 epochs showed both reconstruction and KL loss decreasing.

**Cell 11 — VAE Reconstructions:** VAE reconstructions are slightly blurrier than standard AE — this is expected and not a failure. The VAE samples from a distribution rather than mapping to a fixed point. The blur comes from averaging across slightly different samples. The trade-off is a smooth latent space that enables generation.

**Cell 12 — Latent Space Visualization:** Encoded all 10,000 test images and plotted their μ vectors in 2D colored by class. Well-separated clusters confirmed the VAE learned to organize clothing types spatially — similar items (Shirt, T-shirt, Pullover) clustered near each other, dissimilar items (Trouser, Bag) were far apart. This organization emerged entirely from unsupervised training — no class labels were used.

**Cell 13 — Generation + Interpolation:** Sampling random points from N(0,1) and decoding them produced valid-looking clothing images — confirming the latent space is smooth everywhere. Latent interpolation between a Sandal (class 5) and Sneaker (class 7) encoded two real images, computed 10 evenly spaced points between their μ vectors, and decoded each. The resulting images smoothly morphed from sandal to sneaker — a capability impossible with standard AE.

**Cell 14 — Comparison Plot:** Plotted training curves for all three models and a 5-row grid (Original, Noisy, AE Recon, DAE Denoised, VAE Recon) for 6 test images simultaneously. The comparison made the differences in reconstruction quality, blur, and denoising performance visually concrete.

**Cell 15 — Final Summary:** Computed SSIM for all three models on the same test batch. Printed complete architecture and results summary.

---

## 5. What We Learned — What Changed Performance and Why

### ToTensor Only (No Normalization) — Critical Design Choice
Unlike classification tasks, autoencoders must not standardize pixel values. The Sigmoid decoder outputs [0,1]. If pixels were normalized to [-1,1] (standard classification normalization), the MSE loss between [-1,1] targets and [0,1] outputs would be incorrect. Keeping pixels in [0,1] aligns decoder output range with target range — the loss is meaningful.

### Symmetric Architecture — Why It Works
The encoder and decoder mirror each other: 784→256→128→32 and 32→128→256→784. This symmetry is intentional — if the encoder compresses through certain dimensions, the decoder has matching capacity to expand back. Asymmetric architectures (large encoder, small decoder) produce poor reconstructions because the decoder lacks capacity to fully expand the compressed representation.

### Noise Factor = 0.4 — Calibrating Denoising Difficulty
A noise factor of 0.4 was chosen to make the task challenging but solvable. Too low (0.1) and the DAE barely differs from standard AE — the noise is trivial to ignore. Too high (0.8) and the signal is completely buried — the network cannot recover. At 0.4, the noisy images are visibly corrupted but retain enough structure that a well-trained DAE can recover the original. SSIM improvement from noisy to denoised confirmed this calibration was correct.

### latent_dim=2 for VAE — Visualization Over Performance
A latent dim of 2 was deliberately chosen to enable 2D scatter visualization — the single most informative diagnostic for a VAE. In practice, latent_dim=32 or higher would produce sharper reconstructions because the network has more capacity to encode image details. The blurriness of VAE reconstructions in this experiment is partly a consequence of this low-dimensional constraint, not purely a VAE property. The trade-off was accepted for visualization clarity.

### KL Loss — The Key Ingredient for Generation
Training a VAE without KL loss produces sharper reconstructions but a fragmented latent space. Each image maps to an isolated point with empty space between clusters. Sampling from N(0,1) in empty regions produces garbage. KL loss fills these gaps by pushing all encodings toward a standard normal — every region of the latent space becomes meaningful. This is why VAE can generate new images while standard AE cannot.

### BCE vs MSE for VAE Reconstruction Loss
VAE uses Binary Cross-Entropy for reconstruction, while standard AE uses MSE. BCE is more appropriate because it treats each pixel as an independent Bernoulli variable — either on or off — which matches the bimodal pixel distribution observed in Cell 3. MSE treats pixel errors as Gaussian, which is less appropriate for Fashion MNIST's sparse, binary-like pixel structure.

### Results Summary

| Model | Latent Dim | SSIM | Unique Capability |
|---|---|---|---|
| Standard AE | 32 | Highest | Compression, reconstruction |
| Denoising AE | 32 | High (from noisy) | Noise removal |
| VAE | 2 | Moderate (blurry) | Generation, interpolation |

---

## 6. Practical Use — Real-World Applications

### Standard Autoencoder

**Anomaly Detection:** train an AE on normal data only. At inference, abnormal inputs (defective products, fraudulent transactions, network intrusions) reconstruct poorly — high MSE signals an anomaly. No labels needed. Used in manufacturing quality control, cybersecurity intrusion detection, and financial fraud detection.

**Dimensionality Reduction:** the 32-dimensional latent vector is a compact representation of a 784-dimensional image. Feed this compressed representation to a classifier instead of raw pixels — faster training, better generalization. PCA (from Group 4 of ML Lab) does the same thing linearly; AE does it non-linearly, capturing more complex structure.

**Recommendation Systems:** compress user behavior (purchase history, click patterns) into a latent vector. Users with similar latent vectors have similar preferences — use this for collaborative filtering without explicit similarity metrics.

### Denoising Autoencoder

**Medical Imaging:** MRI and CT scans are inherently noisy due to hardware limitations and short scan times. A DAE trained on clean-noisy pairs removes acquisition noise while preserving diagnostic features — the same operation demonstrated on Fashion MNIST. Radiologists see cleaner images; diagnosis accuracy improves.

**Old Document Restoration:** scan degraded historical documents — faded ink, paper stains, physical damage. DAE trained on clean-degraded pairs recovers the original text and illustrations. Libraries and archives use this to digitize and restore historical records.

**Speech Enhancement:** replace Fashion MNIST images with audio spectrograms, replace Gaussian noise with background noise (traffic, crowd, wind). The DAE learns to recover clean speech from noisy recordings — used in hearing aids, video conferencing systems, and voice assistants.

**Satellite Imagery:** atmospheric interference, sensor noise, and compression artifacts degrade satellite images. DAE removes these artifacts to produce cleaner images for land use analysis, weather prediction, and military reconnaissance.

### Variational Autoencoder

**Data Augmentation:** when labeled training data is scarce, generate new training samples by sampling from the VAE's latent space. New clothing images generated from VAE can augment Fashion MNIST training — the classifier sees more variety, generalizes better.

**Drug Discovery:** encode molecular structures into a smooth latent space. Interpolate between two known drug molecules to find new candidate molecules with intermediate properties. Sample from the latent space to propose entirely new molecular structures. This is active use in pharmaceutical research — VAEs help explore chemical space efficiently.

**Face Generation and Editing:** VAE trained on face images learns a latent space where different dimensions correspond to age, expression, hair color, and pose. Interpolating along one dimension smoothly changes one attribute while preserving others. Used in entertainment, game character creation, and privacy-preserving synthetic data generation.

**Anomaly Detection with Uncertainty:** VAE not only reconstructs but also provides uncertainty estimates through its probabilistic output. High reconstruction error AND high variance in the latent distribution both signal anomalies — a richer signal than standard AE's reconstruction error alone.

---

### Decision Guide — Which Autoencoder for Which Problem

| Situation | Use | Reason |
|---|---|---|
| Compress data, reduce dimensions | Standard AE | Simple, effective, high SSIM |
| Remove noise from corrupted data | Denoising AE | Trained specifically for this task |
| Generate new samples | VAE | Smooth latent space enables sampling |
| Detect anomalies | Standard AE or VAE | High reconstruction error = anomaly |
| Explore latent space structure | VAE | Continuous, visualizable space |
| Need sharp reconstructions | Standard AE | Fixed latent point, no blur from sampling |
| Need interpretable latent dims | VAE | Each dim often corresponds to a feature |

---

## 7. Conclusion

Experiment 5 demonstrated that unsupervised learning through reconstruction is a powerful paradigm — the network learns meaningful representations of clothing images without ever being told what a T-shirt or sandal looks like. The progression from Standard AE to Denoising AE to VAE is not just increasing complexity — each step adds a genuinely new capability.

The Standard AE established that compression works — 32 numbers can faithfully describe a 784-pixel image. The Denoising AE showed that the learned representation captures true structure, not surface pixels — it recovered images from corrupted inputs the training process never explicitly saw. The VAE demonstrated that a probabilistic latent space is qualitatively different — smooth, continuous, generative, and visualizable in ways that standard AE cannot match.

The latent space visualization in Cell 12 was the conceptual payoff of the entire experiment. Ten clothing classes, 10,000 images, organized into clusters by an unsupervised model using only reconstruction loss and KL divergence — no labels, no explicit class information. This emergent organization is the core promise of representation learning, and the experiment demonstrated it concretely.

The interpolation result in Cell 13 directly connected this experiment to modern generative AI. The same principle — smooth interpolation in a learned latent space — underlies image editing tools, face generation systems, and molecular design platforms in production today.

---

*Experiment 5 Lab Report | Deep Learning Lab (CS-405)*
*Models: Standard AE + Denoising AE + VAE | Dataset: Fashion MNIST | Framework: PyTorch*
