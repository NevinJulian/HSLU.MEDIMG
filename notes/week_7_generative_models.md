# Week 07 — Generative Models in Medical Imaging

**Course:** Medical Image Analysis — HSLU  
**Lecturer:** Fabian Gröger (fabian.groeger@hslu.ch)  
**Topics:** Motivation · Autoencoders · Variational Autoencoders (VAE) · Generative Adversarial Networks (GAN) · Diffusion Models (DDPM) · Evaluation · Clinical challenges

---

## Table of Contents

1. [Why Generative Models Matter in Medical Imaging](#1-why-generative-models-matter-in-medical-imaging)
2. [Generative vs. Discriminative Models](#2-generative-vs-discriminative-models)
3. [Autoencoders (AE)](#3-autoencoders-ae)
   - [Architecture and Training](#31-architecture-and-training)
   - [Latent Space Problem](#32-latent-space-problem)
   - [Self-supervised Pre-training](#33-self-supervised-pre-training)
4. [Variational Autoencoders (VAE)](#4-variational-autoencoders-vae)
   - [From AE to VAE — Smooth Latent Space](#41-from-ae-to-vae--smooth-latent-space)
   - [Probabilistic Formulation](#42-probabilistic-formulation)
   - [Training — The Evidence Lower Bound (ELBO)](#43-training--the-evidence-lower-bound-elbo)
   - [VAEs in Medical Imaging — Unsupervised Anomaly Detection](#44-vaes-in-medical-imaging--unsupervised-anomaly-detection)
5. [Generative Adversarial Networks (GAN)](#5-generative-adversarial-networks-gan)
   - [Two-Player Game Intuition](#51-two-player-game-intuition)
   - [Architecture](#52-architecture)
   - [Training Objective](#53-training-objective)
   - [CycleGANs](#54-cyclegans)
   - [GANs in Medical Imaging](#55-gans-in-medical-imaging)
6. [Denoising Diffusion Probabilistic Models (DDPM)](#6-denoising-diffusion-probabilistic-models-ddpm)
   - [Forward Diffusion Process](#61-forward-diffusion-process)
   - [Diffusion Kernel — Direct Sampling](#62-diffusion-kernel--direct-sampling)
   - [Reverse Denoising Process](#63-reverse-denoising-process)
   - [Training and Model Architecture](#64-training-and-model-architecture)
   - [Image Generation Procedure](#65-image-generation-procedure)
   - [Diffusion Models in Medical Imaging](#66-diffusion-models-in-medical-imaging)
7. [Evaluating Generative Models](#7-evaluating-generative-models)
   - [Structural Similarity Index Measure (SSIM)](#71-structural-similarity-index-measure-ssim)
   - [Fréchet Inception Distance (FID)](#72-fréchet-inception-distance-fid)
   - [CLIP Distance](#73-clip-distance)
   - [Clinical Evaluation Pyramid](#74-clinical-evaluation-pyramid)
8. [Clinical Workflow Integration](#8-clinical-workflow-integration)
9. [Risks Specific to Generative Medical Imaging](#9-risks-specific-to-generative-medical-imaging)
10. [Key Takeaways](#10-key-takeaways)
11. [Key Papers](#11-key-papers)

---

## 1. Why Generative Models Matter in Medical Imaging

Generative models are positioned alongside classification, detection, and segmentation as a core task in visual AI — mapping some condition $y$ (or noise $z$) to a full image $x$.

Four structural problems in medical imaging make generative models especially valuable:

| Problem | Description |
|---|---|
| **Data scarcity** | Curated, labelled medical datasets are small compared to natural image datasets. A radiologist hour yields only a few dozen labels. |
| **Privacy & data sharing** | Patient data cannot leave the hospital. Synthetic data enables cross-institution collaboration while complying with HIPAA, GDPR, and Swiss DSG. |
| **Class imbalance & rare disease** | Pathologies of interest are often <1% of cases (e.g., malignant nodules). Generative augmentation can rebalance training distributions. |
| **Modality gaps** | Translation between modalities: MR ↔ CT, low-dose → full-dose, low-pigmentation → rich-pigmentation skin images. |

---

## 2. Generative vs. Discriminative Models

| Type | What it learns | Direction |
|---|---|---|
| **Discriminative** | $p(y \mid x)$ — a label given an input | image → class/label |
| **Generative** | $p(x)$ — the data distribution itself | ? → image |
| **Conditional generative** | $p(x \mid y)$ — data distribution given a condition | label/text/mask → image |

Three dominant neural architectures for image generation: **(V)AE**, **GAN**, **Diffusion Models**.

---

## 3. Autoencoders (AE)

### 3.1 Architecture and Training

An autoencoder learns a compact representation $z$ of the input $x$ by training without labels:

- **Encoder** (CNN with ReLU): $x \mapsto z$ where $z$ is lower-dimensional than $x$ (dimensionality reduction / bottleneck)
- **Decoder** (upsampling CNN): $z \mapsto x'$ (reconstructed image)

**Loss:**

$$\mathcal{L}_{\text{AE}} = \| x - x' \|_2^2$$

Since the loss uses no labels, autoencoders are **self-supervised**. The bottleneck forces the network to learn a compressed, informative representation of image content.

### 3.2 Latent Space Problem

Standard autoencoders produce a latent space that may have **gaps** — regions of $z$ that do not correspond to any valid image — and that is not necessarily smooth or interpolable. Sampling a random point in this space can produce garbage output.

**Solution:** Force the latent distribution to be Gaussian → Variational Autoencoders.

### 3.3 Self-supervised Pre-training

Autoencoders can be repurposed as an **initialisation** for supervised models:

1. Train the autoencoder on unlabelled data (reconstruction task)
2. Discard the decoder
3. Use the encoder weights to initialise a supervised classifier / segmenter

This is especially useful in medical imaging where labelled data is scarce but unlabelled data is plentiful.

---

## 4. Variational Autoencoders (VAE)

### 4.1 From AE to VAE — Smooth Latent Space

A standard AE maps each input to a **single point** in latent space. A VAE instead maps each input $x$ to a **distribution** over latent codes:

$$q_\phi(z \mid x) = \mathcal{N}(\mu_{z|x},\, \Sigma_{z|x})$$

This forces structurally similar images to occupy overlapping, nearby regions of $z$, making interpolation and sampling well-behaved.

### 4.2 Probabilistic Formulation

The full forward pass involves two parameterised distributions:

**Encoder** outputs a Gaussian posterior over $z$ given $x$:

$$q_\phi(z \mid x) = \mathcal{N}(\mu_{z|x},\, \Sigma_{z|x})$$

A sample $z$ is drawn from this distribution, then passed to the decoder.

**Decoder** outputs a Gaussian over reconstructed $x'$ given $z$:

$$p_\theta(x \mid z) = \mathcal{N}(\mu_{x|z},\, \sigma^2)$$

The **prior** over latent codes is a standard normal:

$$p(z) = \mathcal{N}(0, I)$$

Because the true posterior $p_\theta(z \mid x)$ is **intractable to compute** (the integral over all $z$ is too expensive), we approximate it with the encoder network $q_\phi(z \mid x)$.

### 4.3 Training — The Evidence Lower Bound (ELBO)

Training maximises the **Evidence Lower Bound (ELBO)**:

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{z \sim q_\phi(z|x)}\!\left[\log p_\theta(x \mid z)\right]}_{\text{Reconstruction loss}} - \underbrace{D_{\text{KL}}\!\left(q_\phi(z \mid x)\;\|\; p(z)\right)}_{\text{Prior / regularisation loss}}$$

The two terms pull in opposite directions:

- **Reconstruction loss** wants $\Sigma_{z|x} \to 0$ and $\mu_{z|x}$ to be unique per input, so the decoder can deterministically reconstruct $x$.
- **Prior loss (KL term)** wants $\Sigma_{z|x} \to I$ and $\mu_{z|x} \to 0$ so the encoder always produces a unit Gaussian — enabling generation by sampling $z \sim \mathcal{N}(0, I)$.

Because $p_\theta(x \mid z)$ is Gaussian, maximising $\log p_\theta(x \mid z)$ is equivalent to minimising the **L2 reconstruction error** $\|x - \mu_{x|z}\|_2^2$.

> **Reparameterisation trick:** To allow gradients to flow through the sampling step, $z$ is written as $z = \mu_{z|x} + \sigma_{z|x} \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$ — the randomness is moved outside the computation graph.

### 4.4 VAEs in Medical Imaging — Unsupervised Anomaly Detection

**Core idea:** Train a VAE exclusively on **healthy images**. At inference time, the model cannot reconstruct pathological structures well — the reconstruction residual $\|x - x'\|$ becomes an **anomaly map**, requiring no pixel-level labels.

**Clinical applications:** Brain MRI for MS lesions / tumours / stroke; chest X-ray triage; OCT retinal scans (drusen, fluid pockets).

**Why VAEs fit clinical reality:** Healthy data is abundant; pathology is rare and diverse. The latent space can be conditioned on age, sex, or scanner to enable harmonisation across sites (Pinaya et al., 2022).

**Caveat:** VAE reconstructions are typically **blurry** due to the L2 loss, which may cause small subtle lesions to be missed.

---

## 5. Generative Adversarial Networks (GAN)

### 5.1 Two-Player Game Intuition

A GAN frames generation as a **minimax game** between two players:

| Player | Analogy | Goal |
|---|---|---|
| **Generator G** | Counterfeiter | Produce fake images indistinguishable from real ones |
| **Discriminator D** | Police | Distinguish real images from fakes |

Both networks improve iteratively: the generator gets better at fooling the discriminator, the discriminator gets better at detecting fakes.

### 5.2 Architecture

- **Generator G:** takes random noise $z \sim \mathcal{N}(0, I)$ as input, outputs a synthetic image $G(z)$
- **Discriminator D:** takes an image (real or fake) as input, outputs a scalar probability $D(x) \in [0, 1]$ representing how likely the image is real

### 5.3 Training Objective

Both networks are trained jointly by optimising a single minimax objective:

$$\min_G \max_D V(D, G) = \mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}(\boldsymbol{x})}\!\left[\log D(\boldsymbol{x})\right] + \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}\!\left[\log(1 - D(G(\boldsymbol{z})))\right]$$

- **Discriminator D** maximises $V$: wants $D(x) \approx 1$ (real) and $D(G(z)) \approx 0$ (fake)
- **Generator G** minimises $V$: wants $D(G(z)) \approx 1$ (fool the discriminator into accepting fakes as real)

At optimality, $D(x) = \frac{1}{2}$ everywhere — the discriminator cannot do better than random guessing.

### 5.4 CycleGANs

**CycleGAN** extends GANs to **unpaired image-to-image translation** — learning to translate between domains $X$ and $Y$ without requiring matched image pairs. Two generators ($G: X \to Y$ and $F: Y \to X$) and two discriminators are trained with an additional **cycle-consistency loss**: $F(G(x)) \approx x$ and $G(F(y)) \approx y$.

This is particularly relevant for medical imaging where paired data (e.g., same patient in both MRI and CT) is rare.

### 5.5 GANs in Medical Imaging

| Application | Example |
|---|---|
| **Cross-modality translation** | MR → synthetic CT for radiotherapy planning (avoids extra CT dose); CycleGAN for unpaired CT ↔ MRI, PET ↔ CT |
| **Image enhancement** | Low-dose CT denoising; MRI super-resolution and accelerated reconstruction |
| **Data augmentation** | StyleGAN for synthetic dermoscopy / chest X-ray / histology patches; GAN-augmented liver lesion classification (+7% sensitivity) |
| **Stain normalisation** | Harmonise H&E staining across labs and scanners |

**Watch out:** Mode collapse can hide rare pathology; hallucinated anatomical structures are a real risk.

---

## 6. Denoising Diffusion Probabilistic Models (DDPM)

### 6.1 Forward Diffusion Process

Diffusion models define a **fixed** (non-learned) forward process that gradually corrupts a real image $x_0$ into pure Gaussian noise $x_T$ over $T$ timesteps.

Each step adds a small amount of Gaussian noise, controlled by a **variance schedule** $\{\beta_t\}_{t=1}^T$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\right)$$

The joint distribution over all timesteps factorises as:

$$q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})$$

### 6.2 Diffusion Kernel — Direct Sampling

A key property of the Gaussian forward process is that we can sample $x_t$ **directly from $x_0$** without stepping through all intermediate timesteps. Defining $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

At $t = 0$: $x_t \approx x_0$ (original image). As $t \to T$: $x_t \approx \mathcal{N}(0, I)$ (pure noise). This closed-form expression makes training efficient.

### 6.3 Reverse Denoising Process

The goal of the model is to learn the **reverse process** — step by step removing noise to recover a clean image from pure noise:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\; \mu_\theta(x_t, t),\; \sigma_t^2 I\right)$$

where $\mu_\theta$ is a trainable neural network parameterised by $\theta$ (in practice a U-Net) that predicts the noise $\epsilon_\theta(x_t, t)$ added at step $t$.

The denoising joint distribution:

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)$$

> DDPMs are mathematically similar to **hierarchical VAEs** with a fixed encoder and a very deep latent hierarchy (one level per timestep).

### 6.4 Training and Model Architecture

**Training objective:** Minimise the simplified noise-prediction loss (which is equivalent to the ELBO):

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\left\|\epsilon - \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\; t\right)\right\|^2\right]$$

The network learns to predict the noise $\epsilon$ that was added to $x_0$ to produce $x_t$.

**Architecture:** DDPMs typically use **U-Net** backbones with ResNet blocks and self-attention layers. Time conditioning is provided via sinusoidal positional embeddings or random Fourier features, fed into residual blocks via addition or adaptive group normalisation.

### 6.5 Image Generation Procedure

To generate an image:
1. Sample pure noise: $x_T \sim \mathcal{N}(0, I)$
2. For $t = T, T-1, \ldots, 1$: compute $x_{t-1} = \text{model}(x_t, t)$
3. Return $x_0$

The main practical drawback is that the model must be applied $T$ times sequentially (typically $T = 1000$), making naive generation slow. Much research has been devoted to accelerating sampling (DDIM, consistency models, etc.).

### 6.6 Diffusion Models in Medical Imaging

| Application | Examples |
|---|---|
| **Conditional generation from reports** | RoentGen (text → chest X-ray); Cheff (controllable CXR synthesis for classifier training) |
| **Reconstruction from undersampled measurements** | Score-based MRI reconstruction; sparse-view and low-dose CT |
| **Segmentation and inpainting** | MedSegDiff (diffusion as segmentation backbone); lesion inpainting for counterfactual explanations |
| **3D and multi-modal** | Latent diffusion for whole-volume brain MRI; cross-modality synthesis MR → CT and PET attenuation correction |

---

## 7. Evaluating Generative Models

### 7.1 Structural Similarity Index Measure (SSIM)

SSIM measures **perceptual similarity** between a reference image $x$ and a generated image $y$ by combining three components: luminance comparison, contrast comparison, and structure comparison.

$$\text{SSIM}(x, y) = l(x,y)^{\alpha} \cdot c(x,y)^{\beta} \cdot s(x,y)^{\gamma}$$

where the three terms are computed from local mean, variance, and covariance of the two images. SSIM $\in [-1, 1]$, with 1 indicating perfect similarity.

SSIM is pixel-level and requires a reference image — it is useful for reconstruction tasks (low-dose → full-dose CT) but not for unconditional generation.

### 7.2 Fréchet Inception Distance (FID)

FID measures the **distributional similarity** between real and generated images at the feature level, without needing paired images.

**Procedure:**
1. Pass both real and generated image sets through a pretrained **Inception-v3** network to extract feature embeddings
2. Fit a multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$ to the real features and $\mathcal{N}(\hat{\mu}, \hat{\Sigma})$ to the generated features
3. Compute the Fréchet (Wasserstein-2) distance between the two Gaussians:

$$\text{FID} = \|\mu - \hat{\mu}\|^2 + \text{tr}\!\left(\Sigma + \hat{\Sigma} - 2\sqrt{\Sigma \hat{\Sigma}}\right)$$

Lower FID = more realistic images whose distribution matches the training set. FID is the **de facto standard** for unconditional generation but does not capture diagnostic correctness.

### 7.3 CLIP Distance

CLIP distance measures semantic similarity in the joint embedding space of a vision-language model (CLIP). It is useful for **text-conditioned generation** tasks: does the generated image match the text prompt? Less common in purely image-to-image medical imaging tasks.

### 7.4 Clinical Evaluation Pyramid

**Image quality metrics alone are not sufficient** — a perceptually realistic image can still contain anatomically impossible or diagnostically wrong structures.

| Level | What is measured |
|---|---|
| **(1) Technical** | SSIM, FID, representation-based metrics on held-out scans |
| **(2) Task-based** | Does a downstream classifier / segmenter improve? |
| **(3) Reader study** | Blinded radiologists score realism and detectability (Turing-style) |
| **(4) Clinical utility** | Impact on diagnosis time, treatment decisions, patient outcomes |

Most published work stops at level 1 or 2. Levels 3 and 4 are rarely reported but are required for regulatory approval.

---

## 8. Clinical Workflow Integration

Generative AI enters the clinical pipeline at multiple stages:

**Acquisition:** MRI reconstruction from undersampled k-space (faster scans); low-dose CT denoising (reduced patient radiation exposure).

**Pre-processing:** Harmonisation across scanners and sites (domain adaptation); stain normalisation in digital pathology.

**Model development:** Synthetic data augmentation for rare pathologies; privacy-preserving data sharing between hospitals (federated + synthetic data).

**Reporting and decision support:** Report drafting from images using vision-language models.

**Education and simulation:** Synthetic teaching cases of rare findings for radiology residents.

---

## 9. Risks Specific to Generative Medical Imaging

| Risk | Description |
|---|---|
| **Hallucinated anatomy** | Models can invent lesions, vessels, or organs that look plausible but are anatomically wrong. Especially dangerous for reconstruction tasks (MRI, CT). |
| **Bias and fairness** | Training sets skew toward Western, light-skinned, adult populations. Synthetic data can inherit and amplify these biases. |
| **Memorisation and privacy leakage** | Diffusion models can regurgitate near-copies of training images. Re-identification risk from synthetic patient data is non-zero. |
| **Distribution shift** | A new scanner, acquisition protocol, or patient population causes silent degradation of generation quality. |
| **Evaluation gap** | Most papers report only FID; very few include reader studies or prospective clinical validation. |

> **Regulatory note:** Generative software as a medical device (SaMD) will need new regulatory approval pathways (FDA, MDR in the EU). The field is catching up.

---

## 10. Key Takeaways

- Generative models address medicine's hardest data problems: **scarcity, privacy, class imbalance, and modality gaps**.
- Each architecture occupies a distinct clinical niche: **VAE → unsupervised anomaly detection; GAN → modality translation and augmentation; Diffusion → high-fidelity synthesis, reconstruction, and report-conditioned generation.**
- **Realism ≠ clinical validity.** Always evaluate on a downstream clinical task, not just FID.
- Image quality metrics (FID, SSIM, CLIP) are necessary but not sufficient. The full validation pyramid — technical, task-based, reader study, clinical utility — is required before deployment.
- Be explicit about failure modes: **hallucination, bias, and memorisation are patient-relevant risks**, not just theoretical concerns.
- Regulation is catching up — generative SaMD will face new approval pathways.

---

## 11. Key Papers

| Paper | Contribution |
|---|---|
| Goodfellow et al. (2014). *Generative adversarial nets.* NeurIPS. | Original GAN formulation; minimax training objective |
| Kingma & Welling (2014). *Auto-encoding variational Bayes.* ICLR. | VAE: ELBO derivation, reparameterisation trick |
| Ho et al. (2020). *Denoising diffusion probabilistic models.* NeurIPS. | DDPM: forward/reverse process, simplified training objective |
| Zhu et al. (2017). *Unpaired image-to-image translation using cycle-consistent adversarial networks.* ICCV. | CycleGAN for unpaired domain translation |
| Pinaya et al. (2022). *Unsupervised brain imaging 3D anomaly detection and segmentation with transformers.* Medical Image Analysis. | VAE-based anomaly detection in brain MRI |
| Rombach et al. (2022). *High-resolution image synthesis with latent diffusion models.* CVPR. | Latent diffusion models — computationally efficient diffusion in latent space |
| Wang et al. (2023). *RoentGen: Vision-language foundation model for chest X-ray generation.* | Text-conditioned chest X-ray synthesis from radiology reports |
| Adamkiewicz et al. (2026). *When Pretty Isn't Useful: Investigating Why Modern Text-to-Image Models Fail as Reliable Training Data Generators.* CVPR. | Limitations of synthetic data from generative models for downstream tasks |

---

*Notes compiled from: 07_MEDIMG_generative_(1).pdf — HSLU Medical Image Analysis, Fabian Gröger*
