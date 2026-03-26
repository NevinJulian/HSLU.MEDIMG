# Week 06 — Image Segmentation

**Course:** Medical Image Analysis — HSLU  
**Lecturer:** Dr. Ludovic Amruthalingam  
**Topics:** Segmentation types · Classical approaches · Deep learning (2D/3D) · Pipeline engineering · Noisy labels · Evaluation & robustness

---

## Table of Contents

1. [What is Segmentation?](#1-what-is-segmentation)
2. [Segmentation Tasks](#2-segmentation-tasks)
   - [Semantic Segmentation](#21-semantic-segmentation)
   - [Instance Segmentation](#22-instance-segmentation)
   - [Panoptic Segmentation](#23-panoptic-segmentation)
   - [Binary vs Multi-class vs Multi-label](#24-binary-vs-multi-class-vs-multi-label)
3. [Segmentation Pipeline](#3-segmentation-pipeline)
4. [Classical Segmentation Methods](#4-classical-segmentation-methods)
   - [Threshold-based Segmentation](#41-threshold-based-segmentation)
   - [Edge-based Segmentation — Canny](#42-edge-based-segmentation--canny)
   - [Region-based Segmentation — Watershed](#43-region-based-segmentation--watershed)
5. [Deep Learning Segmentation](#5-deep-learning-segmentation)
   - [2D Segmentation — U-Net](#51-2d-segmentation--u-net)
   - [Transposed Convolution](#52-transposed-convolution--learned-upsampling)
   - [Data Augmentation](#53-data-augmentation)
   - [2D vs 3D Segmentation](#54-2d-vs-3d-segmentation)
   - [1D vs 2D vs 3D Convolution](#55-1d-vs-2d-vs-3d-convolution)
   - [3D Segmentation — V-Net](#56-3d-segmentation--v-net)
   - [U-Net Variants](#57-u-net-variants)
   - [Model Architecture vs Pipeline — nnU-Net v2](#58-model-architecture-vs-pipeline--nnu-net-v2)
   - [Promptable Segmentation — MedSAM](#59-promptable-segmentation--medsam)
6. [Training & Pipeline Engineering](#6-training--pipeline-engineering)
   - [Patch-based Training and Inference](#61-patch-based-training-and-inference)
   - [Postprocessing](#62-postprocessing)
   - [Loss Functions](#63-loss-functions)
7. [Noisy Labels](#7-noisy-labels)
   - [Label Noise](#71-label-noise)
   - [T-Loss — Robust Loss under Noisy Annotations](#72-t-loss--robust-loss-under-noisy-annotations)
8. [Evaluation & Robustness](#8-evaluation--robustness)
   - [Overlap Metrics — Dice & IoU](#81-overlap-metrics--dice--iou)
   - [Surface / Boundary Metrics](#82-surface--boundary-metrics)
   - [Lesion-wise Metrics](#83-lesion-wise-metrics)
   - [Statistical Validation and Confidence Intervals](#84-statistical-validation-and-confidence-intervals)
   - [Evaluation Pitfalls & Best Practices](#85-evaluation-pitfalls--best-practices)
   - [Robust and Uncertainty-aware Inference](#86-robust-and-uncertainty-aware-inference)
   - [Checklist Before Trusting Your Model](#87-checklist-before-trusting-your-model)
9. [Key Takeaways](#9-key-takeaways)
10. [Key Papers](#10-key-papers)

---

## 1. What is Segmentation?

Segmentation assigns a **label to every pixel or voxel** in an image, partitioning it into meaningful regions (e.g., background vs. organ, tumour, vessel, lesion).

**Why it matters — segmentation turns images into measurable objects:**

| Use case | Examples |
|---|---|
| **Quantification** | Surface area, volume, diameter of structures |
| **Planning** | Surgery / radiotherapy: organs-at-risk, target volumes |
| **Monitoring** | Change over time — baseline vs. follow-up |

**Typical model outputs:**
- A **probability map** per class (one value per pixel per class)
- After **thresholding** → binary or multi-class mask

> *Example tool: TotalSegmentator, available as a 3D Slicer extension, automatically segments 104 anatomical structures in CT scans.*

---

## 2. Segmentation Tasks

### 2.1 Semantic Segmentation

Assigns each pixel/voxel **one of K classes** (including background). All pixels of the same tissue type receive the same label — individual instances are not distinguished.

### 2.2 Instance Segmentation

Separates **individual objects of the same class** (e.g., distinguishing between multiple separate lesions). Usually requires postprocessing (connected components analysis) or specialised model heads.

### 2.3 Panoptic Segmentation

Combines semantic and instance segmentation into a single unified output — every pixel is assigned both a class and an instance identity.

### 2.4 Binary vs Multi-class vs Multi-label

| Type | Description | Activation |
|---|---|---|
| **Binary** | Background vs. a single foreground class | Sigmoid |
| **Multi-class** | Mutually exclusive classes — one label per pixel | Softmax |
| **Multi-label** | Overlapping labels — multiple classes can be present at the same pixel | Sigmoid per class |

---

## 3. Segmentation Pipeline

A complete end-to-end segmentation pipeline consists of seven stages:

1. **Data + labels + spacing metadata** — gather images, ground-truth masks, and voxel spacing information
2. **Preprocessing** — resample voxel spacing, intensity normalisation, cropping
3. **Model training** — define loss, data augmentation strategy, patch sampling
4. **Inference on full volumes** — tiling / sliding window approach
5. **Postprocessing** — connected components analysis, morphological operations
6. **Evaluation** — overlap metrics + surface metrics + lesion-wise metrics + statistical uncertainty
7. **Robustness checks + failure detection**

---

## 4. Classical Segmentation Methods

Classical methods remain relevant for:
- **Intuition & sanity checks** — understanding which image cues separate classes
- **Baselines** — in low-data settings, a simple threshold can sometimes be hard to beat
- **Pre/post-processing** — morphological operations, connected components
- **Failure analysis** — understanding why deep learning fails on specific artefacts

### 4.1 Threshold-based Segmentation

Assumes **intensity separates classes**. Two main approaches:

| Method | Description |
|---|---|
| **Global thresholding** | Split pixels based on a fixed intensity threshold $t$ |
| **Otsu's method** | Automatically pick $t$ to maximise between-class variance / minimise within-class variance |

**Work well when:**
- Classes have clearly distinct intensities
- High contrast (e.g., bone vs. soft tissue in CT)

**Fail when:**
- Intensity overlap between classes
- Inhomogeneity or imaging artefacts
- **Partial volume effects** — a voxel contains multiple tissue types; the measured intensity is an average of both

### 4.2 Edge-based Segmentation — Canny

Detects **object boundaries** (edges) by identifying sharp intensity changes.

**Canny edge detection steps:**
1. Smooth with a **Gaussian filter** to reduce noise
2. Compute **image gradients** and determine edge strength
3. **Non-maximum suppression** — thin detected edges to single-pixel width
4. **Hysteresis thresholding** — keep strong edges and any weak edges connected to them

**Output:** An edge map (not a filled region mask) → segmentation requires a subsequent filling step.

**Limitation:** Does not produce regions directly; sensitive to threshold choices.

### 4.3 Region-based Segmentation — Watershed

Treats the image as a **topographic surface** where intensity, gradient magnitude, or distance values correspond to elevation.

- Flooding starts from **markers** (seed points)
- Water basins spread until they meet → **boundaries become the segmentation**

**Two common surface definitions:**
- **Gradient magnitude** — high values mark object boundaries, stopping flooding
- **Distance transform** (for instance separation):
  1. Start from a binary mask (e.g., from thresholding)
  2. Compute each pixel's distance to the nearest background pixel
  3. Local maxima of the distance map become markers

**Common failure mode:** Over-segmentation → mitigated by improving markers, smoothing, or merging regions.

---

## 5. Deep Learning Segmentation

### 5.1 2D Segmentation — U-Net

The **U-Net** (Ronneberger et al., MICCAI 2015) is the standard baseline for medical image segmentation. It follows an **encoder–decoder** architecture:

| Path | Role |
|---|---|
| **Encoder** | Successive downsampling — extracts context features (*what structures are present?*) |
| **Decoder** | Successive upsampling + skip connections — recovers spatial detail (*where are the boundaries?*) |

**Skip connections** copy high-resolution feature maps from encoder layers to the corresponding decoder layers:
- Features near the input are **fine-grained** (edges, textures)
- Features at the bottleneck are **abstract** (semantics)
- Skip connections allow the decoder to recover fine boundary details lost during downsampling

**Strong data augmentation** is critical for learning from small labelled datasets:
- For microscopy images: random shift, rotation, elastic deformations, and grey value variations

### 5.2 Transposed Convolution — Learned Upsampling

The decoder uses **transposed convolutions** (also called deconvolutions) for learnable upsampling.

**Intuition:** Each input pixel is spread over a larger output area; the kernel weights determine how to fill the gaps.

- Unlike bilinear upsampling, transposed convolutions have **learnable parameters**
- Stride $> 1$ increases the output spatial resolution

### 5.3 Data Augmentation

Augmentations should reflect **plausible real-world variation** and must be applied **consistently to both image and label mask**.

| Category | Examples |
|---|---|
| **Spatial** | Random flips, rotations, elastic deformations, scaling |
| **Intensity** | Gaussian noise, MRI bias field, contrast, gamma, blur |
| **Acquisition artefacts** | Simulated motion artefacts |

**Two goals:**
- Reduce **overfitting** to the training set
- Improve **robustness** to acquisition differences across scanners/sites

**Best practices:**
- Always **visualise** augmented samples on real data before training
- **Ablate** augmentation types and strengths to understand their individual contributions

> *Recommended library: TorchIO — Python library for efficient loading, preprocessing, augmentation, and patch-based sampling of medical images (Pérez-García et al., 2021)*

### 5.4 2D vs 3D Segmentation

CT and MRI data are inherently **3D volumes** — voxels, not pixels. Key metadata: **voxel spacing**, which is often anisotropic (lower resolution in the z-axis).

**Preprocessing:** Resample to isotropic or near-isotropic voxel spacing before training.

**Modelling choices:**

| Approach | Description | Trade-offs |
|---|---|---|
| **2D** | Segment slice by slice | Cheap; may miss 3D context |
| **2.5D** | Stack neighbouring slices as input channels | Lightweight 3D context |
| **3D** | Full 3D convolutions | Best context; high memory cost |

**When to use which:**
- **2D** — thick slices, limited compute, or when 3D context provides little clinical value
- **3D** — thin slices, complex 3D structures, small lesions, volumetric measurement tasks
- Consider 2.5D as an **ablation baseline**

### 5.5 1D vs 2D vs 3D Convolution

| | 1D Conv | 2D Conv | 3D Conv |
|---|---|---|---|
| **Input shape** | $L \times C_{in}$ | $H \times W \times C_{in}$ | $H \times W \times D \times C_{in}$ |
| **Kernel shape** | $k \times C_{in}$ | $k^2 \times C_{in}$ | $k^3 \times C_{in}$ |
| **Sliding dimensions** | Length | Height, Width | Height, Width, Depth |
| **Output shape** | $L' \times C_{out}$ | $H' \times W' \times C_{out}$ | $H' \times W' \times D' \times C_{out}$ |
| **# Parameters** | $k \cdot C_{in} \cdot C_{out}$ | $k^2 \cdot C_{in} \cdot C_{out}$ | $k^3 \cdot C_{in} \cdot C_{out}$ |
| **# Operations** | $L \cdot k \cdot C_{in} \cdot C_{out}$ | $H \cdot W \cdot k^2 \cdot C_{in} \cdot C_{out}$ | $H \cdot W \cdot D \cdot k^3 \cdot C_{in} \cdot C_{out}$ |

The cubic scaling of parameters and operations is the core reason 3D models are memory-intensive.

### 5.6 3D Segmentation — V-Net

The **V-Net** (Milletari et al., 3DV 2016) extends the U-Net idea to volumetric data:

- Replaces all 2D convolutions and pooling with **3D convolutions and pooling**
- Higher memory cost → requires **smaller batch sizes** and **patch-based training**
- Data augmentation adapted to 3D: dense deformation fields, intensity variations

### 5.7 U-Net Variants

Many architectures build on the U-Net blueprint by modifying skip connections, replacing convolutions with attention mechanisms, or using larger kernels:

| Architecture | Key innovation |
|---|---|
| **UNet++** | Nested and dense skip connections |
| **UNet 3+** | Full-scale skip connections |
| **Swin U-Net** | Transformer-based encoder (Swin Transformer blocks replace convolutions) |
| **MedNeXt** | Large-kernel ConvNeXt blocks adapted for medical 3D volumes |

### 5.8 Model Architecture vs Pipeline — nnU-Net v2

> **Key insight:** Many state-of-the-art gains in segmentation come from **pipeline engineering**, not novel architectures.

The U-Net family dominates because it provides **multi-scale context + precise localisation**. Critical pipeline choices include: preprocessing, resampling strategy, patch sampling, augmentation quality, inference tiling, and robust evaluation.

**nnU-Net v2** (Isensee et al., MICCAI 2024) automates all these design decisions based on a *data fingerprint* (median shape, spacing distribution, intensity distribution, image modality):

1. Rule-based parameters — image target spacing, resampling strategy, patch size, network topology
2. Empirical parameters — ensemble selection, postprocessing configuration
3. Fixed parameters — optimizer, learning rate schedule, augmentation, loss function

nnU-Net remains a **highly competitive baseline** across diverse segmentation tasks without any task-specific tuning.

### 5.9 Promptable Segmentation — MedSAM

**Segment Anything Model (SAM)** is a foundation model for segmentation, trained on the massive SA-1B dataset. Architecture:
- **Image encoder** (Vision Transformer / ViT)
- **Prompt encoder** (accepts points, bounding boxes, or mask prompts)
- **Mask decoder**

**MedSAM** adapts SAM to medical images (CT, MRI, ultrasound):
- Fine-tuned on large-scale medical image datasets
- Better handles low contrast, modality-specific intensity ranges, and anatomical features
- Interaction via **bounding-box prompts**

**Practical use:** MedSAM can assist and accelerate the **annotation process** for new datasets.

*Reference: Ma et al., "Segment anything in medical images." Nature Communications, 2024.*

---

## 6. Training & Pipeline Engineering

### 6.1 Patch-based Training and Inference

Segmentation datasets are often **extremely class-imbalanced** (background dominates). Additionally, full 3D volumes typically do not fit in GPU memory.

**Solution: patch-based training** — crop fixed-size patches and train on those.

**Patch sampling strategies** (cf. TorchIO patch samplers):

| Strategy | Description |
|---|---|
| **Uniform random crops** | Fast and simple; may miss tiny lesions |
| **Positive/negative sampling** | Explicitly balance the ratio of lesion-containing vs. background patches |
| **Weighted sampling** | Sample more frequently near boundaries or from rare classes |

**Sliding window inference:** At test time, apply the model to many overlapping windows across the full volume, then aggregate (e.g., average) the overlapping predictions.

### 6.2 Postprocessing

After model inference, simple rule-based postprocessing can substantially improve results.

**Common steps:**
- **Connected components labelling** — identify separate components in the binary mask; keep only the largest (appropriate for single-organ tasks)
- **Remove small objects** below a size threshold (suppress false positives)
- **Morphological closing** — fill small holes inside the predicted region
- **Morphological opening** — remove small speckle artefacts at the boundary

> **Important:** Postprocessing is **task-dependent** and must be justified and ablated — do not apply blindly.

### 6.3 Loss Functions

Segmentation losses must simultaneously handle **per-pixel classification** and **extreme class imbalance**. Boundary sensitivity is often clinically important.

| Loss | Description |
|---|---|
| **Cross-entropy (CE)** | Standard per-pixel classification loss |
| **Weighted CE** | Upweights minority class pixels |
| **Dice loss** | Directly optimises overlap between prediction and ground truth |
| **Generalised Dice** | Multi-class extension that handles class imbalance |
| **Boundary loss** | Adds a boundary-focused signal (distance-map weighted) |
| **Focal loss** | Down-weights easy examples, focuses on hard pixels |
| **Combo losses** | E.g., Dice + CE — combines region and pixel-level signals |

> **Best practice:** Optimise the loss that most closely aligns with your **evaluation metric and clinical use case**.

*Reference: Ma et al., "Loss odyssey in medical image segmentation." Medical Image Analysis, 2021.*

---

## 7. Noisy Labels

### 7.1 Label Noise

Manual segmentation is **time-consuming, subjective, and inherently noisy**:

| Source | Examples |
|---|---|
| **Random errors** | Annotator fatigue, unintentional mistakes |
| **Systematic bias** | Different annotation guidelines across raters; varying levels of expertise |
| **Boundary ambiguity** | Partial volume effects, low contrast between tissue types |

**Mitigation strategies:**
- Perform **quality checks** and annotation review
- Report **inter-rater agreement** (e.g., Cohen's kappa, Dice between raters)
- Use **STAPLE** (Warfield et al., 2004) for consensus ground-truth estimation:
  - Formalises the absence of a true gold standard
  - Uses an **expectation-maximisation (EM) algorithm** to jointly estimate a probabilistic ground truth and per-rater performance
- Use **robust loss functions** that down-weight uncertain boundary regions

### 7.2 T-Loss — Robust Loss under Noisy Annotations

Standard losses (CE, Dice) assume **clean labels** and over-penalise outliers (mislabelled pixels), which can destabilise training.

**T-Loss** models prediction errors with a **Student-t distribution** instead of a Gaussian:

- For **small errors** → behaves similarly to L2 / CE (normal training signal)
- For **large errors** → the penalty **saturates** (heavy-tailed distribution)
- Mislabelled pixels receive **reduced influence** during training

**Benefits:**
- Focus on the consistent signal from the correctly labelled majority of pixels
- Avoid overfitting to mislabelled boundary regions
- More stable training under noisy annotations

*Reference: Gonzalez-Jimenez et al., "Robust t-loss for medical image segmentation." Medical Image Analysis, 2025.*

---

## 8. Evaluation & Robustness

### 8.1 Overlap Metrics — Dice & IoU

Let $P$ be the predicted mask and $G$ the ground-truth mask.

$$\text{IoU}(G, P) = \frac{|G \cap P|}{|G \cup P|} = \frac{TP}{TP + FP + FN}$$

$$\text{Dice}(G, P) = \frac{2 \cdot |G \cap P|}{|G| + |P|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

**Relationship:** $\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$

**Sensitivity to lesion size:**
- On **small lesions**, a few voxels can swing the score dramatically
- High Dice on large lesions may hide clinically important boundary errors
- Low Dice on a tiny lesion may reflect only a few voxels of error

### 8.2 Surface / Boundary Metrics

Let $X$ and $Y$ be sets of boundary points (surface voxels) of the prediction and ground truth respectively.

**Hausdorff Distance (HD)** — captures the largest single deviation between contours:

$$HD(X, Y) = \max\!\left(\max_{x \in X}\min_{y \in Y} d(x,y),\; \max_{y \in Y}\min_{x \in X} d(x,y)\right)$$

- Very sensitive to **outliers / noise** → in practice, use **HD95** (95th percentile of surface distances)

**Average Surface Distance (ASD)** — mean distance between all boundary points:

$$ASD(X, Y) = \frac{1}{|X| + |Y|}\left(\sum_{x \in X} d(x, Y) + \sum_{y \in Y} d(y, X)\right)$$

**Normalised Surface Dice (NSD)** — overlap of surfaces within a tolerance distance $\tau$ (clinically motivated):

$$NSD(X, Y) = \frac{|\{x \in X \mid d(x, Y) < \tau\}| + |\{y \in Y \mid d(y, X) < \tau\}|}{|X| + |Y|}$$

### 8.3 Lesion-wise Metrics

Global Dice scores can be **dominated by large lesions**. Missing a small but clinically critical lesion (e.g., a small metastasis) may be invisible in the aggregate score.

**Lesion-wise evaluation procedure:**
1. Identify **connected components** (individual lesions) in both prediction and ground truth
2. **Match** predicted lesions to ground-truth lesions using overlap-based matching
3. Compute **per-lesion metrics** (Dice, HD95)
4. **Aggregate** — report mean, median, and lesion-level recall (fraction of ground-truth lesions detected)

This jointly measures **detection quality** (did we find the lesion?) and **segmentation quality** (how accurately did we delineate it?).

### 8.4 Statistical Validation and Confidence Intervals

**Always compute metrics per image per patient** (not globally pooled):
- Avoids bias from different object sizes and varying case difficulty
- Global mean alone can **hide failure cases**

**Report uncertainty:**
- Show per-case metric distributions (boxplot or violin plot), median + interquartile range (IQR)
- Metric distributions are often **non-Gaussian (heavy-tailed)** → use **bootstrap sampling** over cases to estimate 95% confidence intervals

**Model comparison:**
- Use **paired statistical tests** (e.g., Wilcoxon signed-rank test, permutation test)
- Paired tests control for **case difficulty**, making comparisons fairer

### 8.5 Evaluation Pitfalls & Best Practices

> **Key principle: No single metric captures all clinically important errors.**

**Common pitfalls:**

| Pitfall | Consequence |
|---|---|
| Data leakage / test-set tuning | Inflated reported performance |
| Using only Dice | Boundary errors remain hidden |
| Reporting mean without uncertainty | Instability and outliers go undetected |
| Incorrect metric implementations | Multi-class handling errors; ignoring voxel spacing in surface metrics |

**Recommendations:**
- Report **overlap + surface + lesion-wise** metrics together
- Use **case-level statistics + confidence intervals**
- Follow published guidelines, e.g., Müller et al., "Towards a guideline for evaluation metrics in medical image segmentation." *BMC Research Notes*, 2022.

### 8.6 Robust and Uncertainty-aware Inference

**Two types of uncertainty:**

| Type | Description |
|---|---|
| **Aleatoric** | Inherent ambiguity / noise in the data itself (irreducible) |
| **Epistemic** | Model uncertainty — reducible with more data or better models |

**Test-Time Augmentation (TTA):**
- Apply augmentations at **inference time** and **average** the resulting predictions
- Can improve accuracy without any retraining
- Captures **sensitivity to input perturbations** (aleatoric uncertainty proxy)

**Test-Time Dropout (TTD) — Monte-Carlo Dropout:**
- Keep **dropout layers active** at inference (normally disabled after training)
- Sample multiple stochastic forward passes
- Estimate prediction **expectation** (final segmentation) and **variance** (uncertainty map)
- Captures **model uncertainty** (epistemic uncertainty proxy)

Both methods produce **uncertainty maps** that can highlight unreliable regions — useful for flagging cases that may need human review.

### 8.7 Checklist Before Trusting Your Model

> A good metric score is **not sufficient** on its own.

- [ ] **Patient-level split** — no data leakage between train and test sets
- [ ] **Metrics reflect the task** — Dice alone is not enough
- [ ] **Confidence intervals** reported
- [ ] **Visual inspection** of predictions on representative and failure cases
- [ ] **Tested on out-of-distribution data** — different scanner, site, or patient population

---

## 9. Key Takeaways

- Segmentation success depends on **pipeline discipline** as much as model architecture
- **Start with strong baselines** — U-Net or nnU-Net v2 before exploring novel architectures
- **Evaluate thoroughly:** overlap + surface + lesion-wise metrics + uncertainty estimates (confidence intervals)
- **Treat ground truth carefully** — report inter-rater agreement; use STAPLE for consensus; acknowledge label ambiguity
- **Robustness is part of the task** — augmentation, TTA, TTD, and uncertainty maps are not optional extras

---

## 10. Key Papers

| Paper | Contribution |
|---|---|
| Ronneberger et al. (2015). *U-net: Convolutional networks for biomedical image segmentation.* MICCAI. | Foundational encoder-decoder architecture for 2D medical segmentation |
| Milletari et al. (2016). *V-net: Fully convolutional neural networks for volumetric medical image segmentation.* 3DV. | Extension of U-Net to 3D volumes; introduces Dice loss |
| Isensee et al. (2024). *nnU-Net revisited: A call for rigorous validation in 3D medical image segmentation.* MICCAI. | Automated pipeline engineering; strong baseline across tasks |
| Ma et al. (2021). *Loss odyssey in medical image segmentation.* Medical Image Analysis. | Comprehensive survey of segmentation loss functions |
| Ma et al. (2024). *Segment anything in medical images.* Nature Communications. | MedSAM — promptable foundation model for medical segmentation |
| Warfield et al. (2004). *STAPLE: An algorithm for the validation of image segmentation.* IEEE TMI. | EM-based consensus ground truth estimation from multiple raters |
| Gonzalez-Jimenez et al. (2025). *Robust t-loss for medical image segmentation.* Medical Image Analysis. | Noise-robust loss function based on Student-t distribution |
| Müller et al. (2022). *Towards a guideline for evaluation metrics in medical image segmentation.* BMC Research Notes. | Best-practice recommendations for segmentation evaluation |
| Pérez-García et al. (2021). *TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning.* Computer Methods and Programs in Biomedicine. | Standard library for 3D medical image augmentation |

---

*Notes compiled from: 06_MEDIMG_segmentation.pdf — HSLU Medical Image Analysis, Dr. Ludovic Amruthalingam*
