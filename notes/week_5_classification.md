# Week 05 — Image Classification

**Course:** Medical Image Analysis — HSLU  
**Lecturer:** Nipun Sandamal (with slides from Simone Lionetti)  
**Topics:** Classification tasks · CNN building blocks · Convolution · Pooling · Activation functions · Batch normalisation · Transfer learning · Probability calibration · Evaluation metrics · Robust model tips

---

## Table of Contents

1. [Introduction to Image Classification](#1-introduction-to-image-classification)
   - [Medical Classification Tasks](#11-medical-classification-tasks)
   - [Types of Classification Tasks](#12-types-of-classification-tasks)
   - [Label Quality](#13-label-quality)
   - [Label Aggregation](#14-label-aggregation)
2. [Data Splits and Data Leakage](#2-data-splits-and-data-leakage)
3. [Building Blocks of CNNs](#3-building-blocks-of-cnns)
   - [CNN Architecture Overview](#31-cnn-architecture-overview)
   - [2D Convolution](#32-2d-convolution)
   - [Stride and Padding](#33-stride-and-padding)
   - [Number of Filters and Parameters](#34-number-of-filters-and-parameters)
   - [Sub-Sampling / Pooling](#35-sub-sampling--pooling)
   - [Activation Functions](#36-activation-functions)
   - [Batch Normalisation](#37-batch-normalisation)
   - [Fully Connected Layers](#38-fully-connected-layers)
   - [Dropout](#39-dropout)
   - [Training a ConvNet](#310-training-a-convnet)
   - [Softmax](#311-softmax)
4. [Probability Calibration](#4-probability-calibration)
5. [Modern Architectures and Transfer Learning](#5-modern-architectures-and-transfer-learning)
6. [Evaluation Metrics](#6-evaluation-metrics)
   - [Confusion Matrix](#61-confusion-matrix)
   - [Standard Metrics](#62-standard-metrics)
   - [Random Baselines](#63-random-baselines)
   - [ROC and AUC](#64-roc-and-auc)
   - [F1 Score](#65-f1-score)
   - [Matthews Correlation Coefficient](#66-matthews-correlation-coefficient)
   - [Multi-class Extensions](#67-multi-class-extensions)
   - [Pitfalls](#68-pitfalls)
7. [Tips for Robust Classification Models](#7-tips-for-robust-classification-models)
   - [Cross-Validation](#71-cross-validation)
   - [Hyperparameter Tuning](#72-hyperparameter-tuning)
   - [Ablation Studies](#73-ablation-studies)

---

## 1. Introduction to Image Classification

Image classification is something humans do automatically and continuously — recognising faces, objects, and scenes without conscious effort. For a computer, however, an image is simply a **grid of numerical values**; it has no inherent semantic meaning until a model is trained to assign one.

In the medical domain, classification is more precisely called **fine-grained classification**: the differences between classes can be extremely subtle (e.g., distinguishing subtypes of dermatitis), and meaningful visual cues may be drowned out by background clutter, varying illumination, or intra-class variation.

**Key visual challenges in medical imaging:**

| Challenge | Description |
|---|---|
| **Fine-grained distinction** | Different disease subtypes look nearly identical (e.g., urticaria variations) |
| **Intra-class variation** | Same condition appears very differently across patients, body sites, lighting |
| **Background clutter** | Lesions blend into surrounding healthy tissue |
| **Illumination changes** | Same lesion photographed seconds apart can look entirely different |

---

### 1.1 Medical Classification Tasks

Medical image classification is not merely about attaching a label — the output drives **clinical action**:

- **Disease detection** — e.g., pneumonia present or absent on chest X-ray
- **Severity grading** — e.g., tumour stage 1–4
- **Triage** — urgent vs. non-urgent referral

The output of a classifier feeds into decision support, not just a label. This makes **probability calibration** and threshold selection clinically critical. A model that outputs "70% probability of cancer" must be trustworthy enough for a clinician to decide: biopsy, monitor, or discharge?

---

### 1.2 Types of Classification Tasks

| Type | Description | Example |
|---|---|---|
| **Binary** | Two classes | Disease vs. no disease |
| **Multi-class** | One label from several classes | Skin disease type |
| **Multi-label** | Multiple labels simultaneously | Chest X-ray with several concurrent findings |
| **Ordinal** | Ordered severity categories | Eczema severity grades 0–4 |

> Medical tasks are often **multi-label** by nature — a single chest X-ray can show pneumonia, pleural effusion, and cardiomegaly simultaneously.

---

### 1.3 Label Quality

Labels in medical AI almost never represent ground truth with certainty. Understanding label quality is essential for designing training pipelines and interpreting results.

**Sources of labels:**
- Expert annotations (radiologists, dermatologists)
- Automated text mining from radiology reports

**Sources of label uncertainty:**
- **Noise** — labelling errors, ambiguous images
- **Disagreement** — inter-rater variability; different experts label the same image differently
- **Missing labels** — not all relevant findings may be annotated

---

### 1.4 Label Aggregation

When multiple annotators provide labels for the same image, their labels must be reconciled. This is called **label aggregation**.

**Methods (in order of increasing sophistication):**

| Method | Description | Limitation |
|---|---|---|
| **Majority Voting** | Take the most common label | Does not account for annotator expertise or task difficulty |
| **Weighted Majority Voting** | Weight votes by annotator reliability | Requires reliability estimates |
| **Dawid–Skene Algorithm** | EM algorithm estimating annotator confusion matrices | Computationally more involved |
| **GLAD** | Jointly estimates label difficulty and annotator ability | Most principled, highest complexity |

> Majority voting is intuitive, efficient, and performs well in practice — but since it ignores annotator ability and task difficulty, more probabilistic approaches are often needed in clinical settings.

---

## 2. Data Splits and Data Leakage

### Standard Split Strategy

Data is divided into three non-overlapping subsets:

| Split | Typical Size | Purpose |
|---|---|---|
| **Training** | 60% | Fit model parameters (via gradient descent) |
| **Validation** | 20% | Model selection: architecture, hyperparameters, training decisions |
| **Test** | 20% | Final, unbiased estimate of generalisation performance |

The test set must be **locked away** and touched only once, after all modelling decisions are final. The validation set is used for selecting: model class, model hyperparameters (e.g., number of layers, tree depth), and training hyperparameters (e.g., learning rate, stopping criterion).

When data is limited, use **cross-validation** for the training and model selection phases.

---

### Data Leakage

> **Data leakage is the #1 mistake in medical AI.**

Leakage occurs when **train and test sets are not independent** — for example, when multiple images from the same patient appear in both sets. This causes artificially inflated performance metrics that do not reflect true generalisation.

**Always split by patient (or study), not by image.** A single patient may contribute many images; if those images appear in both train and test, the model effectively "sees" the test patient during training.

---

## 3. Building Blocks of CNNs

### 3.1 CNN Architecture Overview

A Convolutional Neural Network (CNN) consists of two main parts:

```
Input Image
    │
    ▼
[Feature Extraction]
  Conv → BN → ReLU → Pool  (repeated)
    │
    ▼
[Classification Head]
  Flatten → Linear → Softmax
    │
    ▼
Class Probabilities
```

- **Convolutional layers** — local operators: detect edges, textures, structures
- **Linear (dense) layers** — global operators: combine extracted features for classification

A single **convolutional block** consists of:
1. Convolution
2. Batch normalisation
3. Activation function
4. Subsampling (pooling)

A **linear block** is the same but without subsampling, and the convolution is replaced by a fully connected operation.

---

### 3.2 2D Convolution

The convolution operation slides a small **filter (kernel)** across the input image, computing the **sum of element-wise multiplications** at each position. This produces a **feature map**.

For an input of size $H \times W$ and a kernel of size $k \times k$ with stride $s$ and padding $p$, the output size is:

$$\text{Output size} = \left\lfloor \frac{H - k + 2p}{s} \right\rfloor + 1$$

**What filters learn:**
- Early layers: edges, colour gradients
- Middle layers: textures, repeating patterns
- Deep layers: semantically meaningful structures (lesion boundaries, organ shapes)

---

### 3.3 Stride and Padding

**Stride** controls how many rows/columns the filter moves at each step:
- Stride 1 → densely sampled output (large feature map)
- Stride 2 → output is roughly half the input size (spatial downsampling)

**Padding** addresses the loss of boundary pixels:
- Without padding: feature map is smaller than the input
- With padding = 1 (for a 3×3 kernel): output height/width equals input height/width

> **Rule of thumb:** Use padding to preserve spatial dimensions when you do not want to downsample.

---

### 3.4 Number of Filters and Parameters

For a multi-channel input (e.g., RGB with 3 channels), each filter must span **all input channels**. To produce $N_\text{out}$ output feature maps from $C_\text{in}$ input channels using a $k \times k$ kernel:

$$\text{Total parameters} = N_\text{out} \times C_\text{in} \times k \times k$$

**Example:** Generating 18 feature maps from a 6-channel input with 3×3 kernels:

$$\text{Filters} = 6 \times 18 = 108, \quad \text{Parameters} = 108 \times 9 = 972$$

The feature maps from each per-channel filter are summed element-wise to produce one output feature map; the results from all $N_\text{out}$ sets of filters are concatenated along the channel axis.

---

### 3.5 Sub-Sampling / Pooling

Pooling reduces spatial dimensions, decreasing memory and computational cost and enabling deeper networks.

| Operation | Rule | Example (2×2 window) |
|---|---|---|
| **Max pooling** | Take the maximum value from each region | Preserves strongest activation; most common |
| **Average pooling** | Take the mean of each region | Smoother; sometimes used in final layers |

Pooling has **no learnable parameters** and is applied independently per channel.

---

### 3.6 Activation Functions

Without non-linear activation functions, stacking multiple linear layers is mathematically equivalent to a single linear layer — the network cannot learn complex patterns.

**Sigmoid and Tanh:**

| Property | Sigmoid | Tanh |
|---|---|---|
| Output range | (0, 1) | (−1, 1) |
| Vanishing gradient | Yes, at extremes | Yes, at extremes |
| Computational cost | High | High |
| Preference | Rarely used in hidden layers | Generally preferred over sigmoid |

**ReLU (Rectified Linear Unit):**

$$\text{ReLU}(x) = \begin{cases} 0 & x \leq 0 \\ x & x > 0 \end{cases}$$

- Fast to compute, no vanishing gradient for positive activations
- **Dying ReLU problem:** neurons can become permanently inactive (gradient = 0) if they always receive negative inputs

**Variants of ReLU:**

| Variant | Key property |
|---|---|
| **Leaky ReLU** | Small negative slope for $x < 0$, prevents dying ReLU |
| **Parametric ReLU** | Learnable negative slope |
| **ELU** | Smooth for negative inputs, reduces bias shift |
| **Swish** | $x \cdot \sigma(x)$; smooth, performs well in deep networks |
| **Maxout** | Learns the activation function itself |

> **Practical default:** Use ReLU in convolutional layers; consider Swish or Leaky ReLU if you observe dying neurons.

---

### 3.7 Batch Normalisation

If the input distribution of a layer shifts constantly during training (**internal covariate shift**), learning becomes unstable. Batch normalisation (BN) addresses this by normalising the pre-activation of each layer to approximately zero mean and unit variance across the current mini-batch.

For convolutional layers, the mean and variance are computed **across all spatial positions and all images in the batch** for each feature map channel. Two learnable parameters ($\gamma$, $\beta$) allow the network to rescale and shift the normalised output.

**Benefits:**
- Stabilises and accelerates training
- Acts as mild regularisation
- Allows higher learning rates

> In ConvNets, Batch Norm is typically applied **after** the convolution and **before** the activation function. It largely replaces the need for dropout in convolutional layers.

---

### 3.8 Fully Connected Layers

After the convolutional feature extractor, the spatial feature maps are **flattened** into a 1D vector and passed through one or more fully connected (dense/linear) layers.

- Convolution = **local** operator (sees a small neighbourhood)
- Dense layer = **global** operator (sees all extracted features simultaneously)

Adding dense layers ensures the network is **end-to-end trainable** via backpropagation. The final dense layer typically has as many outputs as there are classes, followed by a Softmax.

> **Practical note:** The number of input features to the first linear layer depends on the spatial dimensions after all pooling operations. Compute carefully or use global average pooling (GAP) to make it architecture-agnostic.

---

### 3.9 Dropout

Dropout randomly sets a fraction of activations to zero during each training forward pass, with each neuron kept with probability $p$.

- Prevents co-adaptation of neurons → acts as regularisation
- **Less helpful in convolutional layers** (Batch Norm already regularises effectively there)
- Applied in **dense layers** during training only
- If applied during inference with multiple forward passes, the variance of outputs approximates **predictive uncertainty** (Monte Carlo Dropout)

---

### 3.10 Training a ConvNet

Training follows a simple loop:

1. **Sample** a mini-batch of data
2. **Forward pass** through the network → compute predictions and loss
3. **Backpropagation** → compute gradients of the loss with respect to all parameters
4. **Parameter update** → apply an optimiser (e.g., SGD, Adam) using the gradients

This loop repeats for many epochs until convergence.

---

### 3.11 Softmax

The **Softmax** function converts the raw output logits $z_i$ of the final layer into a valid probability distribution:

$$\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Properties:
- All outputs lie in $(0, 1)$
- Outputs sum to 1
- Amplifies differences between logits due to exponentiation

---

## 4. Probability Calibration

A model's predicted probability should reflect **true empirical likelihood**. A model that assigns 0.7 probability to "malignant" should be correct about 70% of the time for that bin of predictions — if it is only correct 20% of the time, the model is **overconfident** and miscalibrated.

**Why Softmax probabilities can be unreliable:**

Exponentiation amplifies differences non-linearly — e.g., $e^2 = 7.38$ vs. $e^4 = 54.59$ — so a small difference in logits can produce an apparently large confidence gap that does not reflect true uncertainty.

**Temperature scaling** is a simple post-hoc calibration method. The Softmax is modified to:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

| Temperature | Effect |
|---|---|
| $T = 1$ | Standard Softmax — unchanged |
| $T > 1$ | **Softer** distribution — probabilities closer to uniform, more uncertainty |
| $T \to 0$ | **Harder** distribution — approaches one-hot |

> **Example:** Hard probabilities = [0.01, 0.01, 0.98] become soft probabilities = [0.2, 0.2, 0.6] with a higher temperature.

*Reference: Hinton, Vinyals & Dean (2015). Distilling the knowledge in a neural network.*

---

## 5. Modern Architectures and Transfer Learning

### Modern CNN Families

Over the years, increasingly deep and sophisticated CNN families have been developed:

| Family | Key innovation |
|---|---|
| **VGG** | Very deep networks with small (3×3) kernels throughout |
| **ResNet** | Residual (skip) connections enable training of very deep networks (50–152+ layers) |
| **Inception** | Multi-scale convolutions in parallel within each block |
| **EfficientNet** | Compound scaling of depth, width, and resolution simultaneously |
| **Vision Transformer (ViT)** | Self-attention replaces convolution; strong with large data |

---

### Transfer Learning

Training a deep network from scratch requires large labelled datasets. In medicine, labelled data is scarce, expensive, and often privacy-sensitive. **Transfer learning** addresses this:

1. Take a CNN pre-trained on a large dataset (e.g., ImageNet)
2. Replace the classification head with a new head for the target medical task
3. Either **freeze** the backbone and train only the head (feature extraction), or **fine-tune** all weights

**Why it works in the medical domain:**
- Early convolutional layers learn generic features (edges, textures) that transfer across domains
- Pre-trained weights are a far better initialisation than random
- Results in better features, faster training, and better generalisation from small datasets

**Foundation models** (large models pre-trained on massive diverse datasets) are increasingly used as general feature extractors — avoiding the need to retrain the full network.

> **Caution:** Model choice must be considered carefully in the medical domain. Different architectures vary in suitability, interpretability, and clinical relevance.

---

## 6. Evaluation Metrics

### 6.1 Confusion Matrix

For binary classification (Positive = Class A = Malignant, Negative = Class B = Benign):

|  | **Predicted Positive** | **Predicted Negative** |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

- **TP:** Malignant correctly predicted as malignant
- **FN:** Malignant incorrectly predicted as benign (missed disease — dangerous)
- **FP:** Benign incorrectly predicted as malignant (unnecessary alarm)
- **TN:** Benign correctly predicted as benign

---

### 6.2 Standard Metrics

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Sensitivity (Recall)} = \frac{TP}{TP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Specificity} = \frac{TN}{TN + FP}$$

$$\text{F1-score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

| Metric | Answers the question |
|---|---|
| Accuracy | What fraction of all predictions is correct? |
| Sensitivity / Recall | Of all actual positives, how many did we catch? |
| Precision | Of all predicted positives, how many are truly positive? |
| Specificity | Of all actual negatives, how many did we correctly identify? |
| F1-score | What is the balance between precision and recall? |

> **Limitation of accuracy:** In an imbalanced dataset (e.g., 90% negative class), a model that always predicts negative achieves 90% accuracy — but is clinically useless.

---

### 6.3 Random Baselines

For a model that predicts positive with probability $p_{PP}$ independently of the input (where $p_{CP}$ is the true prevalence):

| Metric | Random model value |
|---|---|
| Accuracy | $p_{CP} \cdot p_{PP} + (1 - p_{CP})(1 - p_{PP})$ |
| Sensitivity | $p_{PP}$ |
| Specificity | $1 - p_{PP}$ |
| Precision | $p_{CP}$ |

Critically: a model can achieve **sensitivity = 1** by always predicting positive, and **specificity = 1** by always predicting negative — but these cannot both be 1 for the *same* model. Sensitivity and specificity must be considered **together**.

---

### 6.4 ROC and AUC

Neural networks output a continuous probability score; the **decision threshold** converts this into a binary prediction. Changing the threshold changes the operating point (TP rate vs. FP rate trade-off). The **ROC curve** visualises this trade-off across all possible thresholds:

- **X-axis:** False Positive Rate (FPR) = $1 - \text{Specificity}$
- **Y-axis:** True Positive Rate (TPR) = Sensitivity

The **Area Under the ROC Curve (AUC / AUROC)** summarises performance across all thresholds:

| AUC | Interpretation |
|---|---|
| 1.0 | Perfect separation |
| 0.9 | Excellent |
| 0.8 | Good |
| 0.5 | Random guessing (diagonal line) |
| < 0.5 | Worse than random |

> Different clinical tests and even different clinicians operate at different points on the ROC curve. AUC provides a **threshold-independent** comparison.

---

### 6.5 F1 Score

$$F_1 = \frac{2}{\frac{1}{\text{Precision}} + \frac{1}{\text{Recall}}} = \frac{2\,TP}{2\,TP + FP + FN}$$

F1 is the harmonic mean of precision and recall. It is useful when **false positives and false negatives both matter**. However:

> **F1 completely ignores True Negatives (TN).** For a random model that always predicts positive ($p_{PP} = 1$), $F_1 = \frac{2\, p_{CP}}{1 + p_{CP}}$ — e.g., 18% F1 for a dataset with 10% positive prevalence.

---

### 6.6 Matthews Correlation Coefficient

The **MCC (Matthews Correlation Coefficient)** measures the overall quality of binary classification:

$$\phi = \frac{TP \times TN - FP \times FN}{\sqrt{CP \times CN \times PP \times PN}}$$

Key properties:
- $\phi = +1$: Perfect predictions
- $\phi = -1$: Perfectly inverted predictions
- $\phi = 0$: Random model (regardless of class imbalance)

> **MCC is more reliable under class imbalance** than accuracy or F1, as it accounts for all four cells of the confusion matrix.

---

### 6.7 Multi-class Extensions

For $K$ classes with confusion matrix $C$ (where $C_{ij}$ = number of class-$i$ samples predicted as class $j$):

$$\text{Accuracy} = \frac{\sum_i C_{ii}}{\sum_{i,j} C_{ij}}$$

Per-class metrics (treating class $i$ as positive vs. all others):

$$\text{Recall for class } i = \frac{C_{ii}}{\sum_j C_{ij}}, \quad \text{Precision for class } i = \frac{C_{ii}}{\sum_j C_{ji}}$$

**Averaging strategies:**

| Strategy | Definition | When to use |
|---|---|---|
| **Macro-average** | Unweighted mean over all classes | When all classes are equally important |
| **Micro-average** | Pool all TP/FP/FN, then compute | Dominated by frequent classes |
| **Weighted-average** | Macro but weighted by class frequency | — |

Note: macro-averaged recall equals **balanced accuracy** — useful for imbalanced datasets.

---

### 6.8 Pitfalls

**Metrics pitfalls:**
- Always consider the **trade-off** between precision and recall, or sensitivity and specificity. Reporting only one side is misleading.
- Be precise about what you are **averaging over**: classes, thresholds, augmentations, ensemble members, patient examinations.
- Do not blindly apply **binary metrics** to multi-class or multi-label problems.
- Mean Average Precision (mAP) has a specific definition — do not confuse it with a simple average.

**Generalisation pitfalls:**
- Split data by **examination, patient, or institution** — not by image.
- Evaluate on **out-of-distribution data** to measure true generalisation.
- Account for differences in devices, protocols, and pre-processing between training and deployment.
- Relate metrics to **clinical and economic utility**, not just benchmark numbers.
- Verify the model is not exploiting **spurious correlations** (e.g., gender, scanner artefacts, image acquisition metadata).

---

## 7. Tips for Robust Classification Models

### 7.1 Cross-Validation

Cross-validation estimates how well a model generalises across different subsets of the data, for a given configuration.

**K-Fold Cross-Validation:**
- Split data into $K$ folds
- In each of $K$ iterations, use one fold as the validation set and train on the remaining $K-1$ folds
- Average performance across all $K$ folds

**Stratified Cross-Validation:** Ensures each fold preserves the original class proportions — important for imbalanced datasets.

> When working with patient data, ensure the **same patient does not appear in both training and validation** within any fold. Split by patient, not by image.

---

### 7.2 Hyperparameter Tuning

Hyperparameter tuning identifies the parameter settings that give the best performance on the validation set for a given dataset.

**Common methods:**

| Method | Description |
|---|---|
| **Grid Search** | Exhaustively evaluates all combinations in a defined grid |
| **Random Search** | Randomly samples combinations; efficient for large spaces |
| **Bayesian Optimisation** | Uses a surrogate model to guide search toward promising regions |

> **Practical approach:** Start with random search to explore the space efficiently, then apply Bayesian Optimisation to refine promising regions.

---

### 7.3 Ablation Studies

An ablation study systematically removes or modifies individual components of a model (modules, loss functions, data augmentation strategies) and measures the effect on performance. This reveals the **contribution of each component** to the final result and justifies design decisions.

**Example:** Compare a full model vs. the same model without a specific module (e.g., without attention, without a particular loss term) to quantify that component's contribution.

---

*Notes compiled from: 05_MEDIMG_classification.pdf — HSLU Medical Image Analysis, Nipun Sandamal*
