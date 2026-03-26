# Week 04 — Self-Supervised and Semi-Supervised Learning

**Course:** Medical Image Analysis — HSLU  
**Lecturer:** Fabian Gröger  
**Topics:** Label bottleneck · Self-supervised pretext tasks · Contrastive learning · Beyond contrastive methods · SSL for medical imaging · Semi-supervised learning · Practical guidelines

---

## Table of Contents

1. [Motivation: The Label Bottleneck](#1-motivation-the-label-bottleneck)
2. [Key Terminology](#2-key-terminology)
3. [Self-Supervised Learning via Pretext Tasks](#3-self-supervised-learning-via-pretext-tasks)
   - [Context Prediction](#31-context-prediction)
   - [Jigsaw Puzzles](#32-jigsaw-puzzles)
   - [Rotation Prediction](#33-rotation-prediction)
   - [Limitations of Simple Pretext Tasks](#34-limitations-of-simple-pretext-tasks)
4. [Contrastive Methods](#4-contrastive-methods)
   - [MoCo](#41-moco-momentum-contrast)
   - [SimCLR](#42-simclr)
   - [MoCo v2](#43-moco-v2)
   - [Problems with Contrastive Learning](#44-problems-with-contrastive-learning)
5. [Beyond Contrastive Methods](#5-beyond-contrastive-methods)
   - [BYOL](#51-byol-bootstrap-your-own-latent)
   - [DINO](#52-dino)
   - [MAE](#53-mae-masked-autoencoders)
6. [SSL for Medical Imaging](#6-ssl-for-medical-imaging)
   - [Challenges Specific to Medical Imaging](#61-challenges-specific-to-medical-imaging)
   - [Domain-Specific Pretext Tasks](#62-domain-specific-pretext-tasks)
   - [General Framework](#63-general-framework-for-adapting-ssl)
7. [Semi-Supervised Learning](#7-semi-supervised-learning)
   - [Disclaimer and Assumptions](#71-disclaimer-and-assumptions)
   - [Methods](#72-methods)
8. [Practical Guidelines](#8-practical-guidelines)

---

## 1. Motivation: The Label Bottleneck

Supervised deep learning requires large amounts of **labelled** data. In medical imaging, annotation is a severe bottleneck:

| Problem | Details |
|---|---|
| **Expensive** | Requires domain experts (radiologists, pathologists) |
| **Time-consuming** | A single 3D CT segmentation can take hours |
| **Subjective** | High inter-observer variability for many tasks |
| **Scarce** | Rare diseases have very few annotated examples |

Meanwhile, vast amounts of **unlabelled** medical images exist — hospitals store millions of imaging studies, and public repositories are growing rapidly.

> **Key question:** How can we leverage unlabelled data?

The **two-stage approach** of self-supervised learning addresses this:

1. **Stage 1 — Pretext task:** Train an encoder on unlabelled data using a self-generated supervisory signal. The model learns to solve a proxy task that forces it to learn useful image representations.
2. **Stage 2 — Downstream task:** Fine-tune the encoder on the small labelled dataset for the actual clinical problem.

---

## 2. Key Terminology

| Term | Definition |
|---|---|
| **Pretext task** | A self-supervised task (e.g., context prediction) designed to force the network to learn useful representations without human-annotated labels |
| **Downstream task** | The actual clinical problem to solve (e.g., tumour segmentation, disease classification) using the pretrained encoder |
| **Probing** | Training simple classifiers (e.g., k-NN or linear predictors) on a frozen pretrained encoder to test whether it has learned specific features — not about maximising performance |
| **Pseudo-label** | A highly confident prediction made by a model on unlabelled data, used as a target "label" for further training |

**Why self-supervision?**

✅ Can procedurally generate potentially infinite amounts of annotation  
✅ Borrows tricks from supervised learning without requiring labels  
✅ Focuses on only the relevant information (e.g., not raw pixels)  
✅ Answering generated questions requires fundamental understanding of data  

⚠️ Designing good questions also requires domain understanding  
⚠️ No one-fits-all solution for creating pretext tasks  

---

## 3. Self-Supervised Learning via Pretext Tasks

### 3.1 Context Prediction

*Doersch, Gupta & Efros (2015). Unsupervised Visual Representation Learning by Context Prediction. ICCV.*

**Idea:** Divide an image into a 3×3 grid of patches. A centre patch (patch 3) is shown alongside one other randomly selected patch. The model must predict the relative position (one of 8 possible locations) of the second patch relative to the centre.

- The label is generated automatically from the patch's position — no human annotation needed.
- Solving this task requires semantic understanding of the image content.
- Uses a **Siamese Network**: two identical sub-networks with shared weights process each patch independently, then their representations are concatenated for classification.

**Trivial shortcut problem:** The network exploited low-level cues such as boundary continuation and chromatic aberration rather than learning semantics. Authors mitigated this with explicit gaps and colour jitter between patches.

---

### 3.2 Jigsaw Puzzles

*Noroozi & Favaro (2016). Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles. ECCV.*

**Idea:** Divide the image into a 3×3 grid of tiles, randomly scramble them, and train the network to predict the correct permutation (i.e., solve the puzzle).

- Labels are the permutation indices — again, fully automatic.
- Solving jigsaw puzzles demands understanding of object structure and part relationships.

---

### 3.3 Rotation Prediction

*Gidaris, Singh & Komodakis (2018). Unsupervised Representation Learning by Predicting Image Rotations. ICLR.*

**Idea:** Rotate each image by one of four angles (0°, 90°, 180°, 270°) and train the network to predict which rotation was applied.

- This is a 4-class classification problem.
- Predicting orientation correctly requires understanding the semantic content of the image — a random texture pattern looks the same upside-down, but a bird or face does not.

**Results (linear classifier on ImageNet representations):**

| Method | Conv1 | Conv3 | Conv5 |
|---|---|---|---|
| ImageNet labels (supervised) | 19.3 | 44.2 | 50.5 |
| Jigsaw Puzzles | 18.2 | 34.0 | 27.1 |
| **(Ours) RotNet** | **18.8** | **38.7** | **36.5** |

RotNet outperforms all prior pretext tasks at most layers, though it still falls well short of fully supervised pretraining.

---

### 3.4 Limitations of Simple Pretext Tasks

These early approaches shared fundamental weaknesses:

- **Shortcut learning / collapse:** The model solved the proxy task without learning semantically meaningful representations — e.g., using chromatic aberration for context prediction rather than object understanding.
- **Weak semantic signal:** Pretext tasks had only a weak connection to the features needed for downstream tasks.
- **No scaling:** Performance did not improve with more data or larger models — the narrow puzzle could be solved without needing richer representations.
- The phenomenon of a model finding trivial solutions is known as **collapse** in the SSL literature.

> **Next:** Contrastive methods address these issues by creating scalable objectives that explicitly prevent collapse.

---

## 4. Contrastive Methods

The core principle of contrastive learning: **pull together representations of augmented views of the same image (positives) and push apart representations of different images (negatives).**

This operates across the entire dataset, not just a narrow spatial puzzle, forcing global semantic organisation of the representation space.

---

### 4.1 MoCo: Momentum Contrast

*He, Fan, Wu, Xie & Girshick (2020). Momentum Contrast for Unsupervised Visual Representation Learning. CVPR.*

**Problem with naive contrastive learning:** You need many negatives to avoid the model collapsing to trivial solutions. Having a very large batch is computationally prohibitive.

**MoCo's solution — a dynamic queue:**
- A **queue** of $K-1$ encoded negative keys $k_0, k_1, k_2, \ldots$ is maintained across mini-batches.
- The queue is updated every step: the current mini-batch's keys are enqueued and the oldest are dequeued.
- This decouples the number of negatives from the batch size.

**Momentum encoder:** Keys are encoded by a slowly updating **momentum encoder** rather than the main encoder:

$$\theta_k \leftarrow m\theta_k + (1-m)\theta_q$$

where $m \in [0, 1)$ is the momentum coefficient (typically 0.999). This ensures the key representations are consistent across the queue despite being encoded at different steps.

**Contrastive loss (InfoNCE):**

$$\mathcal{L}_q = -\log \frac{\exp(q \cdot k_{\text{pos}} / \tau)}{\sum_{i=0}^{K} \exp(q \cdot k_i / \tau)}$$

where $q$ is the query, $k_{\text{pos}}$ is the positive key (augmented view of same image), and $\tau$ is a temperature hyperparameter.

---

### 4.2 SimCLR

*Chen, Kornblith, Norouzi & Hinton (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.*

**Framework:**
1. Two independent augmentations $t \sim \mathcal{T}$ and $t' \sim \mathcal{T}$ are applied to the same input image $x_1$, producing two views $\tilde{x}_1$ and $\tilde{x}_1'$.
2. A shared encoder $f(\cdot)$ (e.g., ResNet-50) maps each view to a representation $e$.
3. A **projection head** $g(\cdot)$ (MLP) maps each representation to $z$, on which the loss is computed.
4. After training, the projection head is discarded and only $f(\cdot)$ is kept for downstream tasks.

**NT-Xent loss** (Normalised Temperature-scaled Cross Entropy):

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(\boldsymbol{z}_i, \boldsymbol{z}_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\boldsymbol{z}_i, \boldsymbol{z}_k)/\tau)}$$

where $\text{sim}(\boldsymbol{u}, \boldsymbol{v}) = \boldsymbol{u}^\top \boldsymbol{v} / \|\boldsymbol{u}\| \|\boldsymbol{v}\|$ is cosine similarity, and $N$ is the batch size (giving $2N$ views total).

Key finding: SimCLR **requires very large batch sizes** (4096+) to produce enough negatives for strong performance. With a 4× wider ResNet-50, it begins to approach supervised performance.

---

### 4.3 MoCo v2

*Chen, Fan, Girshick & He (2020). Improved Baselines with Momentum Contrastive Learning. ArXiv.*

A short paper that incorporated SimCLR insights into MoCo:
1. Added an **MLP projection head** at the encoder output (as in SimCLR; not used for downstream feature extraction).
2. Added **SimCLR-style data augmentation** (colour jitter, Gaussian blur).

Result: MoCo v2 matches SimCLR performance with only batch size 256 instead of 4096+ — much more hardware-efficient.

---

### 4.4 Problems with Contrastive Learning

- **False negatives:** When sampling random negatives from a large dataset, some "negatives" will actually belong to the same semantic class as the query — e.g., two different photos of a golden retriever treated as negatives. This is especially problematic with **class imbalance**, which is common in medical imaging (pathology is rare).
- **Large batch sizes required:** Enough negatives are needed to prevent the model from ignoring them; this demands high GPU memory.

---

## 5. Beyond Contrastive Methods

### 5.1 BYOL: Bootstrap Your Own Latent

*Grill et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. NeurIPS.*

**Key innovation:** BYOL achieves strong representations **without any negative samples** — avoiding the false negative problem entirely.

**Architecture:**
- **Online network:** encoder $f_\theta$ + projector $g_\theta$ + predictor $q_\theta$. Trained by gradient descent.
- **Target network:** encoder $f_\xi$ + projector $g_\xi$. Updated only via EMA (no gradients):

$$\xi \leftarrow m\xi + (1-m)\theta$$

**Loss:** The online network's prediction $p$ is trained to match the target network's projection $z'$:

$$\mathcal{L}_\text{BYOL} = \|\bar{p} - \bar{z}'\|_2^2$$

where $\bar{p}$ and $\bar{z}'$ are $\ell_2$-normalised vectors. The loss is computed symmetrically (swapping online and target views).

**Key properties:**
- Robust to smaller batch sizes — the loss does not degrade nearly as much as SimCLR when batch size is reduced.
- Robust to removing data augmentations.
- Outperforms SimCLR and MoCo v2 at comparable model sizes.
- How exactly it avoids collapse (without negatives) is not fully theoretically understood.

---

### 5.2 DINO

*Caron et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. CVPR.*

**Key innovation:** Applies self-distillation to Vision Transformers (ViT), producing representations with remarkable semantic properties.

**Architecture (mirrors BYOL):**
- **Student network:** processes local and global crops; updated by gradient descent.
- **Teacher network:** processes only global crops; updated by EMA.
- Loss is cross-entropy between student and teacher softmax outputs:

$$\mathcal{L}_\text{DINO} = -p_t \log p_s$$

**Multi-crop strategy:** The student sees multiple small (local) crops as well as large (global) crops, while the teacher only sees the global crops. This enforces **local-to-global correspondence** — the model learns that a local patch must be consistent with the global image.

**Emergent properties:** Without any supervised signal, DINO's attention maps spontaneously segment objects — different attention heads focus on different object parts. This property does not emerge in supervised or CNN-based SSL methods.

**Results:** DINO with ViT-B/8 achieves 80.1% top-1 accuracy on ImageNet with a k-NN classifier directly on the frozen representation — the best k-NN performance reported at the time.

---

### 5.3 MAE: Masked Autoencoders

*He et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR.*

**Idea:** Inspired by BERT in NLP. Randomly mask a very high proportion (75%) of image patches and train an autoencoder to reconstruct the missing patches from the visible ones.

**Architecture:**
- Image is divided into non-overlapping 16×16 patches.
- 75% of patches are masked (removed from input).
- **Encoder** (ViT) processes only the ~25% visible patches — this is computationally very efficient.
- A **lightweight decoder** reconstructs the masked patches at the pixel level.
- **Loss:** MSE on the masked patches only.

**Key properties:**
- Scales very well — benefits from larger models and more data.
- Computationally efficient at training time because the encoder processes only 25% of tokens.
- The embedding space can be less semantically structured than contrastive or distillation methods (smoothness issue), but fine-tuning performance is state-of-the-art.

---

## 6. SSL for Medical Imaging

### 6.1 Challenges Specific to Medical Imaging

Medical images differ fundamentally from the natural images used to develop most SSL methods:

| Challenge | Implication |
|---|---|
| **Grayscale / limited channels** (CT, MRI, X-ray) | Colorisation pretext task doesn't apply |
| **3D volumetric data** (CT, MRI) | Need specialised handling of slices, volumes, or 3D patches |
| **Strong spatial anatomy priors** | Consistent anatomy across patients can be exploited for pretext tasks |
| **Multi-modal data** (CT + MRI, image + report) | Enables cross-modal SSL objectives |
| **Small dataset sizes** | Less unlabelled data than typical natural image corpora |
| **Class imbalance** | Pathology is rare; contrastive methods suffer from false negatives |

---

### 6.2 Domain-Specific Pretext Tasks

Because natural-image pretext tasks may not generalise, researchers design medical-domain-specific tasks:

| Task | Description |
|---|---|
| **Anatomical position prediction** | Predict body region or slice position within a 3D volume — exploits consistent anatomy across patients |
| **Organ/structure recognition** | Predict which organ is visible in a given patch |
| **3D context tasks** | Predict relative slice position (above/below); Rubik's cube recovery (shuffle 3D sub-volumes and restore) |
| **Cross-modal prediction** | Predict one MRI sequence from another (e.g., T1 → T2) |
| **Image-text pairing** | Match radiology images with report text (CLIP-style contrastive learning) |

**Medical-specific positive pair strategies** (for contrastive methods):
- **Multi-instance:** Multiple images of the same patient serve as positive pairs.
- **Multi-view:** Different imaging views of the same anatomy serve as positives.

> Designing these tasks requires understanding the domain structure. The most powerful pretext tasks are those that act as true data augmentations — predicting any part of the input from any other part (LeCun's framing).

---

### 6.3 General Framework for Adapting SSL

*Based on Azizi et al. (2023). Robust and Efficient Medical Imaging with Self-Supervision. ArXiv.*

The recommended recipe is a **two-stage pretraining pipeline**:

1. **Non-medical SSL pretraining:** Initialise from a model pretrained on natural images (ImageNet) using standard SSL (SimCLR, BYOL, DINO, etc.). This provides generic low- and mid-level visual features.
2. **Medical-specific SSL pretraining (intermediate):** Continue pretraining on unlabelled in-domain medical images using domain-specific pretext tasks and augmentations.
3. **Supervised fine-tuning:** Fine-tune the encoder on the small labelled dataset for the target clinical task.

**Key findings from Azizi et al.:**
- This two-stage SSL approach consistently outperforms purely supervised baselines on in-distribution evaluation.
- SSL significantly reduces the amount of labelled out-of-distribution data needed to match baseline performance — reducing annotation costs by up to 94% in some tasks.
- Evaluated across 6 clinical tasks: dermatology, diabetic macular edema, chest X-ray, pathology (metastases detection and survival prediction), mammography.

**Practical reminders:**
- SSL does not work out-of-the-box in medical domains — adaptation is required.
- Design domain-specific pretext tasks; use natural positive/negative pairs when available.
- Keep class imbalance in mind when choosing an SSL method (contrastive methods are especially susceptible).

---

## 7. Semi-Supervised Learning

**Semi-supervised learning (Semi-SL)** operates on a mixture of labelled and unlabelled data simultaneously, combining supervised and unsupervised signals in a single training procedure.

It sits between fully supervised learning (all data labelled) and fully unsupervised learning (no labels at all).

---

### 7.1 Disclaimer and Assumptions

> **Important caveat:** Semi-supervised learning is not a guaranteed performance booster. Adding unlabelled data can actually *degrade* performance — this has been observed in practice and is likely under-reported due to publication bias. The better the supervised baseline already is, the less likely Semi-SL will help.

Semi-SL rests on three core assumptions about the data:

| Assumption | Statement |
|---|---|
| **Smoothness** | If two samples are close in input space, their labels should also be similar. *"If they look similar, they are similar."* |
| **Low-density** | The decision boundary should not pass through high-density regions of the input space. Class boundaries should lie in areas with few samples. |
| **Manifold** | Data points on the same low-dimensional manifold should have the same label. *"Data points with the same features should have the same label."* |

If these assumptions are violated, Semi-SL can actively harm performance. Always verify them for your dataset before applying these methods.

---

### 7.2 Methods

#### Self-Training (Pseudo Labelling)

1. Train a model on the available labelled data to get a partially trained model.
2. Run this model on the unlabelled data. Take the **most confident predictions** as pseudo-labels.
3. Retrain or fine-tune the model on both the original labelled data and the pseudo-labelled data.
4. Repeat iteratively.

**Risk:** Early errors in pseudo-labelling compound — if the initial model is poor, bad pseudo-labels corrupt further training.

---

#### Co-Training

1. Train two or more models independently on the labelled data, potentially on different views/augmentations of the input.
2. Each model infers pseudo-labels for the unlabelled data.
3. Each model's **most confident predictions** are added to the other models' training sets as labels.
4. All models are retrained with the expanded label sets.

**Advantage:** The disagreement between independent models provides a self-checking mechanism — one model's confident but wrong predictions may be caught by the other.

---

#### Cluster-then-Label

1. Train a feature extractor on unlabelled data (e.g., via SSL pretraining).
2. Extract features for all samples (labelled and unlabelled) and cluster them.
3. Assign each unlabelled sample the label of its nearest labelled neighbour in feature space.

This naturally connects with SSL: a good SSL-pretrained encoder can directly enable effective cluster-then-label pipelines without any label-specific training.

---

## 8. Practical Guidelines

The following decision process summarises when and how to apply SSL and Semi-SL in medical imaging projects:

**1. Unlabelled in-domain data is available → consider SSL pretraining**
- Follow the two-stage principle: non-medical pretraining first, then medical-specific intermediate pretraining.
- Whenever possible, use real-world augmentations (e.g., patient-level positives, cross-modal pairs) rather than purely synthetic ones.

**2. Evaluate the pretrained model with simple probes**
- Use k-NN or linear classifiers on frozen features to assess representation quality.
- Evaluate across multiple distribution shifts: behavioural, technological (scanner/protocol), and population shifts.

**3. Model performance is still insufficient and labelled data is very limited → consider semi-supervised learning**
- Ensure unlabelled data comes from the **same distribution** as labelled data.
- Verify that your data satisfies the smoothness, low-density, and manifold assumptions.
- Critically evaluate for amplified biases or degraded performance on underrepresented classes.

> **Key reminder:** Semi-supervised learning should be treated as one option among several — not a guaranteed win. Always compare against a strong supervised baseline.

---

## Key Papers

| Paper | Contribution |
|---|---|
| Doersch et al. (2015). *Unsupervised Visual Representation Learning by Context Prediction.* ICCV. | First influential pretext task: patch context prediction |
| Noroozi & Favaro (2016). *Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles.* ECCV. | Jigsaw puzzle pretext task |
| Gidaris et al. (2018). *Unsupervised Representation Learning by Predicting Image Rotations.* ICLR. | Rotation prediction pretext task |
| He et al. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning.* CVPR. | MoCo: queue + momentum encoder for scalable contrastive learning |
| Chen et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* ICML. | SimCLR: projection head + strong augmentations |
| Grill et al. (2020). *Bootstrap Your Own Latent.* NeurIPS. | BYOL: negative-free SSL via online/target network EMA |
| Caron et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers.* CVPR. | DINO: self-distillation with ViT; emergent segmentation |
| He et al. (2022). *Masked Autoencoders Are Scalable Vision Learners.* CVPR. | MAE: masked patch reconstruction at scale |
| Azizi et al. (2023). *Robust and Efficient Medical Imaging with Self-Supervision.* ArXiv. | Two-stage SSL recipe for medical imaging; REMEDIS |
| van Engelen & Hoos (2020). *A Survey on Semi-Supervised Learning.* Mach Learn. | Comprehensive overview of Semi-SL methods and pitfalls |

---

*Notes compiled from: 04_MEDIMG_self_semi_supervised.pdf — HSLU Medical Image Analysis, Fabian Gröger*
