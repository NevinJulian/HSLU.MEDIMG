# Week 08 — Longitudinal and Sequential Imaging Data

**Course:** Medical Image Analysis — HSLU  
**Lecturer:** Dr. Ludovic Amruthalingam  
**Topics:** Time in medical imaging · Task taxonomy · Fusion architectures (late/mid/early) · Video transformers · Longitudinal modelling · Augmentation, alignment, and sampling · Calibration · Evaluation

---

## Table of Contents

1. [Clinical Importance of Time](#1-clinical-importance-of-time)
2. [Types of Sequential Medical Imaging](#2-types-of-sequential-medical-imaging)
3. [Task Taxonomy](#3-task-taxonomy)
   - [Running Example — Cardiac Cine Imaging](#31-running-example--cardiac-cine-imaging)
4. [Architecture Map — Where to Fuse Time](#4-architecture-map--where-to-fuse-time)
   - [Late Fusion — Strong Baseline](#41-late-fusion--strong-baseline)
   - [Mid Fusion — Order-sensitive Patterns](#42-mid-fusion--order-sensitive-patterns)
   - [Early Fusion — Spatiotemporal Convolutions](#43-early-fusion--spatiotemporal-convolutions)
   - [Video Transformers](#44-video-transformers)
5. [Case Study 1: EchoNet-Dynamic](#5-case-study-1-echonet-dynamic)
6. [Longitudinal Modelling](#6-longitudinal-modelling)
7. [Case Study 2: Temporal Learning for Pediatric Glioma](#7-case-study-2-temporal-learning-for-pediatric-glioma)
8. [Augmentation for Sequential Data](#8-augmentation-for-sequential-data)
9. [Alignment and Image Registration](#9-alignment-and-image-registration)
10. [Sampling Strategy](#10-sampling-strategy)
11. [Label Sparsity in Medical Video](#11-label-sparsity-in-medical-video)
12. [Probability Calibration (Recap)](#12-probability-calibration-recap)
13. [Calibration in Temporal Settings](#13-calibration-in-temporal-settings)
14. [Evaluation — Dataset Splitting](#14-evaluation--dataset-splitting)
15. [Evaluation — Metrics](#15-evaluation--metrics)
16. [Temporal Data Formats](#16-temporal-data-formats)
17. [Temporal Data Pitfalls](#17-temporal-data-pitfalls)
18. [Key Takeaways](#18-key-takeaways)
19. [Key Papers](#19-key-papers)

---

## 1. Clinical Importance of Time

In medical imaging, time is not just context — it is often the primary signal:

- **Motion encodes function** — ventricular contraction and relaxation in echocardiography tell you whether the heart is pumping effectively.
- **Dynamics encode physiology** — contrast uptake curves in perfusion CT/MRI reflect tissue vascularity and blood flow.
- **Visit-to-visit changes encode disease progression** — serial brain MRI scans document tumour growth or atrophy over months and years.

**Treating frames as independent is often wrong.** Neighbouring frames are highly correlated, motion patterns are structured, labels often describe the whole sequence (e.g., ejection fraction, diagnosis), and noise can be time-dependent (probe motion, occlusions). Any model that ignores these facts discards clinically essential information.

---

## 2. Types of Sequential Medical Imaging

| Type | Time scale | Examples | Key challenges |
|---|---|---|---|
| **Video-like (2D+t)** | Within exam | Echocardiography cine loops, endoscopy video | High redundancy, motion artefacts, variable frame rate |
| **Dynamic volumetric (3D+t / 4D)** | Within exam | Cine MRI, perfusion CT/MRI, fMRI time series | Heavy memory footprint, alignment/motion correction required |
| **Longitudinal follow-up (visit-to-visit)** | Across exams (months–years) | Therapy response MRI, neurodegeneration follow-up | Irregular intervals, missing visits, scanner/protocol drift |

---

## 3. Task Taxonomy

Sequential imaging gives rise to a hierarchy of output structures:

| Task | Level | Example |
|---|---|---|
| **Sequence → label** | Exam-level classification / regression | Echo video → EF; capsule endoscopy → "finding present?" |
| **Sequence → sequence** | Dense predictions per frame | Cine MRI → segmentation mask per frame; echo → LV contour per frame |
| **Event detection / keyframe localisation** | Temporal localisation (timepoint-level) | Detect end-diastole (ED) and end-systole (ES) frames in a cardiac cycle |
| **Change detection** | Pair- or series-level | Baseline + follow-up → change score / progression label |
| **Forecasting / progression** | Visit-sequence-level | Use visits up to time $t$ to predict status at $t + \Delta$ |

### 3.1 Running Example — Cardiac Cine Imaging

Key frames in a cardiac cycle:
- **End-diastole (ED):** ventricle maximally filled (largest volume EDV).
- **End-systole (ES):** ventricle most contracted (smallest volume ESV).

Ejection fraction (EF) is a **sequence-derived measurement** requiring at least ED and ES. It is computed via the Simpson method:

$$EF = \frac{EDV - ESV}{EDV} \times 100\%$$

Typical modelling tasks are: video → EF (regression), video → ED/ES frame detection, video → LV mask per frame → EF from volume.

---

## 4. Architecture Map — Where to Fuse Time

The central design question for sequential models is *when* temporal information is combined across the network:

| Strategy | How it works | When to use |
|---|---|---|
| **Late fusion** | Frame encoder → pooling/attention across time → output | Sequence-level labels + limited data |
| **Mid fusion** | Frame encoder → RNN/GRU/LSTM → output | Order matters (phases, events) |
| **Early fusion** | Spatiotemporal convolutions or space-time attention | Long-range context or motion-sensitive features |

**Rule of thumb:** always implement one strong late-fusion baseline first, then add one temporal model. Use attention-based models (transformers) when long-range context matters.

### 4.1 Late Fusion — Strong Baseline

**Pipeline:** (a) Extract features from each frame independently with a shared encoder; (b) aggregate over time with mean/max/attention pooling (pooling discards temporal order; temporal attention pooling lets the model select informative frames); (c) predict a sequence label.

**Strengths:** handles variable-length sequences naturally; stable in low-data settings; easy to debug by inspecting per-frame embeddings or predictions.

**Weakness:** discards model dynamics — misses patterns only visible through temporal change.

### 4.2 Mid Fusion — Order-sensitive Patterns

**Pipeline:** (a) Extract features per frame; (b) run a temporal model over the sequence of embeddings — GRU, LSTM, TCN, transformer encoder, BiLSTM (uses past and future context for offline tasks), or ConvLSTM (per-frame spatial output with temporal consistency); (c) predict from the last or pooled hidden state.

Learns order-sensitive patterns (phases, event context) that mean pooling misses. Suited for event detection (ED/ES) or offline sequence labelling. Handle variable-length sequences with padding and masking; use windowing for long sequences. Higher risk of overfitting on small datasets than late fusion.

### 4.3 Early Fusion — Spatiotemporal Convolutions

Time is treated as an extra axis and the network convolves jointly over space and time. Filters respond to both **appearance** and **temporal changes**, and can detect patterns that only exist *across* frames.

**Strengths:** learns motion-sensitive features; strong performance with enough data.

**Weaknesses:** sensitive to clip length and sampling strategy; clip length defines the temporal receptive field (local motions vs. long-range); higher compute, memory, and data requirements.

**Mitigation:** factorised convolutions (2D spatial conv followed by 1D temporal conv) stabilise training and reduce cost — this approximates the full 3D convolution at a fraction of the parameter count.

### 4.4 Video Transformers

Attention-based models for video model local or long-range dependencies but naive attention scales quadratically with the number of tokens — structure is required.

**TimeSformer (global temporal reasoning):** factorises attention into a temporal pass (across frames, per spatial location) and a spatial pass (within each frame). Can model periodic signals and long-range dependencies.

**Video Swin Transformer (local temporal reasoning):** computes attention in local spatiotemporal windows and uses shifted windows to enable cross-window interaction. More efficient and scalable than global attention.

> Medical datasets are often small → regularisation and pretraining (e.g., from Kinetics-400) are critical for video transformers.

---

## 5. Case Study 1: EchoNet-Dynamic

*Ouyang et al. "Video-based AI for beat-to-beat assessment of cardiac function." Nature, 2020.*

**Task:** Predict left ventricular ejection fraction (EF) from standard apical four-chamber echocardiography videos; simultaneously segment the LV per frame.

**Dataset:** 10,030 annotated echocardiogram videos from Stanford Health Care (unique patients). Train/validation/test split: 7,465 / 1,277 / 1,288 patients.

**Architecture — three components:**

1. **Semantic segmentation (LV mask):** DeepLabV3 with atrous convolutions on a ResNet-50 backbone. Trained with weak supervision — expert tracings at ED and ES only, with the model generalising to all intermediate frames. LV area curve is used to identify cardiac contractions.

2. **EF regression (spatiotemporal CNN):** R2+1D decomposed spatiotemporal convolutions — a 2D spatial conv followed by a 1D temporal conv for each block. Pretrained on Kinetics-400. Input: 32-frame clips sampled every other frame. Trained to minimise squared loss between predicted and human-measured EF.

3. **Beat-to-beat test-time augmentation:** the segmentation arm identifies each ventricular contraction; a 32-frame clip centred on each beat is fed to the EF model; clip-level estimates are averaged to give a beat-to-beat evaluation across the whole video.

**Key results:**
- LV segmentation: Dice 0.92 overall (ED: 0.927, ES: 0.903)
- EF prediction on internal test set: MAE 4.1%, RMSE 5.3%, $R^2$ 0.81
- Cardiomyopathy classification (EF < 50%): AUC 0.97
- External test set (Cedars-Sinai, $n = 2{,}895$): MAE 6.0%, AUC 0.96

**Comparison with human variability:** prospective study on 55 patients showed EchoNet-Dynamic had the least variance on repeat testing (median difference 2.6%, s.d. 6.4) compared to Simpson's biplane (5.2%), monoplane (4.6%), and global longitudinal strain (8.1%). In 43% of the highest-discordance cases, blinded experts preferred the model's EF over the original human label.

**Why videos outperform stills:** previous still-image approaches achieved $R^2$ of only 0.33–0.50. Spatiotemporal convolutions capture the full cardiac cycle, enabling beat-to-beat estimation that mirrors the guideline recommendation of averaging multiple beats — something rarely done in clinical practice due to time constraints.

---

## 6. Longitudinal Modelling

Longitudinal data (visit-to-visit, months/years apart) differs fundamentally from video data:

- Timepoints are often **months or years apart**, with **irregular intervals**.
- Missing visits are informative — treat **time gaps $\Delta t$ as part of the input** (as a covariate or embedding).
- Labels are at the patient/visit level, not the frame level.

**Feasible baselines:**

1. **Paired comparison:** (baseline scan, follow-up scan) → change score. Simple and interpretable.
2. **Visit embeddings + temporal pooling:** encode each visit independently, then aggregate with mean or attention pooling over the visit sequence.
3. **Model the time gap explicitly:** include $\Delta t$ between visits as an additional input covariate or sinusoidal time embedding.

**Time-varying confounders** (e.g., scanner upgrades, protocol changes across years of follow-up) can mimic or mask disease progression. These must be identified; harmonisation strategies or metadata should be incorporated.

---

## 7. Case Study 2: Temporal Learning for Pediatric Glioma

*Tak et al. "Longitudinal Risk Prediction for Pediatric Glioma with Temporal Deep Learning." NEJM AI, 2025.*

**Clinical motivation:** Pediatric gliomas are the most common brain tumours and leading cancer-related cause of death in children. Recurrence is heterogeneous, clinically difficult to predict, and often only detected symptomatically despite frequent MRI surveillance. Almost all children undergo the same intensive surveillance regardless of individual risk — a risk-stratification tool could allow high-risk patients to receive earlier intervention and low-risk patients to reduce surveillance burden.

**Task:** Predict 1-year event-free survival (EFS) from the time of the most recent postoperative MRI scan, using the patient's full longitudinal scan history as input.

**Datasets:** 3,994 T2-FLAIR brain MRI scans from 715 patients across three institutions (DFCI/BCH pLGG, CBTN pLGG, RadART pLGG, DFCI/BCH pHGG).

### 7.1 Architecture

The longitudinal pipeline chains four components:

1. **3D ResNet18 encoder:** encodes each individual 3D MRI scan into a latent vector (spatial feature extraction, up to the global average pool layer).
2. **Multiheaded self-attention (MHSA) block:** 8 attention heads, feature dimension 512. Allows the model to attend selectively to the most informative timepoints in the visit sequence.
3. **LSTM module:** hidden state dimension 512. Models temporal dynamics across the ordered sequence of visit embeddings.
4. **Fully connected classifier:** 512 neurons → 2-class output (event / no event within 1 year).

### 7.2 Temporal Learning — Self-supervised Pretraining

**Problem:** Longitudinal data is scarce and labels are expensive. Directly training the full pipeline from scratch on a small labelled set leads to overfitting and poor generalisation.

**Solution — Temporal Learning (self-supervised pretext task):** The model is pretrained by presenting patients' serial scans in a *random chronological order* and tasked with **classifying whether the sequence is in the correct chronological order** (binary label: 1 for correct, 0 for shuffled). This requires the model to learn meaningful scan-to-scan difference features without any disease labels.

**Oversampling for pretraining:** all possible sub-trajectories from each patient's scan history are enumerated (varying lengths), then shuffled to create both positive (correct order) and negative (shuffled) samples. For the DFCI/BCH dataset, 278 patients yielded 3,531 trajectories.

The pretrained weights are then **fine-tuned** on the downstream EFS task with a small labelled dataset.

> This approach mirrors how an expert neuroradiologist works — by recognising change patterns across serial scans rather than reading each scan in isolation.

### 7.3 Key Results

| Dataset | Approach | AUROC | F1 |
|---|---|---|---|
| DFCI/BCH pLGG | Single scan (baseline) | 0.58 | 0.57 |
| DFCI/BCH pLGG | Longitudinal (from scratch) | 0.77 | 0.67 |
| DFCI/BCH pLGG | **Temporal learning** | **0.83** | **0.80** |
| RadART (external) | Temporal learning | 0.84 | 0.73 |
| DFCI/BCH pHGG (out-of-domain) | Temporal learning | 0.89 | 0.80 |

Temporal learning improved F1 by 6.6–58.5% over standard longitudinal training across datasets. Performance increased incrementally with the number of historical scans, plateauing between 3 and 6 scans depending on the institution.

**Calibration:** ECE ranged from 0.09 to 0.20 across the four test sets, demonstrating that the model's predicted probabilities are well-calibrated against true recurrence rates — a prerequisite for clinical use.

---

## 8. Augmentation for Sequential Data

Augmentation for video and longitudinal data requires extra discipline compared to single images.

**Spatial augmentations must be applied consistently across all frames.** The same random crop, flip, and rotation must be applied to the entire clip — applying different transforms per frame introduces artificial motion that does not exist in the real data.

**Temporal augmentations** (use with caution, as they can break clinical meaning):

- Random start time (clip jitter)
- Temporal subsampling / stride changes
- Speed perturbations
- Time reversal (only valid if the clinical content is order-invariant — generally not safe for cardiac or neurological sequences)

**Key principle:** augmentations must match **clinical invariances**, not just vision invariances. For example, intensity inversion or colour jitter may be valid for natural images but destroy information encoded in MRI intensities.

---

## 9. Alignment and Image Registration

Apparent change between frames or visits can arise from patient/probe position changes rather than true physiological change. Quantitative measures (tumour volume, LV size) assume a consistent spatial reference across time.

**Decide what motion to keep vs. remove.** For example, in echocardiography, probe motion should be removed but myocardial contraction must be preserved. Aggressive registration can inadvertently remove clinically relevant deformation.

**Common alignment strategies:**

| Strategy | When to use |
|---|---|
| **Rigid / affine registration** to a baseline | Standard choice for most longitudinal structural MRI |
| **Deformable registration** | When anatomy genuinely deforms over time (e.g., tumour shrinkage) — use with caution, can distort clinical structure |
| **ROI tracking / cropping** | When the relevant anatomy is localised (e.g., crop around the heart) |

Always **ablate with and without alignment** and inspect failure cases — alignment is not always beneficial.

---

## 10. Sampling Strategy

The model sees only what you sample: clip length, start time, and stride jointly determine what temporal information is available.

**Frame spacing (stride) defines temporal resolution:**
- Low frame rate → captures stable anatomy and context
- High frame rate → captures motion patterns

Different sampling choices can change the clinical content — a short clip may capture only part of a heartbeat, while a longer clip captures the full cycle.

**Common strategies:**
- **Uniform temporal subsampling:** spread frames evenly across the full video to cover all content at fixed input size.
- **Random contiguous clip:** fixed-length clip starting at a random time — good for learning stable local motion patterns.
- **Sliding window inference:** for long videos, produce predictions at overlapping windows and aggregate.

> **Treat rate, length, stride, and augmentation as ablation axes** — always tune them relative to the physiology of interest.

---

## 11. Label Sparsity in Medical Video

Dense per-frame labels are rare in medical video:
- Expert annotation is expensive — only a subset of frames is typically labelled.
- Many targets are inherently sequence-level (EF, diagnosis).

**Common label regimes and modelling responses:**

| Regime | Approach |
|---|---|
| **Sequence label** (video → diagnosis / EF) | Standard sequence classification / regression |
| **Sparse frame labels** (only ED/ES labelled; few frames have masks) | Weak supervision — generalise from sparse labels to all frames (as in EchoNet) |
| **Noisy labels** (inter-observer variation, ambiguous boundaries) | Masked loss (apply pixel loss only on labelled frames); noise-robust losses |

**Modelling patterns:** multi-task learning (segmentation + sequence prediction); masked loss applied only to labelled frames; self-supervised or weakly supervised pretraining on unlabelled sequences.

---

## 12. Probability Calibration (Recap)

High accuracy does not equal reliable decision support. Models can be overconfident or underconfident, and miscalibration leads to unsafe clinical decision thresholds.

**Reliability diagrams** visualise calibration: group predictions into $M$ confidence bins and plot empirical accuracy vs. predicted confidence. A perfectly calibrated model lies on the diagonal.

**Expected Calibration Error (ECE)** measures the average mismatch across bins:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

where $B_m$ is the set of predictions in bin $m$, $n$ is the total number of predictions, $\text{acc}(B_m)$ is the empirical accuracy in the bin, and $\text{conf}(B_m)$ is the mean predicted confidence.

**Temperature scaling:** a simple post-hoc calibration method. The logits $\mathbf{z}_i$ are divided by a scalar temperature $T$ before the softmax:

$$p_i = \text{softmax}\!\left(\frac{\mathbf{z}_i}{T}\right)$$

$T$ is optimised on the validation set. $T > 1$ softens predictions (reduces overconfidence); $T < 1$ sharpens them.

*Reference: Guo et al. "On calibration of modern neural networks." ICML, 2017.*

---

## 13. Calibration in Temporal Settings

Frame-level calibration does not imply sequence-level calibration.

- Model predictions may be individually calibrated at the frame level, but the clinical decision is usually made **per exam** based on aggregated predictions.
- **Calibration can break after temporal aggregation.** Always compute and report calibration at the same unit as evaluation (clip / exam / patient).

**Temporal instability:** frame-to-frame predictions may fluctuate, producing unstable confidence curves over time.

- Plot calibration over time: $\text{ECE}(t)$ or accuracy vs. confidence per timestep.
- Standard ECE assumes i.i.d. samples — for temporally correlated predictions this may **underestimate miscalibration**.

**Practical recommendations:**
- For a video-level classifier, aggregate logits or probabilities across clips first, then compute ECE.
- Compare calibration before and after aggregation.
- Check temporal stability (confidence over time) as part of model validation.

---

## 14. Evaluation — Dataset Splitting

Temporal data introduces additional ways to introduce leakage and inflate results. The split strategy must be made explicit and appropriate:

| Split type | Description |
|---|---|
| **Patient-level split** | No patient appears in more than one split — mandatory minimum |
| **Study / exam-level split** | All frames/clips from one exam stay in the same split |
| **Chronological split** | Train on earlier data, test on later data — simulates real deployment |
| **Online vs. offline tasks** | Online tasks must be evaluated using past frames only (no future leakage) |
| **Cross-site / device split** | Train on some hospitals/devices, test on unseen ones |
| **Class balance** | Maintain label distribution across splits, especially for rare pathologies |

> Always clearly state the split strategy, the sample unit (frame / clip / exam / patient), and whether clips overlap.

*Reference: Tampu et al. "Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images." Scientific Data, 2022.*

---

## 15. Evaluation — Metrics

Two complementary dimensions should be assessed for temporal models:

**Correctness:** standard task metrics (AUROC, MAE, Dice, F1) computed per patient/video.

**Temporal reliability:** the model should produce stable, physiologically plausible predictions over time.
- **Prediction smoothness:** magnitude of probability changes between consecutive frames.
- **Mask consistency:** Dice between consecutive predicted segmentation masks.
- **Physiology sanity checks:** e.g., the LV volume curve should have a clear maximum at ED and minimum at ES.

Beware: temporal reliability metrics can **over-penalise real rapid motion** — always interpret together with correctness metrics and qualitative analysis.

**Aggregation level:** metrics should be computed **per patient/video**, then summarised (median + IQR + mean). Averaging per clip would over-weight patients with many clips. Report distributions (boxplots or per-case tables) rather than single means, and estimate uncertainty with bootstrap confidence intervals over patients.

---

## 16. Temporal Data Formats

A temporal sample is not just pixels — it is pixels + **ordering** + **timing metadata** + spatial metadata.

**DICOM sequences:** stored as a series of instances or as multi-frame files. Ordering may depend on tags like `Temporal Position Identifier` — always verify with metadata and visualisation.

**4D NIfTI (dynamic MRI / fMRI):** time is the 4th dimension. The header affine encodes the spatial reference (voxel → real-world coordinates).

**Longitudinal BIDS:** encode visits as sessions (e.g., `ses-<label>`); store acquisition time in session metadata files.

> **Never assume ordering.** Always run a sequence sanity check — visualise a few complete sequences before training.

---

## 17. Temporal Data Pitfalls

| Pitfall | Consequence |
|---|---|
| **Temporal leakage** | Splitting by frames/slices, or using future frames for online tasks → artificially inflated metrics |
| **Ordering mistakes** | Wrong frame order, missing frames, duplicated frames → temporal signal is destroyed |
| **Sampling mismatch** | Training on a fixed clip length but evaluating on a different length/frame rate → distribution shift at inference |
| **Augmentation mistakes** | Random crop/flip per frame → introduces artificial motion; time-reversal may clinically invalid |
| **Longitudinal confounds** | Scanner or protocol changes over years mimic disease progression → false signals |

---

## 18. Key Takeaways

- **Time encodes clinical signal.** Treating sequential images as independent frames is almost always a modelling error.
- **Start with late fusion.** It handles variable-length sequences, is robust in low-data settings, and provides a strong, debuggable baseline. Add a temporal model (mid or early fusion) only once the baseline is solid.
- **Match architecture to task.** Order/events → mid fusion (LSTM/GRU); motion features → early fusion (3D conv, factorised); long-range context → transformers.
- **Longitudinal ≠ video.** Irregular intervals, missing visits, and time-varying confounders require explicit modelling of $\Delta t$ and harmonisation strategies.
- **Self-supervised temporal pretraining works.** The chronological order prediction task (temporal learning) gives the model a way to learn change features from unlabelled scan sequences — a powerful strategy when labelled longitudinal data is scarce.
- **Calibration at the right level.** Frame-level calibration does not guarantee exam-level calibration. Always compute and report ECE at the same aggregation unit as the clinical decision.
- **Splitting and sampling are model choices.** The choice of clip length, stride, and split strategy directly affects what your model learns and how valid the evaluation is — treat them as ablation axes.

---

## 19. Key Papers

| Paper | Contribution |
|---|---|
| Ouyang et al. (2020). *Video-based AI for beat-to-beat assessment of cardiac function.* Nature 580. | EchoNet-Dynamic: spatiotemporal CNN + weak supervision for EF regression and LV segmentation from echocardiography; first video-based DL model for echo; beats human inter-observer variability |
| Tak et al. (2025). *Longitudinal Risk Prediction for Pediatric Glioma with Temporal Deep Learning.* NEJM AI. | Temporal learning (self-supervised chronological order pretext task) + MHSA + LSTM for longitudinal MRI risk prediction; F1 improvement of 6.6–58.5% over standard longitudinal baselines |
| Guo et al. (2017). *On calibration of modern neural networks.* ICML. | Demonstrates modern neural networks are poorly calibrated; introduces temperature scaling as a simple post-hoc calibration method; formalises reliability diagrams and ECE |
| Tampu et al. (2022). *Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images.* Scientific Data 9:580. | Demonstrates how frame-level splitting (instead of patient-level) dramatically inflates reported accuracy in medical video/sequential imaging tasks |
| Tran et al. (2018). *A closer look at spatiotemporal convolutions for action recognition.* CVPR. | Introduces R2+1D decomposed spatiotemporal convolutions used in EchoNet-Dynamic |
| Bertasius et al. (2021). *Is space-time attention all you need for video understanding?* ICML. | TimeSformer: factorised space-time transformer attention for video |
| Liu et al. (2022). *Video Swin Transformer.* CVPR. | Local spatiotemporal window attention; more efficient than global attention for video |

---

*Notes compiled from: 08_MEDIMG_sequential.pdf, Ouyang et al. (2020) Nature, Tak et al. (2025) NEJM AI — HSLU Medical Image Analysis, Dr. Ludovic Amruthalingam*
