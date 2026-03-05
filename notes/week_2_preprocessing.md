# Week 02 — Preprocessing and Data Handling

**Course:** Medical Image Analysis — HSLU  
**Lecturer:** Dr. Simone Lionetti  
**Topics:** Medical imaging data formats · Preprocessing transforms · Data cleaning · Artefacts & data quality

---

## Table of Contents

1. [Where Medical Imaging Data Comes From](#1-where-medical-imaging-data-comes-from)
2. [Data Storage: PACS](#2-data-storage-pacs)
3. [Medical Imaging Metadata](#3-medical-imaging-metadata)
4. [Image Formats](#4-image-formats)
   - [General Image Formats](#41-general-image-formats)
   - [DICOM](#42-dicom)
   - [NIfTI](#43-nifti)
   - [Other Medical Formats](#44-other-medical-formats)
5. [Raw vs. Preprocessed Data](#5-raw-vs-preprocessed-data)
6. [Preprocessing Transforms](#6-preprocessing-transforms)
   - [Bit Depth and Windowing](#61-bit-depth-and-windowing)
   - [Crop and Region of Interest (ROI)](#62-crop-and-region-of-interest-roi)
   - [Patching](#63-patching)
   - [Resampling](#64-resampling)
   - [Image Registration](#65-image-registration)
   - [Deep Learning-Based Preprocessing](#66-deep-learning-based-preprocessing)
7. [Data Quality and Cleaning](#7-data-quality-and-cleaning)
   - [Garbage In, Garbage Out](#71-garbage-in-garbage-out)
   - [Common Artefacts](#72-common-artefacts)
   - [Shortcut Learning](#73-shortcut-learning)
   - [Near Duplicates](#74-near-duplicates)
   - [Off-topic Samples](#75-off-topic-samples)
8. [Tools for Data Quality](#8-tools-for-data-quality)

---

## 1. Where Medical Imaging Data Comes From

Medical images are produced by a wide variety of imaging devices. Knowing the source modality is essential because it determines the file format, intensity scale, dimensionality, and the preprocessing steps required.

**Radiology / nuclear medicine devices:**
- X-ray scanners
- Computed Tomography (CT)
- Positron Emission Tomography (PET)
- Magnetic Resonance Imaging (MRI) — sequences: T1, T2, T2\*, FLAIR, …

**Other clinical imaging devices:**
- Ultrasound (including Doppler)
- Endoscopy
- Digital microscopy (histopathology, blood-cell imaging)
- Total-body scanners (e.g. for dermatology)

---

## 2. Data Storage: PACS

Medical images are acquired in a variety of clinical settings — hospitals, radiology centres, medical practices, outpatient clinics, and (wet) analysis laboratories — and must be managed according to strict healthcare obligations.

**Healthcare data obligations:**
- **Preservation** — data must be retained for defined periods
- **Integrity** — data must not be altered
- **Retrieval** — data must be accessible on demand
- **Deletion** — data must be removable per patient rights

**Picture Archiving and Communication Systems (PACS):**
All modalities send their images to a centralised PACS server via the **DICOM protocol**. Clinicians access images from workstations (e.g. OsiriX) or web viewers (e.g. WEASIS) via DICOM Q/R or WADO protocols.

**Privacy implications:**
- Unless anonymised, medical images constitute **personal data**
- Anonymisation is legally required before sharing data for research, but is a **grey area** in practice — metadata fields, burned-in text, or even facial structure in head scans can re-identify patients
- Legal requirements for processing apply (e.g. GDPR in Europe)

---

## 3. Medical Imaging Metadata

Every medical image is accompanied by rich **metadata** that ensures traceability and is embedded directly in the file (in DICOM files, for example).

**Typical metadata fields:**
- Patient number, name, sex, date of birth
- Timestamp of acquisition
- Acquisition location (hospital / institution)
- Operator identity
- Scanner manufacturer and model
- Acquisition settings (kV, mA, slice thickness, field of view, etc.)
- Study and series identifiers

> **Important for ML:** Metadata can be a source of **confounders** and **data leaks**. For example, scanner type may correlate with patient population, causing models to learn irrelevant features. Always audit metadata before training.

---

## 4. Image Formats

### 4.1 General Image Formats

At the lowest level, an image is just a grid of numbers — raw values per pixel or voxel. How those values are stored determines file size, quality, and compatibility.

| Format | Compression | Bit depth | Transparency | Notes |
|---|---|---|---|---|
| **Raw** | None | Variable | — | Pure values per pixel/voxel |
| **TIFF** | Lossless | 24-bit | ✅ | Supports layers; used in microscopy |
| **PNG** | Lossless | 8 or 24-bit | ✅ | Good for segmentation masks |
| **JPG/JPEG** | Lossy | 8-bit | ❌ | Smallest size; introduces compression artefacts |

> ⚠️ **Never use JPG for medical images or segmentation masks** — lossy compression changes pixel values and destroys fine detail critical for diagnosis and model training.

---

### 4.2 DICOM

**DICOM (Digital Imaging and Communications in Medicine)** is the universal clinical standard for medical images.

| Property | Details |
|---|---|
| **Extension** | `.dcm` |
| **Structure** | Data and metadata stored **in one file**; the image is one entry among many metadata tags |
| **Usage** | All radiology and nuclear medicine (X-ray, CT, MRI, PET, etc.) |
| **Relevance** | Communication standard in PACS; global standard in clinical environments |
| **Viewing tools** | OsiriX, Horus |
| **Python library** | `pydicom` |

**Key characteristics:**
- A DICOM study can consist of hundreds of files (one per slice for CT/MRI)
- Metadata is stored as typed key–value pairs called **tags** (e.g. `(0010,0010)` = Patient Name)
- Anonymisation requires stripping or replacing all identifying tags — non-trivial because some tags are optional or vendor-specific

---

### 4.3 NIfTI

**NIfTI (Neuroimaging Informatics Technology Initiative)** is the standard format for research neuroimaging.

| Property | Details |
|---|---|
| **Extension** | `.nii` (or `.nii.gz` for compressed) |
| **Structure** | Header + 3D (or 3D+time) image data in one file |
| **Usage** | Brain MRI, fMRI, PET |
| **Relevance** | Optimised for numerical analysis; the **research standard** in neuroimaging |
| **Software** | FreeSurfer, FSL, ANTs, nibabel (Python) |

**Header information includes:**
- Voxel dimensions (mm) — the affine matrix mapping voxel indices to physical coordinates
- Orientation (RAS, LAS, etc.)
- Slice ordering code

> NIfTI encodes a full **affine transformation** from voxel space to world (patient) space — critical for correctly interpreting spatial positions and performing registration.

---

### 4.4 Other Medical Formats

| Format | Extension(s) | Usage |
|---|---|---|
| **NRRD** (Nearly Raw Raster Data) | `.nrrd`, `.nhdr` | Research; popular in ITK/SimpleITK workflows |
| **Analyze** | `.hdr` + `.img` (paired) | Old neuroimaging format (largely superseded by NIfTI) |
| **MetaImage** | `.mha` or `.mhd` + `.raw` | Research; used in ITK and medical image segmentation challenges |
| **ECAT, MINC, HDF** | various | Specialised research use cases |

---

## 5. Raw vs. Preprocessed Data

> **The goal of preprocessing** is to manipulate data to make it better suited for a downstream ML task.

**Key conceptual points:**

- Some modalities (e.g. MRI k-space, PET sinograms) require heavy preprocessing even before humans can interpret the data. Reconstruction of volume scans is itself a preprocessing step.
- **Preprocessing may destroy original information.** Every transform introduces an inductive bias — an assumption about what information matters.
  - The raw data represents the full information content
  - All downstream transforms are forms of inductive bias

> This means that **more preprocessing is not always better**. Carefully chosen minimal preprocessing preserves the most information for the model to learn from.

**Standard library:** PyTorch-style transform pipelines are the modern standard, implemented in **TorchVision** (2D images) and **MONAI** (3D medical images).

> ⚠️ **The order of transformations matters.** Some transforms are not commutative (e.g. normalisation before vs. after cropping yields different results). When transforms involve spatial changes, **labels (segmentation masks) must be transformed identically**.

---

## 6. Preprocessing Transforms

### 6.1 Bit Depth and Windowing

**Bit depth** determines how many distinct intensity levels can be represented:

| Bit depth | Number of levels |
|---|---|
| 8-bit | 256 |
| 12-bit | 4'096 |
| 16-bit | 65'536 |

- Radiology scanners typically produce **12–16 bit** images
- Computer screens and most DL model inputs support only **8–10 bits**
- More bits capture subtle differences — but for noisy signals, extra bit depth only increases file size without adding information

**Windowing** is the process of **linearly remapping** a selected range of intensity values to the displayable range:

- **Window Level (WL):** the centre of the intensity range of interest
- **Window Width (WW):** the width of the range
- Values below `WL - WW/2` → black; values above `WL + WW/2` → white
- Values outside the window are clipped

In CT (Hounsfield Units), different tissues require different windows:

| Window preset | WL | WW | Use case |
|---|---|---|---|
| Lung | −600 | 1500 | Pulmonary structures |
| Soft tissue | 40 | 400 | Abdomen, organs |
| Bone | 400 | 1800 | Skeletal structures |
| Brain | 40 | 80 | Intracranial pathology |

> For ML: windowing is a form of preprocessing that encodes domain knowledge about which tissue types are relevant for a given task.

---

### 6.2 Crop and Region of Interest (ROI)

Often only a **subregion** of a scan is relevant for a given task.

**Region of Interest (ROI):** the portion of the image selected for analysis.

- Reduces computational cost
- Removes irrelevant background that could distract the model
- ROI can be defined manually, by a preceding detection model, or by registration to an atlas

**Randomised crops** as data augmentation — during training, crops are randomly positioned within a defined area to increase dataset diversity.

---

### 6.3 Patching

Patching is the process of dividing a large image into a grid of smaller overlapping or non-overlapping **patches**.

**Why patching is necessary:**
- Very large images (e.g. histopathology whole-slide images, 3D CT volumes) do not fit in GPU memory
- A single whole image may have **low diversity** for self-supervised or contrastive learning
- Fine details (e.g. small lesions) require high-resolution, small-patch analysis
- Many pretrained DL backbones were originally trained at low resolution (e.g. 224×224) and need patch-level input

**Parameters to define:**
- **Patch size** — spatial extent of each patch
- **Hop size / stride** — how far the window moves between patches
- **Overlap** — `overlap = patch_size - hop_size`; overlapping patches can be aggregated at inference time (e.g. by averaging predictions)

---

### 6.4 Resampling

Resampling changes the **spatial resolution** of an image by estimating values at new grid positions.

**Interpolation methods by dimension:**

| Dimension | Available methods |
|---|---|
| 1D | Nearest (order 0), Linear (order 1), Quadratic (order 2), Cubic (order 3) splines |
| 2D | Nearest, Bilinear, Bicubic |
| >2D | Nearest; linear on triangulation; weighted neighbours |

**Trade-offs:**
- Higher-order methods (cubic) produce smoother results but can introduce ringing artefacts
- Nearest-neighbour is preferred for **categorical data** (e.g. segmentation masks) to avoid interpolating class labels
- All resampling methods introduce some degree of **aliasing** — more sophisticated approaches (e.g. polyphase filtering) reduce this

**Common use case:** Standardising voxel spacing across a dataset (e.g. resampling all CT volumes to 1×1×1 mm isotropic) so the model sees consistent physical scales.

---

### 6.5 Image Registration

**Registration** is the process of transforming one image to align it with a reference (template) image, in terms of orientation and anatomical structure.

**Why registration is needed:**
- Compare patients to each other (cross-sectional studies)
- Detect subtle disease progression (longitudinal studies)
- Apply atlas-based ROI definitions to individual images

#### Types of Registration

**Rigid transformation** — rotation + translation only; distances and angles are preserved:

$$x'_i = R_{ij} x_j + t_i \qquad R_{ij}^{-1} = R_{ji}$$

- Used when only patient positioning differs between acquisitions
- Implicitly requires resampling and cropping/padding after the transform

**Similarity transformation** — rigid + isotropic scaling:

$$x'_i = R_{ij}(s x_j) + t_i$$

**Affine transformation** — similarity + shear; any linear mapping with $\det A \neq 0$:

$$x'_i = A_{ij} x_j + t_i$$

- Parallel lines remain parallel; useful for correcting scanner geometry differences

**Non-rigid (deformable) registration** — local deformations that model tissue elasticity; used for brain registration, cardiac motion, or multi-modality alignment.

#### Classic Registration Methods

**Point matching:**
- Identify corresponding anatomical landmarks (bifurcations, bone edges) or fiducial markers
- Find the transformation that minimises the distance between matched point sets

**Intensity matching:**
- Use image similarity metrics: **MSE** (same modality) or **Mutual Information (MI)** (cross-modality, e.g. MRI–CT)
- Iteratively optimise the transformation parameters until convergence

> ⚠️ Always evaluate registration quality on images **not used in the optimisation** — otherwise scores are inflated.

#### Atlases

An **atlas** is a coordinate system for a body region together with a labelled map of anatomical structures.

- Example: **Talairach atlas** for the brain (with Brodmann areas)
- After registering a patient image to the atlas, ROIs are automatically inherited — **registration alone can replace segmentation** for well-defined anatomical structures.

#### Deep Registration

Deep learning models can learn to perform registration in a **single forward pass**, replacing iterative optimisation.

- The registration metric is used as a **training loss**
- The model learns a **dense displacement field** (shift field) or performs direct image-to-image translation
- **Synthetic deformations** can be used as self-supervised training signal (known ground-truth displacement)

| ✅ Pros | ⚠️ Cons |
|---|---|
| Inference is a single forward pass — very fast | Expensive to train |
| Can be trained without paired ground truth | Less interpretable; reduced control over deformation |

---

### 6.6 Deep Learning-Based Preprocessing

DL is increasingly used **inside the preprocessing pipeline** itself, not just in downstream analysis.

**Super-resolution** — reconstruct high-quality images from faster or lower-dose acquisitions:
- **Siemens Healthineers:** "Deep Resolve"
- **Canon Medical Systems:** "Deep Learning Reconstruction"
- **GE Healthcare:** "AIR Recon DL"

**Clinical benefits:**
- Reduced MRI scan time (less patient discomfort, higher throughput)
- Reduced CT radiation dose
- These products are **cleared by regulatory bodies** (FDA, CE) on the basis that they are non-inferior to conventional reconstruction — a key regulatory precedent for AI in clinical preprocessing

---

## 7. Data Quality and Cleaning

### 7.1 Garbage In, Garbage Out

> **Golden rule of data-driven methods: *Garbage-in, Garbage-out.***

ML models extract statistical patterns from data. If the training data contains systematic errors, the model will learn those errors as if they were signal.

| Scenario | Consequence |
|---|---|
| Training on low-quality data | Model learns noise/artefacts; poor generalisation |
| **Evaluating on low-quality data** | **Worse** — misleadingly high or low scores; silent failure after deployment |

> Models trained and evaluated on corrupted or biased data can appear to perform well during development, then fail silently in clinical deployment — a serious safety risk.

---

### 7.2 Common Artefacts

A wide range of quality issues appear in real medical imaging datasets:

| Artefact type | Description / example |
|---|---|
| **Test scans** | Phantom or calibration scans accidentally included in patient data |
| **Device malfunction** | Sensor errors, truncated reconstructions, banding artefacts |
| **Motion** | Patient movement during acquisition; blurring or ghosting |
| **Labelling errors** | Left/Right (L/R) orientation labels applied incorrectly |
| **Rulers / scale markers** | Physical rulers in dermoscopy images that provide non-medical signal |
| **Out-of-focus** | Microscopy images where the focal plane is wrong |
| **Lighting variation** | Inconsistent illumination in clinical photography |
| **Metallic implants** | Cause beam-hardening artefacts in CT (streaks around metal) |

---

### 7.3 Shortcut Learning

Artefacts become particularly dangerous when they are **correlated with the prediction target** — the model will learn to predict from the artefact rather than the underlying pathology.

**Definition:** Shortcut learning is a special case of overfitting in which the model learns a spurious correlation between a confounding feature (e.g. a ruler, a watermark, a scanner brand) and the label.

**Classic example from dermatology:**
- In a melanoma classification dataset, dermoscopy images of malignant lesions were disproportionately acquired with a ruler (for measurement)
- A model trained on this dataset learned to associate rulers with malignancy
- UMAP visualisation of embeddings showed that images separate by **image source** and **ruler presence** just as strongly as by diagnosis

> Shortcut learning is particularly insidious because it does not show up as poor training performance — the model's benchmark metrics look fine. Failures only appear in deployment when the shortcut is absent.

**How to detect:** Visualise model attention (Grad-CAM), stratify performance by confounders, audit embeddings with dimensionality reduction (UMAP, t-SNE).

---

### 7.4 Near Duplicates

**Near duplicates** are pairs of samples that are transformations of one another, or different views of the same underlying content.

**Origins:**
- Longitudinal studies — same patient imaged multiple times
- Re-acquisition (repeated scan)
- Multiple imports from different data sources

**Types:**
- Exact duplicates
- Approximate duplicates (different crops, resolutions, or watermarks)
- Different views of the same lesion/structure

**Effect on ML:**
- If near-duplicates are split across train and test sets → **data leakage**
- Model appears to generalise, but is effectively memorising training samples
- Leads to **inflated evaluation scores** that do not reflect real-world performance

> **Rule:** Always deduplicate at the **patient level**, not just the image level, before splitting train/val/test sets.

---

### 7.5 Off-topic Samples

**Off-topic samples** are data points included in a dataset by mistake that fall outside the dataset's intended scope.

**Origins:**
- Web scraping with imprecise queries
- Uncontrolled or unmonitored acquisition conditions

**Types:**
- Completely unrelated images (e.g. a photo of mice in a skin lesion dataset)
- Images with very low information content (e.g. entirely black frames, blurred images)

**Effects:**
- On models: noise during training and evaluation — degrade both accuracy and reliability
- On humans: **loss of trust** in the system when clinicians encounter clearly wrong outputs

---

## 8. Tools for Data Quality

Several open-source tools help automate data quality auditing and cleaning:

| Tool | Key capabilities |
|---|---|
| **[Voxel51](https://voxel51.com)** | Dataset visualisation, brightness/blurriness/entropy checks, near-duplicate and exact-duplicate detection |
| **[Lightly](https://lightly.ai)** | Self-supervised embedding-based deduplication and dataset curation |
| **[CleanLab](https://cleanlab.ai)** | Label error detection using confident learning; finds mislabelled samples |
| **[FastDup](https://github.com/visual-layer/fastdup)** | Fast detection of duplicates, outliers, and corrupted images in large datasets |
| **[SelfClean](https://github.com/Digital-Dermatology/SelfClean)** | Self-supervised framework for detecting off-topic samples, near duplicates, and label errors — specifically designed for medical imaging |

**SelfClean** pipeline (particularly relevant for medical imaging):
1. Compute self-supervised image representations (no labels needed)
2. Identify **off-topic samples** via agglomerative clustering (isolated points)
3. Identify **near duplicates** via pairwise distance in embedding space
4. Identify **label errors** via intra- vs. extra-class distance ratio

> Running a data quality check before any model training is now considered best practice in medical ML — it directly addresses the "garbage-in, garbage-out" problem.

---

*Notes compiled from: 02_MEDIMG_preprocessing.pdf — HSLU Medical Image Analysis, Dr. Simone Lionetti*