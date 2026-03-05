# Week 01 — Introduction to Medical Image Analysis

**Course:** Medical Image Analysis — HSLU  
**Lecturer:** Dr. Ludovic Amruthalingam  
**Topics:** Medical imaging use cases · Modalities · Clinical challenges · The promise of AI · ML tasks · Real-world projects · Challenges & pitfalls · Reading papers

---

## Table of Contents

1. [What is Medical Imaging?](#1-what-is-medical-imaging)
2. [Imaging Modalities](#2-imaging-modalities)
   - [X-ray Radiography](#21-x-ray-radiography)
   - [Computed Tomography (CT)](#22-computed-tomography-ct)
   - [Magnetic Resonance Imaging (MRI)](#23-magnetic-resonance-imaging-mri)
   - [Ultrasound](#24-ultrasound-imaging)
   - [Nuclear Imaging: PET & SPECT](#25-nuclear-imaging--pet--spect)
   - [Other Modalities](#26-other-modalities)
3. [The Rise of Medical Imaging](#3-the-rise-of-medical-imaging)
4. [Clinical Challenges](#4-clinical-challenges)
5. [The Promise of AI — Augmented Intelligence](#5-the-promise-of-ai--augmented-intelligence)
6. [Medical Images vs. Natural Images](#6-medical-images-vs-natural-images)
7. [ML Tasks for Medical Imaging](#7-ml-tasks-for-medical-imaging)
   - [Classification](#71-classification)
   - [Segmentation](#72-segmentation)
   - [Object Detection](#73-object-detection)
   - [Image Enhancement](#74-image-enhancement)
8. [Real-World Project Case Studies](#8-real-world-project-case-studies)
   - [PASSION Project](#81-the-passion-project--teledermatology-for-sub-saharan-africa)
   - [Hand Eczema Assessment](#82-hand-eczema-assessment)
9. [Challenges and Pitfalls](#9-challenges-and-pitfalls)
10. [Reading ML Research Papers](#10-reading-ml-research-papers)

---

## 1. What is Medical Imaging?

Medical imaging creates **visual representations of the human body** for clinical analysis and intervention. It is the primary non-invasive window into the body.

**Clinical use cases:**
- **Early detection** of asymptomatic conditions (e.g., cancer screening)
- **Treatment planning and monitoring** — track response to therapy
- **Surgical guidance** — intraoperative navigation
- **Patient documentation & follow-up** — longitudinal comparison
- **Information transfer** between clinical experts
- **Teaching** and training of medical professionals

---

## 2. Imaging Modalities

A **modality** defines how an image is acquired and what physical signal is measured. Each modality reveals different information about the body.

| Category | What it shows | Examples |
|---|---|---|
| **Structural / Anatomical** | Anatomy, shape, structure | X-ray, CT, MRI |
| **Functional** | Physiological/metabolic activity | PET, SPECT, fMRI |
| **Combined** | Both structural and functional | PET-CT |

**Historical timeline of imaging invention:**
- 16th century — Microscope
- 1895 — X-ray machine
- 1956 — Ultrasound
- 1972 — CT
- 1977 — MRI

---

### 2.1 X-ray Radiography

**Physical principle:** High-energy electromagnetic radiation (photons) is sent through the body to a detector, producing a **2D projection image**.

- Intensity values = sum of attenuations along the beam path
- **Dense structures** (bone) absorb more X-rays → appear **white**
- **Soft tissues** absorb less → appear **darker**
- Projection means **no depth information** (structures overlap)

**Clinical applications:**
- Chest imaging — lungs (infection, fluid), rib fractures
- Skeletal surveys — broken bones, dental exams
- Mammography — breast cancer screening

**Pros & cons:**

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| Fast & inexpensive | Ionising radiation |
| Excellent for bone/chest | Limited soft-tissue contrast |
| Widely available | No depth (2D projection only) |

---

### 2.2 Computed Tomography (CT)

**Physical principle:** X-ray beams are **rotated around the body**; multiple projections are reconstructed into **2D cross-sectional slices** of a defined thickness (0.5–5 mm), yielding a **3D image stack**.

Key concepts:
- **Voxel** — a 3D volume element; each intensity value corresponds to a voxel
- **Hounsfield Units (HU)** — standardised attenuation scale relative to water (0 HU) and air (−1000 HU)
- **Anisotropic resolution** — out-of-plane (z-axis) resolution is typically lower than in-plane (x,y) resolution

**Clinical applications:**
- **Trauma** — detect internal injuries, bleeding
- **Oncology** — locate and characterise tumours
- **Cardiology** — assess coronary arteries

**Pros & cons:**

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| Better resolution/contrast than plain X-ray | Higher radiation dose than single X-ray |
| Reveals overlapping structures | — |
| High diagnostic value | — |
| 3D volumetric data | — |

---

### 2.3 Magnetic Resonance Imaging (MRI)

**Physical principle:**
1. A **strong magnetic field** aligns hydrogen protons in the body
2. A **radiofrequency (RF) pulse** disturbs their alignment
3. Signals emitted as protons **relax back** are measured

Key concepts:
- Data can be 2D slices, 3D volumes, or **4D** (3D + time, e.g., cardiac MRI)
- MRI intensities have **no fixed physical scale** → require normalisation before ML
- Different **sequences** highlight different tissue properties:

| Sequence | What it emphasises |
|---|---|
| **T1-weighted** | Fat appears bright, fluid dark — good for anatomy |
| **T2-weighted** | Fluid appears bright — better for pathology detection |
| **FLAIR** | Suppresses cerebrospinal fluid — useful in brain lesion detection |
| **PD-weighted** | Proton density — tissue differentiation |

- Multiple sequences per exam → treated as **multi-channel input** in ML models

**Clinical applications:** Brain, spinal cord, ligaments, muscles, heart, soft tissue tumours

**Pros & cons:**

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| Excellent soft-tissue contrast | Slower than CT |
| No ionising radiation | Noisy, enclosed scanner (claustrophobia) |
| Multi-sequence flexibility | Expensive |
| 2D/3D/4D acquisition | Requires intensity normalisation |

---

### 2.4 Ultrasound Imaging

**Physical principle:** A probe emits **high-frequency sound waves** and records **echoes** reflected from tissue boundaries, generating **real-time 2D images**.

Key concepts:
- **Strong echo** at fluid–tissue boundaries; **weak echo** in homogeneous tissue
- **Cannot image** through air (scatters sound) or bone (blocks it)
- **Resolution–depth trade-off:** higher frequency = higher resolution but shallower penetration
- Image quality is **operator-dependent** — probe position changes the image
- Images appear **grainy/speckled** (speckle noise)

**Clinical applications:**
- **Obstetrics** — fetal monitoring
- **Cardiology** — echocardiography (beating heart in real-time)
- **Emergency medicine** — detect internal bleeding
- **Guidance** of needle biopsies

**Pros & cons:**

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| Portable and widely accessible | Limited field of view |
| No radiation, safe | Lower resolution than CT/MRI |
| Real-time imaging | Operator-dependent quality |
| Inexpensive | Cannot image through air/bone |

---

### 2.5 Nuclear Imaging — PET & SPECT

**Purpose:** Visualise **functional or metabolic activity** of tissues, not anatomy.

**How it works:**
- A **radioactive tracer** targeting a specific biological process (e.g., glucose metabolism) is injected
- **PET** detects annihilation photons from positron emission
- **SPECT** detects gamma photons directly from the decaying tracer
- A **3D image of tracer concentration** is reconstructed

**Key properties:**
- **Low spatial resolution:** ~4 mm for PET, ~8 mm for SPECT
- Displayed as **colour heatmaps** (hot colours = high uptake)
- Often **combined with CT or MRI** (e.g., PET-CT) for fused structural + functional imaging

**Clinical applications:**
- **Oncology** — tumours show high glucose uptake (FDG-PET)
- **Neurology** — differentiate dementia types by metabolic patterns
- **Cardiology** — assess myocardial blood flow

---

### 2.6 Other Modalities

Beyond the "big five", many other imaging types are used in clinical practice and ML research:

| Modality | Description |
|---|---|
| **Clinical photography** | Standard camera images of skin conditions, wounds |
| **Full-body imaging** | Standardised 360° photography (e.g., for mole mapping) |
| **Dermoscopy** | Magnified, polarised light imaging of skin lesions |
| **Colonoscopy** | Endoscopic camera imaging of the colon |
| **Fundoscopy** | Camera imaging of the retina |
| **Histology** | Microscopic images of stained tissue sections (often gigapixel) |

---

## 3. The Rise of Medical Imaging

- Medical imaging is **mostly painless and non-invasive** — often the only way to look inside the body without surgery
- Accounts for **~90% of the total digital footprint** in modern health systems (Forbes, 2024)
- **Billions of imaging procedures** performed globally each year (Forbes, 2026)

---

## 4. Clinical Challenges

Despite its power, medical imaging in clinical practice faces major structural and cognitive challenges:

### Volume & Workforce
- **Growing imaging volume** paired with a **shortage of trained specialists** (radiologists, pathologists) and an ageing population

### Time Constraints & Cognitive Load
- Radiologists may read **hundreds of images per study**, dozens of studies per shift
- CT/MRI exams can contain **hundreds of slices**
- Data is increasingly **multimodal and longitudinal**: imaging + clinical notes + lab results + anamnesis

### Subtle & Rare Cases
- Some pathologies are **millimetres in size** (e.g., micro-calcifications, micro-bleeds)
- Others have **very low prevalence**, making them hard to detect and for which little training data exists

### Inter- and Intra-rater Variability
- **Inter-rater variability:** different experts disagree on diagnosis, lesion boundaries, severity grading
  - Depends on experience level, image quality, case ambiguity
- **Intra-rater variability:** the same expert gives different interpretations over time due to fatigue, time pressure, contextual framing

### Medico-Legal Pressure
- Diagnostic errors can have **severe consequences for patients**
- Clinicians bear **legal responsibility** for their decisions

---

## 5. The Promise of AI — Augmented Intelligence

AI in medical imaging is positioned as **augmented intelligence** — a tool to assist, not replace, clinicians.

| AI Capability | Clinical Benefit |
|---|---|
| **Consistency & reproducibility** | Deterministic outputs reduce inter/intra-rater variability; standardised measurements across institutions |
| **Efficiency & scalability** | Triage urgent cases, pre-screen large populations, reduce cognitive load and burnout |
| **Quantitative measurements** | Precise lesion volume, longitudinal comparison of disease progression |
| **Sensitivity to subtle patterns** | Detection of microcalcifications, small nodules, micro-bleeds invisible to the naked eye |
| **Decision support in low-resource settings** | Assist clinicians in regions with limited specialist access; telemedicine |
| **Multimodal & longitudinal integration** | Process imaging + clinical notes + lab data simultaneously |

---

## 6. Medical Images vs. Natural Images

Medical images differ fundamentally from natural photos (ImageNet-style), which has important implications for model design.

### Dimensionality
- **2D:** X-ray, ultrasound (single slice)
- **3D volumes:** CT, MRI (stack of 2D slices)
- **4D (3D + time):** Dynamic sequences, e.g., cine MRI, cardiac ultrasound

### Channels
- MRI often acquired with multiple sequences (T1, T2, FLAIR) → treated as **multiple input channels** (analogous to RGB channels)
- PET-CT involves **multimodal fusion** of anatomy and function

### Voxel Intensities
- **CT** uses Hounsfield Units (HU) — a physically meaningful, standardised scale
- **MRI** uses relative intensity — no fixed scale, requires **normalisation** before use in ML

> **Key takeaway:** The structure of medical images (dimensionality, channels, intensity scale) directly informs **model architecture choices** and **preprocessing pipelines**.

---

## 7. ML Tasks for Medical Imaging

### 7.1 Classification

**Task:** Predict one or more labels for an entire image.

- Binary: e.g., "pneumonia" vs. "normal" on chest X-ray
- Multi-class: e.g., skin condition type
- Multi-label: multiple findings co-present

**Example applications:** Differential diagnosis, triage, screening

**Common architectures:** CNNs (ResNet, EfficientNet, DenseNet), Vision Transformers (ViT, DINOv2), foundation models

---

### 7.2 Segmentation

**Task:** Classify **every pixel/voxel** in the image, partitioning it into labelled regions.

- **Semantic segmentation:** each pixel gets a class label
- **Instance segmentation:** distinguishes individual object instances

**Example applications:** Tumour segmentation, organ delineation, shape analysis, lesion size quantification, severity grading

**Key model:** U-Net (encoder-decoder with skip connections) — the standard architecture for medical image segmentation

---

### 7.3 Object Detection

**Task:** Localise objects of interest with bounding boxes and assign class labels.

**Example applications:**
- Polyp detection in colonoscopy video
- Mole/lesion screening
- Dental charting in panoramic X-rays (detecting teeth, crowns, implants, bridges)

---

### 7.4 Image Enhancement

**Task:** Improve image quality from faster or lower-dose acquisitions using deep learning.

**Example application:** Reduce gadolinium (contrast agent) dose in brain MRI — a deep learning model trained to reconstruct a full-dose image from a 10% dose input.

> Clinical benefit: reduces patient exposure to a potentially toxic contrast agent while preserving diagnostic quality.

---

## 8. Real-World Project Case Studies

### 8.1 The PASSION Project — Teledermatology for Sub-Saharan Africa

**Problem:** Limited access to dermatology specialists in sub-Saharan Africa; 5 common skin conditions (eczema, impetigo, ringworm, scabies, insect bites) cause high burden.

**Solution:** An ML-powered teledermatology system:
- Automated triage of the 5 most common skin conditions
- Treatment strategy suggestions
- Empowers primary care workers and patients

**Study design (3 phases):**

| Phase | Description |
|---|---|
| **A** | In-hospital evaluation: compare ML model vs. dermatologist vs. live consultation |
| **B** | Comparison of human vs. ML-based diagnosis with 3-month follow-up |
| **C** | Prospective long-term evaluation with 12-month follow-up |

An **iterative process** is used: if the ML model is insufficiently accurate, the cycle restarts with newly collected clinical data.

*Reference: Gottfrois et al., MICCAI 2024*

---

### 8.2 Hand Eczema Assessment

**Problem:** Hand eczema affects 5–10% of the population; 50–65% of cases become chronic. Severity assessment is subjective and time-consuming for clinicians.

**Clinical workflow:** Patient examination → Differential diagnosis → **Severity grading** ← (ML targets this step) → Treatment selection

**Step 1 — Lesion segmentation:**
- Dataset: 312 photos labelled by 11 dermatologists
- Model: U-Net
- Precision: 76% (CI 64–82), Sensitivity: 68% (CI 55–81)

**Step 2 — Anatomy segmentation (37 hand regions):**
- Dataset: 215 photos labelled by one medical student
- Model: U-Net
- Precision: 83% (CI 80–85), Sensitivity: 85% (CI 82–88)

**Output — Anatomical stratification report:**  
The eczema and anatomy predictions are merged to produce a structured clinical report. Example output:

> *"The patient's hands show eczema lesions on both the palmar and dorsal sides, namely on 4.8% of the fingertips, 11% of the fingers (without tips), 1.5% of the palms, 3% of the back and 1.1% of the wrists."*

- **Predicted surface intraclass correlation (ICC):** 0.94 (CI 0.90–0.96) — excellent agreement with human annotations

*Reference: Amruthalingam et al., Experimental Dermatology, 2023*

---

## 9. Challenges and Pitfalls

### Data-Level Challenges

| Challenge | Description |
|---|---|
| **Imaging artifacts** | Motion blur, tissue inhomogeneity, aliasing, metallic implants, low SNR |
| **Sparse & noisy labels** | Expert annotation is expensive; inter/intra-rater variability introduces noise |
| **High data volume** | 3D CT/MRI with hundreds of slices; pathology slides are gigapixel images |
| **Heterogeneous data** | Distribution shifts between scanner vendors; different acquisition protocols across hospitals |
| **Isolated silos** | Images stored in separate hospital systems; no centralised access |
| **Specialised formats** | DICOM files contain rich metadata requiring domain expertise and strict anonymisation |

### Model & Deployment Challenges

| Challenge | Description |
|---|---|
| **Maintenance & drift** | Model performance degrades over time as imaging practices change |
| **Data bias / scarcity** | Can lead to poor generalisation; performance gap between retrospective benchmark and clinical deployment |
| **Lack of transparency** | Hinders clinician trust, adoption, and regulatory approval |
| **Regulations** | Data privacy laws (GDPR), explainability requirements, accountability frameworks |
| **Workflow integration** | AI must fit clinical workflow: too many false positives increase burden; unclear output will be ignored; clinicians must be trained |

> **Key insight:** Successful AI solutions in healthcare require **close collaboration between technologists and clinicians** from the start.

---

## 10. Reading ML Research Papers

### Why Read Papers?

- Keeps your skillset **up-to-date** with latest trends and methods
- Builds **deeper understanding** of methodology assumptions, use-cases, strengths, limitations
- Source of **inspiration** for real-world solutions
- Lifelong learning improves **employability** (World Economic Forum)

### Structure of a Typical ML Paper

| Section | Content |
|---|---|
| **Abstract** | Short summary of main idea and results |
| **Introduction** | Problem, why it matters, paper contribution |
| **Related Work** | Previous approaches, how this paper differs |
| **Method/Approach** | Technique, model architecture, algorithm |
| **Experiments & Results** | Evaluation on real data, comparison with baselines |
| **Discussion** | Interpretation of results, limitations |
| **Conclusion & Future Work** | Summary, open questions, next steps |

### Where to Find Papers

- **PapersWithCode** — papers with code implementations and benchmarks
- **arxiv-sanity** — filtered arXiv browser
- **Google Scholar** — broad academic search

**Top venues for medical image analysis:**
- **MICCAI** — Medical Image Computing and Computer-Assisted Intervention (top venue)
- **NeurIPS, ICML, ICLR** — general ML conferences
- **JMLR** — Journal of Machine Learning Research

**Strategy:**
1. Start with a **review or survey paper** to get keywords, taxonomy, and a reference list
2. Check **citation count** and **abstract** to assess relevance
3. Follow the reference chains of relevant papers

### How to Read a Paper (4-Pass Method)

*(Based on Richmond, Alake. Nvidia Technical Blog, 2021)*

**Pass 1 — Context & intuition:** Title + Abstract + Conclusion  
→ Note keywords, datasets, methods. Understand scope and objectives.

**Pass 2 — Familiarisation:** Introduction + Figures  
→ Understand the problem, prior work, research gap, paper contribution. Figures provide a critical visual overview.

**Pass 3 — Deep reading:** Full paper (skip complex math)  
→ Note key insights and takeaways. Mark sections you skipped.

**Pass 4 — Final pass:** Return to skipped sections  
→ Use external resources (lectures, Wikipedia, textbooks) to fill in gaps.

**Recommended note-taking tools:** Logseq, Notion

---

*Notes compiled from: 01_MEDIMG_intro.pdf — HSLU Medical Image Analysis, Dr. Ludovic Amruthalingam*