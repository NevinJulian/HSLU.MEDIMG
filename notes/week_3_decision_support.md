# Week 03 — Clinical Decision Support Systems

**Course:** Medical Image Analysis — HSLU  
**Lecturer:** Dr. Ludovic Amruthalingam  
**Topics:** AI in medical imaging workflows · CDSS design patterns · Integration & interoperability · Evaluation & safe deployment · Monitoring & data drift

---

## Table of Contents

1. [Clinical Terminology Recap](#1-clinical-terminology-recap)
2. [Medical Imaging Workflow](#2-medical-imaging-workflow)
3. [AI in the Medical Imaging Workflow](#3-ai-in-the-medical-imaging-workflow)
4. [Clinical Decision Support Systems (CDSS)](#4-clinical-decision-support-systems-cdss)
5. [CDSS Patterns in Medical Imaging](#5-cdss-patterns-in-medical-imaging)
6. [CDSS Input → Output → Action](#6-cdss-input--output--action)
   - [Threshold Selection and Operating Points](#61-threshold-selection-and-operating-points)
   - [Calibration](#62-calibration)
7. [Workflow Integration](#7-workflow-integration)
   - [Integration Choices](#71-integration-choices)
   - [Human Factors](#72-human-factors)
   - [Iterative Rollout](#73-iterative-rollout)
   - [Technical and Clinical Considerations](#74-technical-and-clinical-considerations)
8. [How CDSS Change Human Behaviour](#8-how-cdss-change-human-behaviour)
9. [Levels of Autonomy](#9-levels-of-autonomy)
   - [Autonomy and Risk](#91-autonomy-and-risk)
10. [Latency](#10-latency)
11. [Interoperability](#11-interoperability)
    - [IHE Standards](#111-ihe-standards)
12. [Evaluation](#12-evaluation)
    - [What to Measure](#121-what-to-measure)
    - [Silent Trials](#122-silent-trials)
13. [Monitoring Deployed CDSS](#13-monitoring-deployed-cdss)
14. [Data Drift Recap](#14-data-drift-recap)
15. [Case Study: Triage for Intracranial Hemorrhage](#15-case-study-triage-for-intracranial-hemorrhage)

---

## 1. Clinical Terminology Recap

Before diving into CDSS design, it is important to be fluent in the vocabulary used in radiology workflows.

| Term | Definition |
|------|------------|
| **Study** | One complete imaging exam for a patient — e.g. "CT head" |
| **Series** | A set of images within a study — e.g. one CT reconstruction |
| **Worklist** | The ordered list of studies waiting to be interpreted by a radiologist |
| **Report** | The clinical interpretation document written by the radiologist |
| **Triage** | The act of prioritising which studies should be read first, based on urgency |
| **False negative** | A case where urgent disease is present but the system fails to detect it |
| **Retrospective study** | An evaluation conducted using previously acquired and already-labelled cases |

> **Why these terms matter for ML engineers:** Building a CDSS means integrating into an existing clinical process with well-defined roles and handoffs. Misunderstanding terms like "stat" vs "routine" can have direct patient safety implications.

---

## 2. Medical Imaging Workflow

The standard radiology pipeline follows these steps:

```
Order → Acquisition → Image storage → Worklist → Interpretation → Report → Communication
```

**Step-by-step:**
1. A clinician (e.g. GP or specialist) orders an imaging study.
2. Technologists acquire the images using the appropriate modality (CT, MRI, X-ray, …).
3. Images are transferred to storage (PACS) and appear in the radiologist's reading worklist.
4. Radiologists work through the worklist, interpret each study, and write a report.
5. The report is communicated back to the referring clinician to guide care decisions.

**Many roles are involved:** referring clinician, imaging technologist, radiologist, IT systems, patient.

**Three critical axes for system design:**
- **Latency** — how fast does the system respond?
- **Interoperability** — does it connect to existing hospital systems?
- **User interface** — is it accessible and non-disruptive for clinicians?

---

## 3. AI in the Medical Imaging Workflow

AI can be integrated at multiple points in the pipeline (see Tejani et al. 2024 for an overview):

- The **EHR** (Electronic Health Record) and **RIS** (Radiology Information System) feed orders, schedules, and patient information — AI can assist here for protocol selection.
- The **Modality** (CT scanner, MRI, …) acquires images in DICOM format and sends them to the **PACS** (Picture Archiving and Communication System).
- An **AI Orchestrator** can receive DICOM series from PACS, route them to the right AI algorithms, and return results back into the reporting system or viewer.
- **AI Algorithms** may run at multiple stages: before acquisition (protocol optimisation), directly at the modality, at the PACS level, or inside the reporting system.

**Decision support is primarily relevant at three stages:**
- **Worklist** — re-prioritisation of studies based on urgency
- **Interpretation** — assisting the radiologist while they read
- **Reporting** — auto-populating structured findings

---

## 4. Clinical Decision Support Systems (CDSS)

> *"Enhance medical decisions with targeted clinical knowledge, patient information, and other health information."*

**History:** Software-based CDSS can be traced to the **1970s**, when they were initially rule-based (explicit if-then rules). Today's systems use ML/DL and are used at the **point-of-care**, where clinicians combine CDSS suggestions with their own clinical judgement.

**The full CDSS pipeline:**

```
Input acquisition → Preprocessing → Inference → Result packaging → Workflow integration → Monitoring
```

**Common CDSS use cases:**
- Reviewing drug dosing, detecting therapy duplication, and checking drug–drug interactions
- Checking adherence to clinical guidelines, providing follow-up and treatment reminders
- Diagnostic support (e.g. "likely pneumonia — confidence 87%")
- Improving documentation and automating administrative tasks in clinical workflows
- **Cost containment:** reducing duplicate tests, suggesting cost-equivalent treatments

---

## 5. CDSS Patterns in Medical Imaging

Five main design patterns define how AI is used in imaging CDSS:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Triage / Prioritisation** | Move likely-urgent cases to the top of the worklist | Flag suspected ICH → promote to "stat" |
| **Second reader** | Highlight candidate findings to reduce missed diagnoses | Outline potential lung nodule for radiologist review |
| **Quantification** | Automatically compute measurements (sizes, volumes) for staging or follow-up monitoring | Tumour volume over time in oncology |
| **Structured reporting support** | Auto-populate a radiology report with consistent, pre-filled findings and metrics | Pre-filled BI-RADS scoring fields for mammography |
| **Autonomous screening** | System makes the final screening decision in a constrained, validated setting | Automated diabetic retinopathy screening in low-resource settings |

> The appropriate pattern depends entirely on the clinical setting, the urgency of the disease, and the available regulatory framework.

---

## 6. CDSS Input → Output → Action

A CDSS converts raw model outputs into clinical actions. The model typically outputs a **probability or score**; a **decision rule** (threshold) then converts that into a discrete action.

### 6.1 Threshold Selection and Operating Points

Choosing a threshold is choosing an **operating point** on the ROC curve. This affects:

- **False positives** — unnecessary urgency, wasted radiologist time
- **False negatives** — missed critical findings, direct patient harm
- **Patient flow** — proportion of cases upgraded to "stat"
- **Costs** — staffing, follow-up tests, interventions
- **Risk exposure** — medico-legal liability

> **Thresholds are policy decisions, not technical ones.** Questions like "what is an acceptable miss rate for ICH?" are ultimately answered by clinicians, hospital management, and regulators — not ML engineers alone.

The fundamental trade-off remains:

$$\text{Sensitivity} = \frac{TP}{TP + FN} \quad \text{vs} \quad \text{Specificity} = \frac{TN}{TN + FP}$$

Increasing sensitivity to catch more true positives will increase false positives, and vice versa.

### 6.2 Calibration

**Calibration** means that the model's output probability corresponds to the true empirical frequency of the event. If a model says 0.9 ("90% confident"), it should be correct approximately 90% of the time across similar cases.

**Why it matters:**
- Poorly calibrated models make thresholding unsafe — a probability of 0.3 might actually mean a 70% chance of disease.
- Calibration can fail silently when distribution shift occurs (new scanner type, new patient population).
- Output probabilities need to be **stable and interpretable** to be used safely in clinical practice.

**Common causes of poor calibration:** class imbalance during training, overconfident softmax outputs from large neural networks (use temperature scaling or Platt scaling to correct).

---

## 7. Workflow Integration

Even the best algorithm provides no clinical benefit if it cannot be accessed efficiently by the clinician at the right moment.

### 7.1 Integration Choices

The same model output can be delivered in very different ways:

| Integration mode | Description | Impact |
|-----------------|-------------|--------|
| **Passive flag** | Result stored; requires an extra click to view | Low disruption, low uptake |
| **Visible label in worklist** | A colour or icon appears next to the study in the reading list | Moderate visibility |
| **Active reprioritisation** | Study is automatically moved to the top of the worklist | High impact; radiologists must notice and respond |

### 7.2 Human Factors

Minimising extra steps reduces **cognitive load** and increases **adherence**:
- Every additional click or navigation step reduces the likelihood that a clinician will actually consult the AI result.
- UI design directly affects patient outcomes when the system is in active clinical use.

### 7.3 Iterative Rollout

Best practice for deploying a new CDSS:
1. **Silent mode** — run the model on live data, log predictions, but do not show results to clinicians
2. **Visible flags** — show results to clinicians, but as advisory-only information
3. **Operational changes** — only after performance is validated in your local population does the system trigger automated workflow changes (e.g. worklist reprioritisation)

### 7.4 Technical and Clinical Considerations

Key integration considerations from Tejani et al. (2024):

**Technical:**
- Conforming to interoperability standards (IHE, DICOM, HL7 FHIR, CDS Hooks)
- Determining which systems can trigger AI inference requests
- Managing the tradeoff between batch processing efficiency and time to result
- Integrating PACS-based AI viewers for result review
- Automating AI result insertion into radiology reports
- Determining storage format and location for AI pixel- and non-pixel-based results

**Clinical:**
- Minimising workflow disruption (number of clicks to access AI results)
- Ensuring AI results are clearly marked as advisory
- Enabling radiologists to accept, modify, or reject AI outputs
- Tracking clinical metrics: turnaround time, time to treatment, clinical outcomes

**Policy:**
- Determining the required level of local validation before real-time deployment
- Designating the reading paradigm (triage, second read, etc.) approved by regulators
- Defining whether and when AI results are accessible to patients

---

## 8. How CDSS Change Human Behaviour

Deploying a CDSS does not only change what the software does — it changes **how clinicians work**. This has both beneficial and harmful consequences.

| Behaviour | Description | Risk |
|-----------|-------------|------|
| **Over-trust** | Clinicians accept AI suggestions even when they conflict with their own judgement | Harm when AI is wrong |
| **Alert fatigue** | Repeated alarms reduce responsiveness — clinicians start ignoring alerts | Missed critical findings |
| **Attention tunneling** | Clinicians focus only on what the AI has highlighted and overlook what it missed | Missed non-flagged pathology |
| **Deskilling risk** | Over time, clinicians may lose diagnostic sharpness if they become over-reliant on AI | Reduced clinical competence if AI is removed |

**Mitigations:**
- Use **conservative thresholds** (prefer high sensitivity, accept more false positives) so that misses are rare.
- Clearly position the system as **advisory only** — the radiologist always makes the final decision.
- Provide explicit **confidence/uncertainty cues** (e.g. "high confidence" vs "borderline") rather than a single binary flag.
- Include **intended use statements** and **override paths** so clinicians can easily document disagreement.

> Integration choices can either reduce or amplify these risks. A well-designed interface that makes it easy to override the AI is not a sign of distrust in the model — it is a patient safety feature.

---

## 9. Levels of Autonomy

AI systems can be designed with different degrees of autonomous decision-making. Each level carries distinct responsibilities and risks.

| Level | Description | Example | Requirements |
|-------|-------------|---------|--------------|
| **Assistive** | AI output is purely advisory; clinician fully controls every decision | Likelihood estimate: "87% chance of pneumonia" | Clinician must interpret and apply contextual judgement |
| **Human-in-the-loop** | Radiologist actively confirms or rejects the final AI-assisted output | AI recommends an examination protocol; radiologist accepts/modifies | Review, modification, acceptance or rejection at each step |
| **Human-on-the-loop** | Radiologist keeps oversight without immediate intervention per case | AI optimises image acquisition; NLP extracts anamnesis | Monitoring and feedback for improving performance over time |
| **Workflow automation** | AI triggers predefined clinical workflow actions without per-case review | Auto-routing suspected ICH brain CTs to priority worklist | Validated triage thresholds and documented fallback protocols |
| **Autonomous** | System makes the final decision within a narrow, predefined clinical scope | Auto-discharge for negative COVID chest X-rays in low-risk patients | Regulatory approval, fail-safes, formal accountability structures |

### 9.1 Autonomy and Risk

> **Increasing autonomy increases risk.**

As autonomy increases, the requirements for controlling risk become correspondingly stricter across all of: data quality, threshold validation, ongoing monitoring, and fail-safe design.

```
Assistive → Human-in-the-loop → Human-on-the-loop → Workflow automation → Autonomous
                                                              ↑ RISKS ↑
```

The autonomy level also changes **responsibility**:
- At lower levels, the clinician retains full accountability.
- At higher levels, responsibility shifts partially toward the system designer, the hospital, and the regulatory body that cleared the device.

**Key question:** Where do we draw the line between assistance and automation? This is ultimately a regulatory, ethical, and clinical governance question — not just a technical one.

---

## 10. Latency

Workflow value depends critically on **when** results are delivered. A technically excellent model that produces results too late provides no clinical benefit.

| Situation | Consequence |
|-----------|-------------|
| Result arrives after radiologist has already read the case | No benefit; wasted compute |
| Urgent alert fires after the clinical decision has been made | Confusion, distrust, alert fatigue |

**Latency requirements vary by task:**
- **Triage** requires results **before or at worklist time** — the result must arrive before the radiologist picks up the study.
- **Quantification** (e.g. tumour volume measurement) is acceptable during or after the interpretation phase, since the radiologist needs it when writing the report.

**AI workflow orchestration** (e.g. the IHE AIW-I standard) exists in part to manage AI traffic — handling requesting, scheduling, and scaling of inference jobs so that results reliably arrive within the required time window.

> **Think like a system engineer:** Where are the bottlenecks? Is inference the bottleneck, or is it data transfer, PACS routing, or network latency? How do you avoid blocking clinical workflows?

---

## 11. Interoperability

**Interoperability** is the ability of different systems (from different vendors) to exchange and correctly use information. In a hospital setting this includes:
- Transferring DICOM images between systems
- Associating AI results with the correct patient study
- Displaying AI outputs in the radiologist's viewer

**Why it is hard:** Hospitals have **multi-vendor environments** — scanners, PACS, RIS, EHR, and AI tools often come from different manufacturers. Without standards, every pair of systems would require a bespoke custom adapter, creating a fragile, high-maintenance integration landscape.

### 11.1 IHE Standards

**IHE (Integrating the Healthcare Enterprise)** develops technical frameworks that standardise health IT interoperability. Two standards are directly relevant to AI:

| Standard | Full name | Purpose |
|----------|-----------|---------|
| **AIW-I** | AI Workflow for Imaging | Describes how AI tools are orchestrated during imaging workflows — request, management, and performance of AI inference; scheduling, retries, throughput, status, and traceability |
| **AIR** | AI Results | Defines how AI-generated results are encoded and attached to the correct study — capture, distribution, and display of imaging analysis results |

**AIW-I** ensures that AI tools can be plugged into any compliant workflow orchestrator.  
**AIR** ensures that measurements, segmentations, and findings appear predictably in the radiologist's viewer, regardless of which AI vendor produced them.

**For prototypes and research projects**, adopt the following design principles even without full IHE compliance:
- **Structured outputs** (JSON with explicit fields: score, confidence, flag)
- **Explicit provenance** (model version, timestamp, input study UID)
- **Clear mapping** from model output to the action it triggers
- **Logs** for reproducibility and audit

---

## 12. Evaluation

### 12.1 What to Measure

CDSS systems induce behaviour change — the evaluation must reflect the **effect on the clinical process**, not just model performance on a held-out test set.

> *"What you measure should match what your system changes."*  
> → Evaluate as a **system** under **realistic conditions**.

Three levels of evaluation:

| Level | Metrics |
|-------|---------|
| **Model performance** | Discrimination: AUROC, sensitivity, specificity, PPV/NPV; segmentation: Dice, IoU; regression: MAE, RMSE |
| **System performance** | Inference latency, throughput, failure rate, uptime/stability |
| **Clinical / operational impact** | Time-to-diagnosis, reader accuracy with/without AI, downstream actions taken, safety incidents |

**The CDSS action determines which metrics matter most:**

| CDSS pattern | Primary metric concern |
|--------------|----------------------|
| **Triage** | Missed urgent cases (false negatives) vs. increased workload (false positive rate) |
| **Prioritise worklists** | Time-to-diagnosis, wrong-priority events |
| **Quantify structures** | Measurement consistency, test-retest reproducibility, robustness across scanners |

### 12.2 Silent Trials

**The problem:** Real hospital data differs from curated research datasets. Formats change, scanners change, acquisition protocols change, patient populations change. This leads to **distribution shift** — the model's real-world performance is worse than its retrospective evaluation suggested.

**Silent trials** address this gap:
- The model runs on **prospective patients in real time** but its outputs are **hidden from clinicians**
- Care decisions are **not influenced** — the trial acts as a dry run
- This allows validation of stability, latency, and performance under genuine operational conditions
- Safety signals can be monitored (e.g. misfires, missed critical cases) before any clinical consequences

**What silent trials reveal:**
- Integration feasibility (does the pipeline actually connect to all necessary systems?)
- Workflow fit (does the result arrive at the right time and place?)
- Model and data pipeline robustness (does it fail on edge cases? unusual scanner types?)
- Drift relative to the retrospective training/testing distribution

> Silent trials are now considered standard practice and are explicitly described in publications such as Kwong et al. (2022).

---

## 13. Monitoring Deployed CDSS

After deployment, performance can **degrade silently** — without any visible error, as the input distribution slowly shifts. Continuous monitoring is required for clinical safety, regulatory compliance, and maintaining user trust.

**Three monitoring domains:**

**1. Data & model behaviour:**
- Monitor input/output distributions and model embeddings over time
- Triggers for investigation: introduction of a new scanner, imaging protocol change, new patient population, epidemic/pandemic
- Drift can break calibration → previously safe thresholds may become unsafe
- Because labels are often unavailable post-deployment, rely on **proxy signals**: alert volume trends, score distribution shifts, proportion of flagged cases

**2. User interaction patterns:**
- Track how often clinicians accept, ignore, or override AI suggestions
- A sudden increase in overrides can indicate model brittleness or trust erosion
- Systematic ignoring of alerts = alert fatigue → threshold should be reconsidered

**3. Workflow and system health:**
- Compare inference latency against the workflow requirement
- Check integration endpoints for failures or silent errors
- Monitor throughput to ensure the system scales with study volume

**Response protocol:**

```
Threshold crossed → Alerting → Investigation → Recalibration / Retraining → Rollback / Suspension
```

---

## 14. Data Drift Recap

For a model to remain reliable, the **data probability distribution must remain consistent between training and inference**.

Causality refers to which variable generates the other:
- **Causal setting** (X causes Y, e.g. scan → disease label): covariate shift and concept shift are the primary concerns.
- **Anti-causal setting** (Y causes X, e.g. disease → scan appearance): label shift and manifestation shift are the primary concerns.
- When causality is unknown, covariate and label shifts are indistinguishable.

| Type of shift | Data P(X) | Targets P(Y) | Concept P(Y\|X) | Manifestation P(X\|Y) |
|---|---|---|---|---|
| **Covariate shift** | **Change** | Change | **Same** | Same |
| **Concept shift** | **Same** | Change | **Change** | Same |
| **Label shift** | Change | **Change** | **Same** | Same |
| **Manifestation shift** | Change | **Same** | Same | **Change** |

The full joint distribution decomposes as:

$$P(Y, X) = P(Y \mid X)\, P(X) = P(X \mid Y)\, P(Y) = P(X, Y)$$

**Practical implications:**
- A new scanner model may cause **covariate shift** (different image texture) without changing the underlying disease patterns.
- A new treatment protocol may cause **label shift** (fewer positive cases) without changing how pathology appears on imaging.
- Both can degrade model performance without any obvious error in the pipeline.

---

## 15. Case Study: Triage for Intracranial Hemorrhage

**Source:** Arbabshirani et al. (2018). *Advanced machine learning in action: identification of intracranial hemorrhage on computed tomography scans of the head with clinical workflow integration.* NPJ Digital Medicine, 1(1), 9.

### Clinical Problem

Intracranial hemorrhage (ICH) is a critical condition accounting for approximately 2 million strokes worldwide per year. Key facts:
- Intra-axial hemorrhage affects 40,000–67,000 patients/year in the US; 30-day mortality rate of 47%.
- 46% of subarachnoid hemorrhage survivors suffer permanent cognitive impairment.
- Nearly half of all ICH-related mortality occurs within the **first 24 hours** — making rapid diagnosis critical.
- CT of the head is the gold-standard diagnostic tool, but interpretation depends on how quickly the radiologist reads the study.
- **Outpatient ICH** is particularly at risk: initial symptoms may be vague, leading to a non-emergent "routine" order — meaning interpretation could be delayed by hours.

### Approach

**Goal:** Train a deep CNN to detect ICH in head CT studies, then deploy it prospectively to automatically reprioritise "routine" studies to "stat" on the radiology worklist.

**Dataset:**
- 46,583 non-contrast head CT studies (~2 million images) from 31,256 patients at Geisinger Health System
- Collected from 17 scanners across 4 manufacturers, from 2007–2017
- Labels derived from official clinical radiology reports (binary: ICH present / absent)
- Training set: 37,084 studies (26.8% positive for ICH); Test set: 9,499 studies

**Preprocessing:**
- Each study resampled to a uniform dimensionality of **24 × 256 × 256** voxels using cubic spline interpolation
- A **blood window** applied: window level = 40 HU, window width = 80 HU (maximises contrast for blood)
- Data augmentation for the training set: random translation (±20 px), rotation (±15°), horizontal mirroring

**Architecture:**
- Fully **3D CNN** (not slice-by-slice 2D): the entire CT volume is classified as a single input
- 5 convolutional layers + 2 fully connected layers + max pooling + local response normalisation
- Ensemble of 4 networks trained independently; final prediction = thresholded average of ensemble outputs
- Trained with stochastic gradient descent until near-zero training loss (overtraining), then selected parameters at peak cross-validation AUC

### Results

**Retrospective test set (9,499 studies):**

| Metric | Value |
|--------|-------|
| AUC (ROC) | 0.846 (95% CI: 0.837–0.856) |
| Operating point chosen | FPR = 0.20 |
| Sensitivity at operating point | 0.730 (95% CI: 0.713–0.748) |
| Specificity at operating point | 0.800 (95% CI: 0.790–0.809) |
| Average inference time | **2.3 seconds per study** |

**Prospective clinical implementation (Jan–Mar 2017, 347 routine studies):**

| Metric | Value |
|--------|-------|
| Studies upgraded to "stat" | 94 / 347 (27%) |
| Of those, confirmed ICH by radiologist | 60 / 94 (PPV = **64%**) |
| New ICH cases detected from outpatients | **5** (1.4% of all routine studies) |
| Median time to diagnosis — routine (control) | **512 min** |
| Median time to diagnosis — reprioritised "stat" | **19 min** |
| Reduction in time to diagnosis | **96%** (p < 0.0001) |
| False positives reviewed by neuroradiologist | 4 out of 34 had **probable ICH missed** by original radiologist |

### Clinical Impact — Case Vignettes

**Case 1:** 88-year-old female on coumadin (anticoagulant) presented with 1 week of mental status changes. Outpatient routine head CT ordered. Algorithm flagged ICH → reprioritised to stat. Interpreted in **39 minutes** → acute intracerebral hemorrhage in the left temporal lobe confirmed. Anticoagulation reversed immediately with prothrombin complex concentrate and vitamin K. Follow-up CT showed stable haematoma; near-complete resolution at 1 month.

**Case 2:** 76-year-old male post-fall, treated non-operatively, returned for outpatient follow-up with dizziness and headaches. Routine head CT ordered. Algorithm flagged ICH → read in **8 minutes** → acute/subacute left subdural haematoma confirmed. Eventually required emergency evacuation 12 days later.

### Discussion & Limitations

**Strengths:**
- First published deployment of a deep learning algorithm into a live clinical radiology workflow
- Demonstrated real-world operational benefit (96% faster diagnosis) not just retrospective AUC
- Algorithm operated blind (no a priori location hint, no scanner control)
- Found 4 probable ICH cases missed by original radiologist among false positives — suggesting AI as safety net

**Limitations:**
- Labels derived from radiology reports, not gold-standard re-read → unknown label noise
- Algorithm outputs a study-level binary label (no localisation) — radiologist cannot see where the ICH is
- 3D architecture constrained by GPU memory → images downsampled to 24 × 256 × 256
- Patient outcomes beyond time-to-diagnosis not formally evaluated (requires multi-year prospective RCT)
- Localisation tools (R-CNN, Grad-CAM) were 2D-only at the time; not directly applicable to 3D volumes

### Mapping to CDSS Concepts

| CDSS concept | How the paper addresses it |
|---|---|
| **CDSS pattern** | Triage / prioritisation (non-interpretive AI) |
| **Autonomy level** | Workflow automation — no per-case radiologist review before reprioritisation |
| **Threshold selection** | FPR = 0.20 chosen as operating point (policy decision) |
| **Workflow integration** | Active reprioritisation; radiologists not told the system was running |
| **Evaluation** | Clinical/operational impact: time-to-diagnosis (the metric that matches the action) |
| **Silent mode** | Not formally used, but retrospective evaluation on held-out data served a similar purpose |
| **Monitoring / drift** | Not discussed in this early paper — an acknowledged limitation |

---

*Notes compiled from: 03_MEDIMG_decision_support.pdf and 2018_triage_intracranial_hemmorrhage.pdf — HSLU Medical Image Analysis, Dr. Ludovic Amruthalingam*