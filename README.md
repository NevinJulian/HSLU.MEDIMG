# 🏥 Medical Image Analysis — HSLU

**Course:** Medical Image Analysis  
**Institution:** Hochschule Luzern (HSLU) — Applied AI Research Lab (AAI), Informatik  
**Lecturer:** Dr. Ludovic Amruthalingam ([ludovic.amruthalingam@hslu.ch](mailto:ludovic.amruthalingam@hslu.ch))

---

## 📖 About This Course

Medical Image Analysis covers the intersection of machine learning and clinical imaging. The course spans the full pipeline — from understanding how different imaging modalities work and what clinical problems they address, to designing and evaluating ML models for tasks like classification, segmentation, object detection, and image enhancement.

A core theme throughout the course is the gap between research performance and real-world clinical deployment: data challenges, regulatory constraints, workflow integration, and fairness are treated as first-class concerns alongside model architecture and metrics.

---

## 🗂️ Repository Structure

```
medical-image-analysis/
│
├── README.md                              ← You are here
│
├── lecture_notebooks/                     ← Jupyter notebooks used during lectures
│   └── (notebooks added per week)
│
└── notes/
    ├── week_1_introduction.md            ← Notes: Introduction & Overview
    ├── week_2_preprocessing.md           ← Notes: Preprocessing and Data Handling
    └── week_3_decision_support.md        ← Notes: Clinical Decision Support Systems
```

---

## 📅 Weekly Overview

| Week | Topic | Notes | Key Paper |
|------|-------|-------|-----------|
| 01 | Introduction: Modalities, Clinical Challenges, AI in Medical Imaging | [📝 Notes](notes/week_1_introduction.md) | — |
| 02 | Preprocessing and Data Handling | [📝 Notes](notes/week_2_preprocessing.md) | — |
| 03 | Clinical Decision Support Systems | [📝 Notes](notes/week_3_decision_support.md) | Arbabshirani et al. (2018) — ICH triage |

---

## 🧠 Topics Covered

The course is structured around the following themes:

- **Medical Imaging Modalities** — X-ray, CT, MRI, Ultrasound, PET/SPECT, and others (dermoscopy, histology, fundoscopy, etc.)
- **Clinical Challenges** — workforce shortages, cognitive load, inter-rater variability, rare/subtle cases
- **The Promise of AI** — augmented intelligence, consistency, quantification, decision support
- **ML Tasks for Medical Imaging** — classification, segmentation, object detection, image enhancement
- **Real-World ML Projects** — case studies including teledermatology (PASSION project) and hand eczema severity assessment
- **Challenges & Pitfalls** — data bias, distribution shift, DICOM formats, regulatory and workflow constraints
- **Preprocessing & Data Handling** — PACS, DICOM/NIfTI formats, windowing, cropping, patching, resampling, registration, data quality, artefacts, shortcut learning
- **Clinical Decision Support Systems (CDSS)** — AI in medical imaging workflows, CDSS design patterns, integration, evaluation, monitoring, data drift
- **Reading Research Papers** — how to find, evaluate, and read ML papers effectively

---

## 📄 Key Papers

| Paper | Topic | Week |
|-------|-------|------|
| Arbabshirani et al. (2018). *Advanced machine learning in action: identification of intracranial hemorrhage on CT scans with clinical workflow integration.* NPJ Digital Medicine. | ICH detection & worklist triage — end-to-end clinical deployment case study | 03 |
| Sutton et al. (2020). *An overview of clinical decision support systems: benefits, risks, and strategies for success.* NPJ Digital Medicine. | CDSS overview | 03 |
| Tejani et al. (2024). *Integrating and adopting AI in the radiology workflow.* Radiology. | AI workflow integration & IHE standards | 03 |
| Bernstein et al. (2023). *Can incorrect AI results impact radiologists?* European Radiology. | Human behaviour effects of CDSS | 03 |
| Kwong et al. (2022). *The silent trial — the bridge between bench-to-bedside clinical AI applications.* Frontiers in Digital Health. | Silent trials methodology | 03 |

---

## 🛠️ Tools & Resources

**Platforms for finding papers:**
- [PapersWithCode](https://paperswithcode.com)
- [arxiv-sanity](http://www.arxiv-sanity.com)
- [Google Scholar](https://scholar.google.com)

**Leading venues in medical image analysis:**
- MICCAI, NeurIPS, ICML, ICLR, JMLR

**Note-taking tools recommended in course:**
- [Logseq](https://logseq.com), [Notion](https://notion.so)

---

## 📌 Notes

- Lecture notebooks are added to `lecture_notebooks/` as the course progresses.
- Weekly notes in `notes/` aim to capture all key concepts from slides and papers — they are meant as a study companion, not a replacement for attending lectures.