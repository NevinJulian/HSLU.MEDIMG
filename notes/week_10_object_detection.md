# Week 10 — Object Detection

**Course:** Medical Image Analysis — HSLU  
**Lecturer:** Dr. Simone Lionetti  
**Topics:** Task taxonomy · Benchmark datasets · Detection metrics · Region-proposal architectures · Single-shot detectors · Transformer-based detection

---

## Table of Contents

1. [What is Object Detection?](#1-what-is-object-detection)
   - [Task Taxonomy](#11-task-taxonomy)
   - [Relationships Between Tasks](#12-relationships-between-tasks)
2. [Benchmark Datasets](#2-benchmark-datasets)
   - [PASCAL VOC](#21-pascal-voc)
   - [Microsoft COCO](#22-microsoft-coco)
   - [Open Images](#23-open-images)
3. [Metrics](#3-metrics)
   - [Confusion Matrix for Detection](#31-confusion-matrix-for-detection)
   - [Precision and Recall](#32-precision-and-recall)
   - [Intersection over Union (IoU)](#33-intersection-over-union-iou)
   - [Average Precision and Precision–Recall Analysis](#34-average-precision-and-precisionrecall-analysis)
   - [Mean Average Precision (mAP)](#35-mean-average-precision-map)
   - [Aggregation](#36-aggregation)
4. [Architectures — Region Proposals](#4-architectures--region-proposals)
   - [Traditional Object Detection](#41-traditional-object-detection)
   - [R-CNN](#42-r-cnn)
   - [Fast R-CNN](#43-fast-r-cnn)
   - [Faster R-CNN](#44-faster-r-cnn)
   - [Mask R-CNN](#45-mask-r-cnn)
   - [Summary of Region-Proposal Architectures](#46-summary-of-region-proposal-architectures)
5. [Architectures — Single-Shot Detection](#5-architectures--single-shot-detection)
   - [YOLO](#51-yolo-you-only-look-once)
   - [SSD](#52-ssd-single-shot-detector)
   - [RetinaNet and Focal Loss](#53-retinanet-and-focal-loss)
6. [More Recent Developments](#6-more-recent-developments)
   - [CornerNet and CenterNet](#61-cornernet-and-centernet)
   - [DETR — Detection Transformer](#62-detr--detection-transformer)
7. [Key Takeaways](#7-key-takeaways)
8. [Key Papers](#8-key-papers)

---

## 1. What is Object Detection?

### 1.1 Task Taxonomy

Object detection is one of several fundamental vision tasks. They differ in the question they answer, the output format, and the model architecture required:

| Task | Question | Output | Architecture |
|---|---|---|---|
| **Multi-class classification** | Is object class $c$ present in the image? | Single label $\hat{y} = f(\mathbf{x})$ | Single forward pass |
| **Multi-label classification** | Which classes are present? | Set of labels | Single forward pass |
| **Object detection** | Where are all instances of class $c$? | Set of labelled bounding boxes $(\hat{y}_i, \hat{b}_i) = f(\mathbf{x}_i)$ | Region proposals + classifier |
| **Semantic segmentation** | Is class $c$ present at pixel $[n,m]$? | Per-pixel label map $\hat{y}[n,m] = f(\mathbf{x})$ | Encoder–decoder |
| **Instance segmentation** | Is instance $i$ of class $c$ present at pixel $[n,m]$? | Per-pixel instance + class label | Region proposals + mask head |

Object detection thus sits between classification (no localisation) and segmentation (pixel-level localisation): it provides bounding boxes per instance with associated class labels and confidence scores.

### 1.2 Relationships Between Tasks

The five tasks are not independent — they can be derived from each other:

- **Instance segmentation → Object detection**: take the bounding box of each instance mask.
- **Instance segmentation → Semantic segmentation**: merge all masks of the same class.
- **Object detection → Multi-class classification**: report only the dominant class.
- **Semantic segmentation → Multi-label classification**: report which classes appear anywhere.

Understanding these relationships helps when selecting a task formulation: if instance-level masks are available, a detection head can be derived "for free", and vice versa.

---

## 2. Benchmark Datasets

### 2.1 PASCAL VOC

The **PASCAL Visual Object Classes (VOC)** challenge established early benchmarks for detection and segmentation.

- First release: 2005; landmark release: 2007.
- **20 classes** organised in a simple hierarchy (vehicles, animals, household objects, people).
- Annotations: bounding boxes stored as boundary coordinates in **XML** files.
- Scale: ~10'000 images, ~25'000 annotated objects.
- 2012 extension added ~7'000 pixel-level segmentation annotations.

PASCAL VOC established the conventions for bounding box annotation (corner coordinates) and the use of IoU-based evaluation that all later datasets follow.

*Reference: Everingham, Van Gool, et al. "The PASCAL Visual Object Classes (VOC) Challenge." IJCV, 2009.*

### 2.2 Microsoft COCO

**MS COCO (Common Objects in Context)** became the dominant benchmark for modern detectors.

- First release: 2014; challenges ran until 2020.
- **80 object categories** and 91 stuff categories.
- Annotations: bounding boxes specified by **centre coordinates and size** in **JSON** files.
- Scale: 328k images (200k labelled), 2.5M annotated objects.
- Additional annotations: segmentation masks, captions, and keypoints.

COCO is harder than VOC because objects appear in realistic, cluttered scenes and evaluation averages over multiple IoU thresholds (see Section 3.6).

*Reference: Lin, Maire, et al. "Microsoft COCO: Common Objects in Context." ECCV, 2014.*

### 2.3 Open Images

**Open Images** is the largest publicly available detection benchmark.

- First release: 2016; landmark V4 release with 2018 challenge.
- **600 classes** with a complex hierarchy; includes positive and negative image-level labels.
- Annotations: bounding boxes with boundary coordinates in **CSV** files.
- Scale: 1.9M images, 16M annotated objects.
- Additional annotations: segmentation masks and visual relationships between objects.

*Reference: Kuznetsova, Rom, et al. "The Open Images Dataset V4." IJCV, 2018.*

---

## 3. Metrics

### 3.1 Confusion Matrix for Detection

In object detection, each detected box is matched to a ground-truth box. The outcome of each detection is:

| Outcome | Meaning |
|---|---|
| **True Positive (TP)** | Detected box matches a ground-truth box (IoU above threshold, correct class) |
| **False Positive (FP)** | Detected box does not match any ground truth (spurious detection) |
| **False Negative (FN)** | Ground-truth object was not detected |
| **True Negative (TN)** | Background correctly not detected (rarely used in detection evaluation) |

> **Note:** True negatives are ill-defined in detection because the number of possible background regions is unbounded. Evaluation therefore focuses on precision and recall.

### 3.2 Precision and Recall

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$

- **Precision** — of all boxes the model predicted, what fraction were correct?
- **Recall** — of all ground-truth objects, what fraction did the model find?

There is an inherent trade-off: lowering the confidence threshold increases recall (more detections) but typically decreases precision (more false positives).

### 3.3 Intersection over Union (IoU)

IoU (also known as the **Jaccard Index**) measures how well a predicted bounding box $A$ overlaps with the ground-truth box $B$:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

- $\text{IoU} = 1$: perfect overlap.
- $\text{IoU} = 0$: no overlap at all.

IoU is used as the **matching criterion**: a detection is counted as a TP only if $\text{IoU} \geq \theta$ for some threshold $\theta$ (commonly 0.5 for VOC, averaged over 0.50–0.95 for COCO).

### 3.4 Average Precision and Precision–Recall Analysis

To summarise a detector's performance across confidence thresholds, the **Precision–Recall (PR) curve** is computed by sweeping the confidence threshold and plotting precision against recall.

**Average Precision (AP)** is the area under the PR curve:

$$\text{AP} = \sum_{i=1}^{n} P_i (R_i - R_{i-1})$$

where detections are sorted by decreasing confidence and $P_i$, $R_i$ are the precision and recall after the $i$-th detection.

Additional metrics for specific operating points:

- **$F_1$ score** — harmonic mean of precision and recall at a fixed threshold:

$$F_1 = \frac{2}{\frac{1}{\text{Precision}} + \frac{1}{\text{Recall}}}$$

- **FPR95** — False Positive Rate at 95% recall; useful when high recall is a clinical requirement.

### 3.5 Mean Average Precision (mAP)

**mAP** is the standard summary metric for object detection. The procedure for a single class and IoU threshold:

1. **Select** an IoU threshold $\theta$.
2. **Rank** all detections across all images in descending confidence order.
3. **Match greedily**: assign each detection to the highest-IoU ground-truth box (above $\theta$) that has not yet been matched.
4. Label matched detections as **TP**, unmatched detections as **FP**, unmatched ground truths as **FN**.
5. **Compute AP** from the resulting PR curve.

mAP then **averages AP over all classes** (the "m" in mAP). For COCO, AP is further averaged over IoU thresholds from 0.50 to 0.95 in steps of 0.05.

### 3.6 Aggregation

mAP aggregates along three axes:

- **Over samples**: all detections and ground truths across the dataset are pooled. Missing a single object in a dense scene is penalised equally to missing a prominent isolated one.
- **Over IoU thresholds**: COCO's standard metric averages over $\{0.50, 0.55, \ldots, 0.95\}$, rewarding detectors that localise tightly, not just loosely.
- **Over classes**: each class contributes equally to the final mAP regardless of frequency.

---

## 4. Architectures — Region Proposals

### 4.1 Traditional Object Detection

Before deep learning, detection was a three-stage pipeline:

| Stage | Methods |
|---|---|
| **Region Proposal** | Heuristics, superpixels — combinatorially expensive |
| **Feature Extraction** | Viola–Jones (Haar features), Histogram of Oriented Gradients (HOG), Deformable Part Models (DPM), feature pyramids |
| **Classification** | Decision Trees, Support-Vector Machines |

The key bottleneck was the region proposal stage: exhaustive sliding windows over scales and aspect ratios produce thousands of candidates, and each required independent feature extraction. Deep learning resolved this by sharing computation.

### 4.2 R-CNN

**R-CNN** (Regions with CNN features, Girshick et al., 2013) was the first architecture to combine selective search region proposals with CNN features:

1. **Input image** → extract ~2'000 region proposals using selective search.
2. **Warp** each proposal to a fixed size.
3. **CNN forward pass** per proposal → feature vector.
4. **Classify** each region with a class-specific SVM; refine bounding box with a separate regressor.

**Limitation:** the CNN runs once per proposal (~2'000 forward passes per image), making training and inference very slow.

*Reference: Girshick, Donahue, et al. "Rich feature hierarchies for accurate object detection and semantic segmentation." CVPR, 2013.*

### 4.3 Fast R-CNN

**Fast R-CNN** (Girshick, 2015) resolved R-CNN's speed bottleneck by running the CNN **once on the full image**:

1. The full image passes through a **deep ConvNet** → shared convolutional feature map.
2. Each region proposal is **projected** onto the feature map.
3. **RoI pooling** extracts a fixed-size feature vector per proposal from the shared feature map.
4. Two parallel **fully connected heads** predict: (a) softmax class probabilities and (b) bounding box offsets.

Because the convolutional computation is shared, Fast R-CNN is ~9× faster than R-CNN at test time. The remaining bottleneck is the external region proposal step (still selective search).

*Reference: Girshick. "Fast R-CNN." ICCV, 2015.*

### 4.4 Faster R-CNN

**Faster R-CNN** (Ren, He, et al., 2015) eliminated the external region proposal step by introducing a **Region Proposal Network (RPN)** that shares the same convolutional backbone:

- The RPN slides a small network over the shared feature map.
- At each spatial position it predicts $k$ **anchor boxes** of different scales and aspect ratios.
- For each anchor the RPN outputs: $2k$ **objectness scores** (object vs. background) and $4k$ **box offsets**.
- High-scoring proposals are passed to the RoI pooling + classification head.

The entire network (backbone + RPN + classifier) is trained **end-to-end**. Faster R-CNN achieves real-time-capable speeds (~5 fps at the time) while maintaining high accuracy.

*Reference: Ren, He, et al. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." NeurIPS, 2015.*

### 4.5 Mask R-CNN

**Mask R-CNN** (He, Gkioxari, et al., 2017) extends Faster R-CNN to instance segmentation by adding a **mask prediction head** that runs in parallel with the classification and box heads.

Key contribution: **RoIAlign** replaces RoI pooling. Standard RoI pooling uses integer coordinate quantisation when extracting features, introducing misalignment errors that are acceptable for bounding box regression but harmful for pixel-accurate mask prediction. RoIAlign uses bilinear interpolation to sample feature values at continuous sub-pixel coordinates, preserving spatial precision.

*Reference: He, Gkioxari, et al. "Mask R-CNN." ICCV, 2017.*

### 4.6 Summary of Region-Proposal Architectures

| Architecture | Region Proposals | Feature Sharing | Training |
|---|---|---|---|
| **R-CNN** | Selective search (external) | None — CNN per proposal | Independent stages |
| **Fast R-CNN** | Selective search (external) | Full image ConvNet + RoI pooling | Joint (classifier + bbox) |
| **Faster R-CNN** | RPN (internal, learned) | Full image ConvNet + RoI pooling | End-to-end |
| **Mask R-CNN** | RPN (internal, learned) | Full image ConvNet + RoIAlign | End-to-end + mask head |

---

## 5. Architectures — Single-Shot Detection

Region-proposal methods are accurate but inherently two-stage. **Single-shot detectors** eliminate the proposal stage entirely and predict all boxes and classes in a single forward pass.

### 5.1 YOLO: You Only Look Once

**YOLO** (Redmon, Divvala, et al., 2015) reformulates detection as a single regression problem:

1. Divide the input into an $S \times S$ grid.
2. Each grid cell predicts $B$ bounding boxes with confidence scores and $C$ class probabilities simultaneously.
3. A single CNN forward pass produces both the **class probability map** and the **bounding box coordinates and confidences**.
4. Non-maximum suppression (NMS) filters the raw predictions to final detections.

YOLO is extremely fast (real-time capable) because the entire detection pipeline is a single network evaluation. The trade-off is that each grid cell predicts only a small number of boxes, making dense or small-object detection harder.

*Reference: Redmon, Divvala, et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR, 2015.*

### 5.2 SSD: Single-Shot Detector

**SSD** (Liu, Anguelov, et al., 2016) addresses YOLO's difficulty with multi-scale objects by making predictions at **multiple feature map resolutions**:

- A VGG-based backbone produces feature maps at multiple scales (e.g., 8×8, 4×4, …).
- At each scale, default anchor boxes of several aspect ratios are placed at every spatial location.
- For each anchor the network predicts:
  - **Localisation offsets**: $\Delta(c_x, c_y, w, h)$ relative to the anchor.
  - **Class confidences**: $(c_1, c_2, \ldots, c_p)$ for all classes.
- Larger feature maps detect small objects; smaller feature maps detect large objects.

SSD combines the speed of YOLO with better performance on small objects through multi-scale feature pyramids.

*Reference: Liu, Anguelov, et al. "SSD: Single Shot MultiBox Detector." ECCV, 2016.*

### 5.3 RetinaNet and Focal Loss

Single-shot detectors face a severe **class imbalance** problem: a typical image contains thousands of background anchor boxes and only a handful of foreground (object) boxes. The gradient signal from easy background negatives dominates training and degrades accuracy.

**RetinaNet** (Lin, Goyal, et al., 2017) tackles this with two contributions:

**Feature Pyramid Network (FPN) backbone**: produces rich multi-scale features by combining bottom-up and top-down pathways with lateral connections, giving strong semantics at all scales.

**Focal Loss**: a modification of cross-entropy that down-weights easy examples:

$$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

- For **well-classified examples** ($p_t \to 1$): the factor $(1 - p_t)^\gamma$ approaches zero, strongly reducing their contribution.
- For **hard examples** ($p_t$ small): the factor is close to 1, retaining the full loss signal.
- $\gamma = 0$ recovers standard cross-entropy; $\gamma = 2$ is the recommended default.

This allows training on the full dense set of anchors without the need for hard-negative mining.

*Reference: Lin, Goyal, et al. "Focal Loss for Dense Object Detection." ICCV, 2017.*

---

## 6. More Recent Developments

### 6.1 CornerNet and CenterNet

**Anchor-free** detectors avoid the need to define anchor shapes and sizes manually.

**CornerNet** (Law, Deng, 2018) represents each bounding box as a pair of keypoints — top-left and bottom-right corners. A single ConvNet predicts heatmaps for corner locations and **embedding vectors** that allow matching each top-left corner to its corresponding bottom-right corner. This eliminates anchors entirely and removes the associated hyperparameters.

**CenterNet / Objects as Points** (Zhou, Wang, Krähenbühl, 2019) further simplifies the representation by modelling each object as a single **centre point** plus width and height offsets. The network predicts a heatmap of object centres; each peak corresponds to one detection. CenterNet extends naturally to 3D detection and pose estimation by adding additional output heads.

### 6.2 DETR — Detection Transformer

**DETR** (Carion, Massa, et al., 2020) reformulates detection as a **set prediction problem** and is the first fully end-to-end detector without anchors or NMS:

**Architecture:**
1. **Backbone** (CNN) extracts image features; positional encodings are added.
2. **Transformer encoder** processes the flattened feature map with self-attention.
3. **Transformer decoder** takes $N$ learned **object queries** and attends to the encoder output, producing $N$ output embeddings (one per query).
4. **FFN prediction heads** map each embedding to a class label and bounding box (or "no object").

**Training with bipartite matching loss:** predictions and ground-truth objects are matched one-to-one using the Hungarian algorithm. Unmatched predictions are supervised towards the "no object" class. This removes the need for NMS post-processing.

DETR achieves competitive accuracy with Faster R-CNN on COCO but has longer training times. It opened the door to transformer-based detection as a research direction (e.g., Deformable DETR, DINO).

*Reference: Carion, Massa, et al. "End-to-End Object Detection with Transformers." ECCV, 2020.*

---

## 7. Key Takeaways

- Object detection **jointly solves localisation and classification** — a fundamentally harder problem than image-level classification.
- The field evolved from slow, hand-crafted pipelines (Viola–Jones, HOG, DPM) to deep learning architectures in one decade.
- **Two-stage detectors** (R-CNN family) are generally more accurate; **single-shot detectors** (YOLO, SSD, RetinaNet) trade some accuracy for speed.
- **mAP** is the standard metric but requires careful interpretation: it aggregates over classes equally and is sensitive to the IoU threshold chosen.
- **Focal Loss** is the key ingredient enabling single-shot detectors to handle the extreme foreground/background class imbalance in dense anchor grids.
- **DETR** demonstrates that transformers can replace anchors, NMS, and region proposals entirely — at the cost of longer training and sensitivity to hyperparameters.
- In medical imaging, detection is used for tasks such as blood cell counting, lesion localisation, and organ/landmark detection, where the same architectural choices and metric pitfalls apply.

---

## 8. Key Papers

| Paper | Contribution |
|---|---|
| Everingham, Van Gool, et al. (2009). *The PASCAL Visual Object Classes (VOC) Challenge.* IJCV. | Foundational detection benchmark; established IoU-based evaluation |
| Lin, Maire, et al. (2014). *Microsoft COCO: Common Objects in Context.* ECCV. | Large-scale benchmark; introduced multi-threshold mAP |
| Kuznetsova, Rom, et al. (2018). *The Open Images Dataset V4.* IJCV. | Largest public detection dataset; complex class hierarchy |
| Girshick, Donahue, et al. (2013). *Rich feature hierarchies for accurate object detection and semantic segmentation.* CVPR. | R-CNN: first CNN-based detection pipeline |
| Girshick (2015). *Fast R-CNN.* ICCV. | Shared ConvNet + RoI pooling; end-to-end classifier training |
| Ren, He, et al. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS. | Integrated RPN; fully end-to-end two-stage detector |
| He, Gkioxari, et al. (2017). *Mask R-CNN.* ICCV. | RoIAlign + mask head; extends detection to instance segmentation |
| Redmon, Divvala, et al. (2015). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR. | First real-time single-shot detector |
| Liu, Anguelov, et al. (2016). *SSD: Single Shot MultiBox Detector.* ECCV. | Multi-scale anchor predictions; strong small-object performance |
| Lin, Goyal, et al. (2017). *Focal Loss for Dense Object Detection.* ICCV. | Focal Loss + FPN backbone; solves class imbalance in dense detectors |
| Law, Deng (2018). *CornerNet: Detecting Objects as Paired Keypoints.* ECCV. | Anchor-free detection via paired corner heatmaps |
| Zhou, Wang, Krähenbühl (2019). *Objects as Points.* arXiv. | CenterNet: anchor-free detection via centre-point heatmaps |
| Carion, Massa, et al. (2020). *End-to-End Object Detection with Transformers.* ECCV. | DETR: set prediction with bipartite matching; no anchors or NMS |

---

*Notes compiled from: 10_MEDIMG_object_detection.pdf — HSLU Medical Image Analysis, Dr. Simone Lionetti*
