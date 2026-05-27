# Pixel-Level Biomedical Image Forgery Detection
**Private Score: 0.550 oF1 &nbsp;|&nbsp; Public Score: 0.450 oF1**

---

## What This Project Does

Scientific publications sometimes contain manipulated biomedical figures — western blots with duplicated bands, microscopy images with copy-pasted cell regions, gel panels reused across experiments. This project detects and **localises** those manipulations at the **pixel level**, outputting a binary mask over the forged region.

The dataset consists of 3,000+ biomedical images from published papers, each annotated with a ground-truth forgery mask (or labelled authentic). The task is treated as **pixel-level binary segmentation**: predict exactly which pixels belong to a manipulated region.

---

## Pipeline Architecture

Rather than applying a single end-to-end model to raw images, this solution decomposes the problem into a **structured multi-stage pipeline** tailored to the structure of scientific figures:

```
            Input Image
            │
            ▼
[Stage 1]  YOLO Panel Detector (dual-model ensemble)
           Localises sub-panels: Blots, Microscopy, Graphs, Flow Cytometry ...
            │
            ▼
[Stage 2]  Contrastive Embedding Similarity Screening (SupCon)
           Finds candidate duplicate panel pairs via cosine similarity
            │
            ▼
[Stage 3]  LightGlue Geometric Verification (SIFT / ALIKED)
           Keypoint matching + MAGSAC homography → pixel-level copy-move masks
            │
            ▼
[Stage 4]  Strip Detector (YOLOv8-Seg + embeddings)
           Detects reused horizontal strips within western blot panels
            │
            ▼
[Fallback] DINOv2 Encoder + U-Net-style Decoder (Segmentation model)
           Pixel-level semantic segmentation for non-copy-move forgeries
            │
            ▼
Output: RLE-encoded binary forgery mask  OR  "authentic"
```

---

## Why This Approach?

### 1. Panel Detection First — Not Raw Image Search

Searching for duplicate regions across an entire scientific figure would generate enormous numbers of false-positive candidate pairs (background regions, figure borders, white space). By first detecting semantically meaningful **panels** (blots, microscopy crops, graphs), we restrict the search space to regions that could plausibly be forged and that share the same imaging modality.

A dual-model YOLO ensemble (640 px + 960 px input resolution) improves recall on small or narrow blot panels that a single-resolution detector misses.

### 2. Two-Stage Verification: Embedding → Geometry

A naive approach would run expensive geometric matching on all panel pairs. Instead, we use a fast **embedding screening pass** first:

- **SupCon-trained embedders** (trained with supervised contrastive loss) produce compact 128-d representations where visually similar panels cluster in embedding space.
- Only pairs that exceed a cosine similarity threshold proceed to geometric verification.
- This reduces the number of LightGlue calls by ~10-20×, making full-dataset inference feasible.

LightGlue with MAGSAC homography estimation then provides robust geometric verification — it is invariant to brightness/contrast changes, mild rotation, and scale differences, all of which are common in copy-move forgeries.

### 3. Domain-Specific Fine-Tuning of Feature Extractors

Pre-trained natural-image keypoint detectors (SIFT, SuperPoint, ALIKED) are biased toward edges and corners that are common in photographs but not in western blot images. The ALIKED extractor and LightGlue matcher used for blot panels are **fine-tuned end-to-end** on blot-specific training pairs for 25,000 iterations, redirecting keypoint attention to band boundaries and gel-lane texture — the features that actually carry forgery signal in blot images.

### 4. DINOv2 as a Generalising Fallback

Not all forgeries are copy-moves. Some manipulations involve brightness adjustment, contrast enhancement, or splicing from entirely different images — patterns that keypoint matching cannot detect. A **DINOv2-Base encoder with a lightweight U-Net-style CNN decoder** serves as a fallback segmentation model when the geometric pipeline finds nothing.

**Why DINOv2 over EfficientNet/ResNet encoders?**

| Encoder | Pretraining | Patch-level semantics | Generalisation to biomedical images |
|---------|-------------|----------------------|--------------------------------------|
| ResNet-50 | ImageNet supervised | Weak (global features dominate) | Moderate |
| EfficientNet-B4 | ImageNet supervised | Moderate | Moderate |
| **DINOv2-Base** | **Self-supervised (DINO v2)** | **Strong (patch = local region)** | **Strong** |

DINOv2's self-supervised objective forces the model to learn meaningful patch-level representations without class labels, which transfers surprisingly well to biomedical texture analysis. The encoder is **frozen** — only the lightweight decoder (~1.5 M parameters) is trained on the ~3,000 training images, avoiding overfitting.

The decoder uses progressive bilinear upsampling across 3 convolutional blocks (768→384→192→96→1), closely following the U-Net decoder philosophy of channel reduction with spatial resolution recovery.

### 5. Adaptive Post-processing

Raw probability maps from the segmentor are sharpened using a **gradient-enhanced adaptive threshold**:

```
prob_enhanced = 0.55 * prob + 0.45 * sobel_gradient_norm
threshold = mean(prob_enhanced) + 0.3 * std(prob_enhanced)
```

This combines the model's confidence with the *sharpness* of probability transitions, yielding tighter boundaries around forged regions without requiring a fixed threshold tuned to a specific probability range.

---

## Model Components

| Component | Architecture | Input Size | Purpose |
|-----------|-------------|-----------|---------|
| YOLO Panel Detector (primary) | YOLOv12 | 640 px | Detect all panel types |
| YOLO Panel Detector (secondary) | YOLOv12 | 960 px | Enrich blot panel recall |
| Blot Overlap Embedder | SupCon encoder | 320×64 | Find partially overlapping blot pairs |
| Blot Duplication Embedder | SupCon encoder | 320×64 | Find fully duplicated blot panels |
| Microscopy Embedder | SupCon ensemble (3 ckpts) | 224×224 | Find duplicate microscopy panels |
| LightGlue (Microscopy) | SIFT + LightGlue | variable | Geometric verification for microscopy |
| LightGlue (Blots) | ALIKED (fine-tuned) + LightGlue | variable | Geometric verification for blots |
| Strip Detector | YOLOv8-Seg + SupCon embedder | 128×128 | Detect reused blot strips |
| Fallback Segmentor | DINOv2-Base + U-Net decoder | 518×518 | Pixel-level segmentation fallback |

---

## Results

| Split | oF1 Score |
|-------|-----------|
| Public leaderboard | 0.450 |
| **Private leaderboard** | **0.550** |

The evaluation metric is **object-level F1 (oF1)**: a predicted mask is counted as a true positive only if its IoU with the ground-truth mask exceeds 0.1. This rewards region-level detection accuracy over pixel-perfect boundary tracing.

---

## Repository Structure

```
├── forgery_detection_1st_place.ipynb   # Fully documented inference notebook
├── README.md                           # This file
```

**Note:** Model weights and dataset files are not included in this repository due to size constraints. Checkpoint paths in the notebook follow the original input directory structure and should be updated to match your local paths.

---

## Technical Stack

- **PyTorch** — model inference and tensor operations
- **Ultralytics** — YOLOv8/v11/v12 panel detection and segmentation
- **LightGlue / LightGlue** — sparse keypoint matching
- **HuggingFace Transformers** — DINOv2 encoder loading
- **OpenCV** — morphological post-processing and image operations
- **Albumentations** — inference-time transforms

---

## Key Takeaways for Practitioners

1. **Decompose before you detect.** Scientific figures have rich spatial structure; exploiting it (panel detection) dramatically reduces search space and false positives.

2. **Cascade cheap-to-expensive verification.** Embedding cosine similarity is O(n²) but fast; geometric matching is expensive — always screen first.

3. **Fine-tune domain-specific components.** Generic keypoint detectors trained on natural images underperform on biomedical textures. Even lightweight fine-tuning on domain data yields significant gains.

4. **Use self-supervised encoders as frozen feature extractors when labelled data is scarce.** DINOv2 patch features provide strong generalisation on ~3,000 training images where supervised encoders would overfit.

5. **Adaptive thresholding outperforms fixed thresholds** when the score distribution varies across image types (blots vs. microscopy vs. fluorescence).
