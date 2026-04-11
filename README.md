# Digitization of ECG Images

Team: Aleksei Anisimov, Dishaa Bornare, Cenling Gao

---

## Overview

Electrocardiograms (ECGs) have been the standard of cardiac diagnostic monitoring for more than a century. While modern hospitals store raw signal data digitally (XML or DICOM), a vast majority of the world's clinical cardiac history still exists only as paper recordings.

These physical documents are subject to degradation, difficult to share for remote consultation, and incompatible with modern AI-based diagnostic tools. This project uses computer vision to turn static ECG images into digital time-series data.

Competition: [PhysioNet ECG Image Digitization](https://www.kaggle.com/competitions/physionet-ecg-image-digitization)

## Dataset

https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data

The images are derived from the PTB-XL dataset (21,799 clinical 12-lead ECG signals from 18,889 patients, 10-second recordings). The dataset includes diverse real-world artifacts:

- **Scans**: high-resolution flatbed scans in color and grayscale
- **Photos**: mobile phone captures under varying lighting (shadows, glare)
- **Physical damage**: synthetic mold, stains, water damage, creases
- **Digital noise**: background text, redacted headers, varying grid opacities
- **Layout**: standard 12-lead (3 rows × 4 columns) with Lead II rhythm strip at the bottom

---

## Research Steps

### Step 1: Classical CV Pipeline

Before using any deep learning, we built a heuristic pipeline to understand ECG image structure:

| Step | File | Purpose |
|------|------|---------|
| 1 | `grid_scale.py` | Measure grid spacing (px/mm) |
| 2 | `remove_grid.py` | Remove red grid (R-G thresholding) |
| 3 | `remove_text.py` | Remove text labels (connected components) |
| 4 | `find_rows.py` | Find 4 lead rows (horizontal projection) |
| 5 | `find_pulse.py` | Find calibration pulse → baseline per row |
| 6 | `remove_artifacts.py` | Remove pulse + separator bars |
| 7 | `find_leads.py` | Split into 13 leads (white gap detection) |
| 8 | `trace.py` | Trace signal (weighted center of mass) → mV |
| 9 | `export_and_compare.py` | Compare with ground truth |

This was not intended as a final solution — it helped us understand the image structure, what noise looks like, and where the challenges are.

```bash
# Classical CV pipeline
python main.py
```

### Step 2: Reproduce the 2nd Place Solution (23.38 dB)

To establish a strong baseline, we reproduced the 2nd place competition solution using their published code and weights. File: `inference_baseline.py`.

We assembled the pipeline from two sources:
- **Stage 0+1** (preprocessing): hengck23's keypoint detection + grid rectification
- **Stage 2** (segmentation): someya's 6-model ensemble

The pipeline works in 3 stages:
- **Stage 0**: ResNet18 UNet detects 9 keypoints + image orientation. Homography warps the image flat. Uses 4× TTA.
- **Stage 1**: ResNet34 UNet detects grid lines (44 horizontal + 57 vertical). Rectifies image to a standard 1700×2200 layout.
- **Stage 2**: Two UNet types predict sparse masks (y-position of the ECG trace at each pixel column):
  - *Whole model*: input 1280×5600, EfficientNet B7/V2-L encoder + CoordConv decoder
  - *Lead model*: input 4×480×5600 (per-row crops), EfficientNet B6/V2-L encoder + cross-lead fusion
  - *Ensemble*: 2 whole + 4 lead models, predictions averaged

Submitted and confirmed: **23.27 public / 23.38 private LB**.

**Data & Weights**

- [hengck23 Stage 0+1 (Kaggle)](https://www.kaggle.com/datasets/hengck23/hengck23-demo-submit-physionet) — preprocessing models
- [2nd place Stage 2 weights (Kaggle)](https://www.kaggle.com/datasets/takashisomeya/physionet-final-submission-models) — 6-model ensemble

```bash
# inference_baseline.py  # reproduce 2nd place submission
```

### Step 3: Can We Improve Post-Processing?

Before training new models, we checked whether simply changing the post-processing could improve results. Tested 4 resampling and 4 filtering methods on 5 samples using their pre-trained models:

| Method | SNR (dB) |
|--------|----------|
| scipy.signal.resample (baseline) | 32.77 |
| torch interpolate linear | 30.37 |
| butterworth lowpass | 24.36 |
| savitzky-golay | 9.86 |

Conclusion: the baseline resampling is already optimal. Filtering removes real signal content. Post-processing is not where the improvement lies.

### Step 4: Our Pipeline

We explored two alternative segmentation approaches:

### Approach A: Transformer-based model (Cenling)

Replaced the convolutional segmentation with a Transformer-based model. Instead of a UNet encoder-decoder, the input image is split into patches and processed with a Transformer encoder to capture long-range spatial dependencies. The motivation is that ECG traces span the full image width, so global context through self-attention may help.

Files: 
* ecg-transformer-based.py: notebook1, training
* ecg-transformer-based-submission.py: notebook2, generate submission.csv

1. Training — Transformer-based Segmentation

Unlike the UNet-based approach used by other team members, this approach replaces the segmentation backbone with a Vision Transformer (ViT)-style model.

#### Model Design

The model consists of three main components:

* Patch Embedding:
The input image (resized to 512×512) is divided into non-overlapping patches (16×16), each projected into a 256-dimensional embedding space.
* Transformer Encoder:
A stack of 4 Transformer encoder layers (multi-head self-attention) processes the sequence of patch embeddings, enabling global context modeling.
* Decoder:
A transposed convolution layer upsamples the Transformer output back to spatial resolution to produce a segmentation mask.

This design allows the model to capture long-range dependencies, which is theoretically beneficial for ECG signals spanning horizontally across the image.

#### Training Setup

* Dataset: 977 rectified ECG images (clean variant only)
* Input size: 512 × 512
* Loss function: Binary Cross Entropy (BCEWithLogitsLoss)
* Optimizer: Adam (lr = 1e-4)
* Batch size: 2
* Epochs: 3

#### Training Results
* Samples: 977
* Epoch 1, Loss: 0.0799
* Epoch 2, Loss: 0.0075
* Epoch 3, Loss: 0.0051

#### Analysis of Training Behavior

The training loss decreases rapidly:

From 0.0799 → 0.0051 in just 3 epochs
Indicates the model quickly fits the segmentation task.

This suggests:

* The model successfully learns pixel-level reconstruction of ECG traces
* The dataset (clean images only) is relatively easy for segmentation
* Transformer has sufficient capacity for this task

However, this also hints at possible overfitting, since training data is limited (977 samples)
No noisy or degraded image variants are included
No validation metric was tracked.

2. Inference and Submission Pipeline

For submission, the trained Transformer model was used to generate segmentation masks on the test set.

However, instead of reconstructing ECG waveforms from the predicted masks, a dummy signal was used:

signal = np.zeros(num_rows)

This was done to:

* Validate the end-to-end submission pipeline
* Ensure correct formatting of submission.csv

#### Submission Results
   Public Score: 0.09360
   Private Score: 0.10574

#### Analysis of Results

(1) Missing Signal Reconstruction Step

The competition evaluates numerical ECG signals, not segmentation masks. As a result:
The predicted masks are completely ignored
Submission contains no meaningful ECG information. 
This is the primary reason for the low score.

(2) Lack of Generalization to Noisy Data

The test set includes: Scanned images， Mobile photos and Damaged ECG prints.
Transformer models generally require larger datasets and strong augmentation.
Without exposure to noisy data, the model likely:
Overfits clean patterns
Fails on real-world distortions

#### Future Improvements

To make this approach competitive, the following steps are required:

(1) Integrate Signal Reconstruction (Critical)

(2) Train on Multi-Variant Data

(3) Add Data Augmentation

(4) Use Signal-aware Loss

(5) Increase Model Capacity and Training Time

### Approach B: UNet (Aleksei)

#### 1. Preprocessing (Stage 0 + Stage 1) — from hengck23 baseline
I kept the existing preprocessing pipeline which converts messy ECG images into a standardized format:
- Stage 0: Detects keypoints and orientation using a ResNet18 UNet, then applies homography to normalize the image.
- Stage 1: Detects grid lines (44 horizontal, 57 vertical) using a ResNet34 UNet, then warps the image to a canonical 1700×2200 rectified output.

After rectification, every image has the same geometry — signal baselines at known y-positions, signal region in columns 301–5300.

Storage Limitation Workaround
The full dataset (977 images × 9 variants = ~60 GB) exceeds Kaggle's 20 GB output limit. I solved this by preprocessing in 5 separate batches:

| Batch | Segments | Description | Size |
|-------|----------|-------------|------|
| [batch1](https://www.kaggle.com/datasets/tylerde/ecg-training-data-0001-only) | 0001 | Clean digital renders | 6.1 GB |
| [batch2](https://www.kaggle.com/datasets/tylerde/ecg-training-data-0003-0005-0006) | 0003, 0005, 0006 | Color scan, phone photo, screen photo | 20.5 GB |
| [batch3](https://www.kaggle.com/datasets/tylerde/ecg-training-data-0004-0009) | 0004, 0009 | BW scan, stained/soaked prints | ~14 GB |
| [batch4](https://www.kaggle.com/datasets/tylerde/ecg-training-data-0010-0011) | 0010, 0011 | Damaged prints, moldy color scans | ~14 GB |
| [batch5](https://www.kaggle.com/datasets/tylerde/ecg-training-data-0012) | 0012 | Moldy BW scans | ~7 GB |

Each batch is a self-contained dataset with identical structure: rectified images, sparse masks, ECG CSVs, and fold assignments.

I wrote my own mask generation that converts ground-truth CSV values (millivolts) to pixel positions using the formula:
```
y_pixel = baseline - mV × 79.0
```
Sub-pixel accuracy via floor/ceil weighting, stored in sparse COO format. Verified with overlay visualizations — round-trip error is negligible.

- [preprocessing script](https://www.kaggle.com/code/tylerde/ecg1-preprocess)   
- [overlay verification and visualization script](https://www.kaggle.com/code/tylerde/ecg1-verify-overlay-all-batches)

#### 2. Training (Stage 2) — Evolution of My Approach
Trainig code: https://github.com/Anisimov-AA/digitization-of-ECG-images   
Inference code: https://www.kaggle.com/code/tylerde/ecg3-submit-model-0001

Training Strategy:   
Stage A: Train on 977 clean images → model learns signal patterns   
Stage B: Fine-tune on 200×9 mixed types → model learns noise robustness   
Stage C: Full dataset (977×9) → maximum data coverage   
Stage D: Upgrade backbone → more capacity if needed   
   
**Phase 1 — Simple UNet Baseline (16.43 dB)**

First attempt: ResNet18 UNet with binary segmentation.
- Cleaned images by removing pink grid via color filter
- Per-row crops (480×5600) instead of whole image
- BCE loss, 20 epochs, 977 images (variant 0001 only)
- Result: **16.43 dB** on validation
- Gap to top solutions: smaller encoder, fewer epochs, single variant, no regression head

```bash
# Kaggle notebooks (run in order with datasets attached):
# 01_data_preparation.py # save output as dataset
# 02_train_model.py      # train and evaluate
```

**Phase 2 — Soft-Argmax + JSD Loss**

Applied key techniques:
- **Row crops (480×5000):** crop each of 4 signal rows separately, centered on known baselines. Reduces input size 4x and simplifies the task — model only finds one trace per crop.
- **Soft-argmax head:** instead of binary mask → argmax, predicts probability distribution over vertical axis, computes expected y-position. Differentiable, sub-pixel accurate.
- **JSD + SNR combined loss:** JSD (Jensen-Shannon Divergence) teaches WHERE the signal is in pixel space. SNR loss directly optimizes the competition metric in millivolt space.
- **CoordConv decoder:** injects y/x coordinates at every decoder level so model knows its spatial position.
- Kept grid lines in the image (no cleaning) — top solutions found the grid carries useful calibration information.
- ResNet34 backbone, 30 epochs, 977 clean images (0001 only)
- Result: **23.33 dB** on validation — surpasses all top solutions' single model scores

**First Submission — 3.0 dB on Leaderboard**   

Despite 23.33 dB on clean validation data, the model scored only 3.0 dB on the real test set. Investigation revealed the cause:
```
0001 (clean):        22-30 dB ✓
0003 (color scan):    7.2 dB  ✗
0005 (phone photo): -12.1 dB  ✗
0006 (screen photo): -5.1 dB  ✗
```
The model had never seen noisy/degraded images and couldn't generalize.

**Phase 3 — Curriculum Fine-tuning**   

To teach the model to handle all image types without massive GPU costs, I created a compact training dataset:
- 200 random samples from each of the 9 image types = 1800 images
- Fine-tune from best Phase 2 checkpoint (23.33 dB), 30 epochs
- Lower learning rate (5e-5 vs 1e-4) to preserve learned features
- This is curriculum learning: first learn the core task on clean data, then adapt to noise

**Validation: 17.91 dB** (on all 9 image types)   
**Submission: 17.0 dB** (Private: 16.95, Public: 17.13)   

**Phase 4 — Post-processing experiments (no retraining)**
- Resampling fix (split-then-resample each lead): **Submission: 15.5 dB** — worse, model trained with old method
- Einthoven's Law correction (II=I+III): **Submission: 15.5 dB** — worse, model not accurate enough for physics corrections
- Conclusion: at 17 dB level, post-processing hurts. All gains must come from training.

**Phase 5 — Grayscale + Augmentations + Adaptive Sigma (in progress)**   
Same 200×9 dataset, same Phase 2 checkpoint, but with training improvements:
- Grayscale input: convert to gray, copy to 3 channels. Removes color as noise source — grid colors vary wildly across image types (pink, gray, absent) but signal is always dark on light.
- Augmentations: random gamma (0.8–1.2), Gaussian blur (k=3,5), noise (σ=2–8), contrast shift. Each applied with 50% probability. Simulates degradation without needing more data.
- Adaptive sigma: wider Gaussian target on sharp peaks where signal changes rapidly. From 3rd place solution — makes sharp peaks easier to learn.

**Validation: TBD**   
**Submission: TBD**   

## Collaboration

Aleksei Anisimov: evaluation of 2nd place solution, reproduction, data preparation, training experiments, SNR analysis

Dishaa Bornare: data preparation, post-processing

Cenling Gao: segmentation model, transformer approach implementation, training on kaggle, generate submission.csv, report about transformer part

## References

[1] M. A. Reyna et al., "Digitization and Classification of ECG Images: The George B. Moody PhysioNet Challenge 2024," Computing in Cardiology, vol. 51, pp. 1–4, 2024. [PDF](https://moody-challenge.physionet.org/2024/papers/cinc_paper.pdf)

[2] K. K. Shivashankara et al., "ECG-Image-Kit: A Synthetic Image Generation Toolbox to Facilitate Deep Learning-Based Electrocardiogram Digitization," Physiological Measurement, vol. 45, p. 055019, 2024. [arXiv:2307.01946](https://arxiv.org/abs/2307.01946)

[3] M. A. Reyna et al., "ECG-Image-Database: A Dataset of ECG Images with Real-World Imaging and Scanning Artifacts; A Foundation for Computerized ECG Image Digitization and Analysis," arXiv preprint arXiv:2409.16612, 2024. [arXiv:2409.16612](https://arxiv.org/abs/2409.16612)

[4] PhysioNet, "George B. Moody PhysioNet Challenge 2024: Digitization and Classification of ECG Images," 2024. [Challenge page](https://physionet.org/news/post/challenge-2024/)

[5] T. Someya, "2nd Place Solution — PhysioNet Digitization of ECG Images," Kaggle Competition Writeup, Jan. 2026. [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/discussion)

[6] F. Krones, B. Walker, T. Lyons, and A. Mahdi, "Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts," Computing in Cardiology, vol. 51, 2024.

[7] H.-C. Yoon et al., "Segmentation-based Extraction of Key Components from ECG Images: A Framework for Precise Classification and Digitization," Computing in Cardiology, vol. 51, 2024.
