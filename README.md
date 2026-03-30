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

## Results

| Approach | SNR (dB) |
|----------|----------|
| 2nd place reproduced (their weights, 6 models) | 23.38 |
| Our UNet (ResNet18, per-row, cleaned images) | 16.43 |

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

### Step 3: Can We Improve Post-Processing?

Before training new models, we checked whether simply changing the post-processing could improve results. Tested 4 resampling and 4 filtering methods on 5 samples using their pre-trained models:

| Method | SNR (dB) |
|--------|----------|
| scipy.signal.resample (baseline) | 32.77 |
| torch interpolate linear | 30.37 |
| butterworth lowpass | 24.36 |
| savitzky-golay | 9.86 |

Conclusion: the baseline resampling is already optimal. Filtering removes real signal content. Post-processing is not where the improvement lies.

### Step 4: Train Our Own Segmentation

Since preprocessing (Stage 0+1) works well, we kept it and focused on replacing Stage 2 with our own models.

**Why we cannot replicate their full training**: their 6 models require 100+ GPU hours (Kaggle limit: 30h/week), and the full dataset (977 × 9 variants = 22 GB) exceeds Kaggle's 20 GB output limit.

We explored two alternative segmentation approaches:

#### Approach A: Transformer Segmentation

Replaced the convolutional segmentation with a Transformer-based model. Instead of a UNet encoder-decoder, the input image is split into patches and processed with a Transformer encoder to capture long-range spatial dependencies. The motivation is that ECG traces span the full image width, so global context through self-attention may help. File: `transformer_ecg_segmentation.py`.

#### Approach B: Simple UNet on Cleaned Images

The idea: if we remove visual noise (grid, text) before feeding the image to the model, a simpler architecture might suffice.

**Data preparation** (`01_data_preparation.py`):
- Rectified all 977 training images through Stage 0+1
- Wrote our own mask generation script (missing from someya's repo), verified round-trip error = 0.000000 mV
- Fixed a bug: Lead II in the CSV has 10 seconds of data, but Row 1 on the image shows only 2.5 seconds — must truncate before mask creation
- Key finding: even clean digital images (variant 0001) require rectification. Without Stage 0+1, the mask drifts from the signal.   
[Prepared training data for training (0001-only for now)]() — 977 rectified images + masks + fold CSV

**Training** (`02_train_model.py`):
- Cleaned images: removed pink grid via color filter (gray < 120 AND R-G < 10)
- Per-row crops (480×5600) instead of whole image — the whole-image approach gave ~0 dB SNR
- ResNet18 UNet, 20 epochs, 977 images (variant 0001 only)
- Result: **16.43 dB** mean SNR on validation   
 [Our trained weights]() — ResNet18 UNet, 16.43 dB

The 6.7 dB gap to the 2nd place single model (23.10 dB) is likely due to: smaller encoder (14M vs 43-66M parameters), fewer epochs (20 vs 50), less training data (1 variant vs 9), and no cross-lead fusion.

---

## Usage

```bash
# Classical CV pipeline
python main.py
```

```bash
# Kaggle notebooks (run in order with datasets attached):
# 01_data_preparation.py # save output as dataset
# 02_train_model.py      # train and evaluate
# inference_baseline.py  # reproduce 2nd place submission
```

## Collaboration

Aleksei Anisimov: evaluation of 2nd place solution, reproduction, data preparation, training experiments, SNR analysis

Dishaa Bornare: data preparation, post-processing

Cenling Gao: segmentation model, transformer implementation

## References

[1] M. A. Reyna et al., "Digitization and Classification of ECG Images: The George B. Moody PhysioNet Challenge 2024," Computing in Cardiology, vol. 51, pp. 1–4, 2024. [PDF](https://moody-challenge.physionet.org/2024/papers/cinc_paper.pdf)

[2] K. K. Shivashankara et al., "ECG-Image-Kit: A Synthetic Image Generation Toolbox to Facilitate Deep Learning-Based Electrocardiogram Digitization," Physiological Measurement, vol. 45, p. 055019, 2024. [arXiv:2307.01946](https://arxiv.org/abs/2307.01946)

[3] M. A. Reyna et al., "ECG-Image-Database: A Dataset of ECG Images with Real-World Imaging and Scanning Artifacts; A Foundation for Computerized ECG Image Digitization and Analysis," arXiv preprint arXiv:2409.16612, 2024. [arXiv:2409.16612](https://arxiv.org/abs/2409.16612)

[4] PhysioNet, "George B. Moody PhysioNet Challenge 2024: Digitization and Classification of ECG Images," 2024. [Challenge page](https://physionet.org/news/post/challenge-2024/)

[5] T. Someya, "2nd Place Solution — PhysioNet Digitization of ECG Images," Kaggle Competition Writeup, Jan. 2026. [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/discussion)

[6] F. Krones, B. Walker, T. Lyons, and A. Mahdi, "Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts," Computing in Cardiology, vol. 51, 2024.

[7] H.-C. Yoon et al., "Segmentation-based Extraction of Key Components from ECG Images: A Framework for Precise Classification and Digitization," Computing in Cardiology, vol. 51, 2024.
