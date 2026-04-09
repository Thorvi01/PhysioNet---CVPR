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

#### Approach A: Transformer-based model (Cenling)

Replaced the convolutional segmentation with a Transformer-based model. Instead of a UNet encoder-decoder, the input image is split into patches and processed with a Transformer encoder to capture long-range spatial dependencies. The motivation is that ECG traces span the full image width, so global context through self-attention may help. File: `transformer_ecg_segmentation.py`.

#### Approach B: UNet (Aleksei)

##### 1. Preprocessing (Stage 0 + Stage 1) — from hengck23 baseline
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

**All data was preprocessed and verified using two Kaggle notebooks: one for preprocessing and another for verification.**

**In the verification notebook, three random samples were selected from each of the nine image variants. The signal masks were converted into pixel space and overlaid on the images. The results show a clear visual alignment, indicating that the masks correctly match the image signals.**

[preprocessing script](https://www.kaggle.com/code/tylerde/ecg1-preprocess)
[overlay verification and visualization script](https://www.kaggle.com/code/tylerde/ecg1-verify-overlay-all-batches)



---

## Usage

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
