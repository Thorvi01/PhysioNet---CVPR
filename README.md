# digitization-of-ECG-images

Team: Aleksei Anisimov, Dishaa Bornare, Cenling Gao

## Overview

---

Electrocardiograms (ECGs) have been the standard of cardiac diagnostic monitoring for more than a century. While modern hospitals rely on digital systems whereby the raw signal data is stored (XML or DICOM), a vast majority of the world's clinical cardiac history (and many current practices in resource limited environments) still rely on paper based recordings.

These physical documents, however, are subject to degradation, are hard to share for remote consultation, and are, most importantly, incompatible with modern diagnostic tools based on artificial intelligence. This project uses advanced computer vision to close this digital divide, turning static, "trapped" images into dynamic, actionable medical data.



## Problem Statements

---

The main challenge in this project is the inherent loss of data fidelity that goes along with flattening a continuous electrical signal into a physical printout. With the medical data that needs to be reclaimed from these images, there are three complex computer vision problems to solve:

1. Geometric and Environmental Degradation:

Unlike digital waveforms, paper ECGs are subject to physical "noise." Capturing these documents through mobile cameras or legacy scanners adds perspective skew and non-linear warping (e.g., from curved paper or folds) causing mathematical distortion of the relationship between the x-axis (time) and y-axis (voltage). Additionally, there are environmental artifacts such as shadows, ink bleed-through, and physical stains that cause "false signals" that may mislead standard detection algorithms.

2. Low-Contrast Signal-Grid Interference: ECG waveforms are usually superimposed on a 1mm x 1mm reference grid. In many of the degraded samples it is difficult to tell the difference between the visual intensity of the grid lines and the cardiac signal itself. Standard filtering of the image will often suffer from this, cutting the grid background will often lead to the inadvertent "erasure" of the QRS complex, those critical, high frequency spikes of the heartbeat that are important for a proper diagnosis.
3. Lead Overlap and Morphological Complexity:

In a standard 12-lead layout, spatial density is high. Waveforms from adjacent leads are often intersecting or overlapping. Traditional heuristic-based image processing is unable to "track" a signal once it crosses another; it does not have the contextual intelligence to know which path is which lead. This requires a more robust, deep learning strategy that can be used to perform semantic separation and thus preserve the integrity between each individual cardiac channel.



## Dataset

https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data

The images in this dataset are derived from the PTB-XL dataset, which contains 21,799 clinical 12-lead ECG signals from 18,889 patients.

The dataset is designed to resemble the "messiness" of real clinical data: Volume: 21800+ training records with 10-second 12-lead ECG signal

Diversity of artifacts on the data:

* Scans: High resolution flatbed high color and grayscale scans.

* Photos: Photos taken with mobile phones under different lighting conditions (presence of shadows, glare). Physical Damage Synthetic mold, stains, water damage, and creases.

* Digital Noise: Background text, redacted hospital headers and various grid opacities.

* Layouts: Standard 12 lead layouts (usually 3 rows x 4 columns), including a long "Rhythm Strip" (Lead II) at the bottom.



## Methods

---

The goal of this project is to digitize ECG waveforms from scanned ECG images by extracting the waveform traces and reconstructing them as digital signals. Our approach follows the general pipeline of the second solution from the PhysioNet ECG Image Digitization competition, but replaces the original convolutional segmentation network with a Transformer-based model. The main motivation for this modification is that Transformer architectures are able to capture long-range spatial dependencies through self-attention mechanisms, which may improve the detection of continuous ECG traces across large image regions.

The overall pipeline consists of three major stages. First, the ECG images are preprocessed to normalize their resolution, remove noise, and convert them into standardized grayscale images suitable for model training. During this stage, segmentation masks indicating the location of ECG waveform pixels are also generated. Second, a Transformer-based segmentation model is trained to identify the ECG waveform regions in the image. Instead of using a convolutional encoder–decoder architecture such as U-Net, the model converts the input image into a sequence of patches and processes them using a Transformer encoder. This allows the model to learn global relationships between different parts of the ECG trace. Finally, the predicted segmentation masks are used in the next stage of the pipeline to extract the waveform coordinates and reconstruct the ECG signal as a time-series representation.

This approach aims to improve segmentation accuracy for ECG waveforms that extend across the entire image width, while maintaining a modular pipeline that separates preprocessing, segmentation, and signal reconstruction.



## Tools

---

The implementation of the proposed approach is primarily based on the Python programming language and the PyTorch deep learning framework. Additional libraries are used to support different stages of the pipeline.

NumPy is used for efficient numerical computation and for loading the preprocessed ECG image and mask data stored in .npy format.

The Torchvision library provides additional neural network utilities and model components that simplify the implementation of image-based deep learning models.

Matplotlib is used for visualization purposes, such as inspecting segmentation predictions and analyzing model outputs during development.

The experiments are executed in a Jupyter Notebook environment, which allows different stages of the pipeline to be implemented in sequential code blocks and makes it easier to document the workflow, visualize intermediate results, and reproduce the training process

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```



## Usage

---

```bash
python main.py
```

All intermediate results saved to `outputs/` folder



## Pipeline

---

| Step | File                    | What it does                                     |
| ---- | ----------------------- | ------------------------------------------------ |
| 1    | `grid_scale.py`         | Measure grid spacing (px/mm) — grid is our ruler |
| 2    | `remove_grid.py`        | Remove red grid (R-G thresholding)               |
| 3    | `remove_text.py`        | Remove text labels (connected components)        |
| 4    | `find_rows.py`          | Find 4 lead rows (horizontal projection)         |
| 5    | `find_pulse.py`         | Find calibration pulse → baseline per row        |
| 6    | `remove_artifacts.py`   | Remove pulse + separator bars (shape detection)  |
| 7    | `find_leads.py`         | Split into 13 leads (white gap detection)        |
| 8    | `trace.py`              | Trace signal (weighted center of mass) → mV      |
| 9    | `export_and_compare.py` | Compare with ground truth                        |
| 10    | `transformer_ecg_segmentation.py` | Do segmentation with transformer model                        |


## Collaboration

---

Aleksei Anisimov: evaluation, implement the original solution

Dishaa Bornare：data preprocessing

Cenling Gao : transformer-based segmentation model



## REFERENCES

---

[1] M. A. Reyna et al., "Digitization and Classification of ECG Images: The George B. Moody PhysioNet Challenge 2024," Computing in Cardiology, vol. 51, pp. 1–4, 2024.

[2] K. K. Shivashankara et al., "ECG-Image-Kit: A synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization," Physiological Measurement, vol. 45, p. 055019, 2024.

[3] T. Someya, "2nd Place Solution — PhysioNet Digitization of ECG Images," Kaggle Competition Writeup, Jan. 2026. [Online]. Available: https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/2nd-place-solution

[4] F. Krones, B. Walker, T. Lyons, and A. Mahdi, "Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts," Computing in Cardiology, vol. 51, 2024.

[5] H.-C. Yoon et al., "Segmentation-based Extraction of Key Components from ECG Images: A Framework for Precise Classification and Digitization," Computing in Cardiology, vol. 51, 2024.
