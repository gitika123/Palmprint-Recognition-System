# Palmprint Recognition with CNN & Scattering Networks

A hybrid deep-learning system for palmprint identification that combines Convolutional Neural Networks (CNN) with Scattering2D features to achieve high accuracy and robustness.

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Dependencies](#dependencies)  
- [Installation](#installation)  


## Project Overview

Palmprint recognition is a biometric approach that leverages the unique texture patterns on an individual's palm. In this project, we implement three models:

1. **CNN-based classifier**  
2. **Scattering2D-based classifier**  
3. **FusionNet**: combines CNN + Scattering features  

Our hybrid **FusionNet** achieves **99.38% Top-1 accuracy** on the BMPD Palmprint dataset.

## Features

- **Data loading & preprocessing**:  
  - ROI extraction via thresholding & contour detection  
  - RGB & grayscale tensors generation
- **Model definitions**:  
  - `PalmprintCNN`, `ScatClassifier`, `FusionNet` 
- **Training & evaluation**:  
  - Unified `train_model` function, Top-k accuracy, confusion matrices 
- **End-to-end pipeline**:  
  - `run_palmprint_pipeline.py` orchestrates dataset splits, model loops, saving plots 

## Dependencies

- Python 3.7+  
- PyTorch  
- torchvision  
- kymatio  
- OpenCV (`cv2`)  
- NumPy  
- scikit-learn  
- Pillow (`PIL`)  
- matplotlib  
- seaborn  
- tqdm  

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/palmprint-recognition.git
   cd palmprint-recognition
