# MSI_PROJECT – Deep Learning for Waste Classification

---

## 📁 Project Structure

```
MSI_PROJECT/
│
├── images/                     # Original (raw) dataset
│   ├── cardboard/
│   ├── glass/    
│   ├── metal/    
│   ├── paper/    
│   ├── plastic/  
│   └── trash/    
│
├── augmented/                  # Automatically generated augmented images
│   ├── cardboard/
│   ├── glass/    
│   ├── metal/    
│   ├── paper/    
│   ├── plastic/  
│   └── trash/    
│
├── extracted_features/         # Extracted features (numpy arrays)     
│   ├── X.npy                   # PCA-reduced features
│   ├── X_raw.npy               # Raw CNN embeddings
│   ├── y.npy                   # Labels
│   ├── scaler.pkl              # StandardScaler model
│   ├── pca.pkl                 # PCA model
│   └── class_map.txt           # Class index mapping
│
├── *.py                        # Python scripts for the project
│
└── venv/                       # (Optional) Python virtual environment
```

---


## 📘 Executive Summary

This project implements a **complete ML pipeline for automated waste classification** using deep learning–based feature extraction and classical classifiers.
It achieves **near-perfect accuracy** by combining:

* **Data augmentation** to expand limited datasets
* **MobileNetV2 feature extraction** via transfer learning
* **KNN and SVM classifiers** for efficient and accurate prediction
* **Real-time inference** using a webcam feed

The system is optimized for **CPU-only deployment** while maintaining high accuracy and practical inference speed.

### Key Performance Metrics

| Metric            | Value                    |
| ----------------- | ------------------------ |
| Original Dataset  | 1,960 images (6 classes) |
| Augmented Dataset | 7,460 images (3.8×)      |
| Raw Feature Size  | 1,286 dimensions         |
| PCA Feature Size  | 200 dimensions           |
| Variance Retained | 75.92%                   |
| KNN Accuracy      | 99.73%                   |
| SVM Accuracy      | 99.87%                   |
| Inference Speed   | 5–15 FPS (CPU)           |

---

## 📂 File 1: `data_augmentation.py`

### Overview and Purpose

This module solves the **limited data problem** by expanding the dataset from 1,960 to 7,460 images.
Augmentation increases visual diversity while preserving class labels, improving generalization and reducing overfitting.

### Workflow and Logic

The pipeline follows four clear steps:

1. Validate input directories and recreate class folder structure
2. Iterate over each image per class
3. Resize images to **224×224** (MobileNetV2 requirement)
4. Generate **three safe augmentations** per image

Each original image produces **four total samples** (original + 3 variants).

### Core Functions and Implementation

* **Horizontal Flip**
  Preserves object semantics since waste items are orientation-invariant.

* **Rotation (±15°)**
  Simulates camera tilt while avoiding distortions and border artifacts.

* **Brightness Adjustment (HSV space)**
  Scales the Value channel (0.8–1.2) to mimic lighting variation without altering color fidelity.

The `augment_dataset()` function manages directory traversal, augmentation execution, and output saving while preserving labels.

### Configuration Parameters

* **Image Size:** 224×224 (MobileNetV2 standard)
* **Rotation Range:** ±15° (empirically safe)
* **Brightness Factor:** 0.8–1.2 (realistic lighting simulation)
* **Interpolation:** Linear (quality vs speed balance)
* **Border Mode:** Reflect (avoids black artifacts)

### Dataset Statistics

After augmentation:

* Cardboard: 988
* Glass: 1,540
* Metal: 1,260
* Paper: 1,796
* Plastic: 1,452
* Trash: 424

This maintains class proportions while significantly increasing sample count.

---

## 📂 File 2: `feature_extraction_mobilenet.py`

### Overview and Purpose

This module replaces handcrafted vision features with **deep learning embeddings** using **MobileNetV2 pretrained on ImageNet**.
Transfer learning enables strong feature extraction with limited data and CPU-only hardware.

### Workflow and Logic

The pipeline consists of six stages:

1. Load augmented images and assign numeric labels
2. Initialize MobileNetV2 without the classification head
3. Preprocess images (RGB, resize, normalization)
4. Extract **1,280-D CNN embeddings**
5. Append **LAB color statistics (6-D)**
6. Apply **StandardScaler + PCA (200-D)**

Final features balance expressiveness, compactness, and efficiency.

### Architecture and Design Decisions

* **MobileNetV2** chosen for efficiency via depthwise separable convolutions, plus it is small and fast when trained using CPU only 
* **Global Average Pooling** produces fixed-length embeddings
* **Pretrained ImageNet weights** provide robust visual primitives
* **Frozen backbone** avoids overfitting and reduces computation

### Core Functions and Implementation

* `build_mobilenet()`
  Loads MobileNetV2 with `include_top=False`, `pooling="avg"`, and ImageNet weights.

* `lab_stats()`
  Extracts mean and standard deviation from LAB color channels to encode color distribution.

* `load_images_from_dir()`
  Loads images and assigns consistent class indices.

* `extract_and_save()`
  Handles batching, preprocessing, feature concatenation, scaling, PCA reduction, and saving.

### Configuration Parameters and Rationale

* **Image Size:** 224×224 (network compatibility)
* **Batch Size:** 16 (memory–performance tradeoff)
* **PCA Components:** 200 (75.92% variance retained)
* **Color Features:** Enabled (improves class separability)
* **Random State:** Fixed (reproducibility)

### Transfer Learning Rationale

Training CNNs from scratch is impractical due to:

* Small dataset size
* High computational cost
* Risk of overfitting


---

## 📂 File 3: `knn_training.py`

### Overview and Purpose

This module implements **K-Nearest Neighbors**, a non-parametric classifier that predicts labels based on local feature similarity.
It emphasizes interpretability and strong performance in high-quality embedding spaces.

### Workflow and Logic

1. Load PCA-reduced features and labels
2. Load class name mapping
3. Perform **80/20 stratified split**
4. Train or load KNN model
5. Evaluate accuracy and per-class performance
6. Generate comprehensive visualizations

### Stratified Splitting Rationale

Stratification preserves class proportions in both training and testing sets, ensuring fair evaluation—especially for minority classes like trash.

### Hyperparameter Configuration

* **K = 7**
  Balances noise sensitivity and locality.

* **Weights = distance**
  Closer neighbors have higher influence.

* **Metric = cosine**
  Robust in high-dimensional embedding spaces.

* **Algorithm = auto**
  Enables efficient tree-based search when applicable.

values were chosen throw trial and error

### Visualization Components

Generated plots include:

* Class distribution
* 2D and 3D PCA projections
* Train/test split visualization
* Confusion matrix
* Correct vs incorrect prediction mapping

These provide both quantitative and intuitive performance insight.

---

## 📂 File 4: `svm_training.py`

### Overview and Purpose

This module trains a **Support Vector Machine** to learn a global decision boundary with maximum margin, providing strong generalization.

### Algorithm Fundamentals

SVM identifies **support vectors** and constructs a hyperplane that maximizes separation.
For non-linear data, kernel functions map features into higher-dimensional spaces.

### Hyperparameter Configuration

* **Kernel:** RBF
  Captures non-linear class boundaries effectively.

* **C = 10**
  Balances margin smoothness and training error.

* **Gamma = scale**
  Automatically adapts to feature variance.

* **Probability = True**
  Enables calibrated confidence scores via Platt scaling.

values were chosen throw trial and error

### Performance Analysis

* Accuracy: **99.87%**, slightly higher than KNN
* More robust to noise due to global decision function
* Both models benefit strongly from MobileNetV2 embeddings

---

## 📂 File 5: `input.py`

### Overview and Purpose

This module enables **real-time waste classification** using a webcam, ensuring full consistency with the training pipeline.

### Workflow and Logic

Each frame undergoes:

1. RGB conversion
2. Resize to 224×224
3. MobileNetV2 feature extraction
4. LAB color statistics extraction
5. Feature scaling
6. PCA reduction
7. KNN & SVM probability prediction
8. Confidence thresholding (≥60%)
9. On-frame annotation and display

### Feature Extraction Consistency

* Identical preprocessing steps as training
* Same feature order and concatenation
* Prevents feature distribution shift

### Transform vs Fit-Transform

* **Training:** `fit_transform()`
* **Inference:** `transform()` only

This preserves learned scaling and PCA projections and avoids invalid transformations.

### Confidence Thresholding

* Threshold = 60%
* Low-confidence predictions labeled as **“unknown”**
* Prevents misleading outputs for ambiguous inputs

### Dual Model Display

* Displays both KNN and SVM predictions

---

## 🎯 Complete Pipeline Summary

### Data Flow Architecture

1. Load original images
2. Apply data augmentation
3. Extract CNN + color features
4. Scale and reduce dimensions (PCA)
5. Train classifiers
6. Perform real-time inference

Each stage feeds cleanly into the next with serialized artifacts.

### Critical Design Decisions

* Shift from handcrafted to learned features
* Use of transfer learning for efficiency
* MobileNetV2 for CPU feasibility
* Explicit color encoding (LAB)
* PCA for noise reduction and efficiency
* Dual classifier comparison (KNN vs SVM)

### Performance Analysis

* Extremely low error rates (2–4 misclassifications per test set)
* All classes exceed 98% accuracy
* 5–15 FPS sufficient for interactive use

### Practical Applications

* Recycling education tools
* Smart waste bins
* Automated sorting assistance
* Edge-device deployment without GPUs

---

## 📊 Performance Summary

### Dataset Statistics

| Class     | Original | Augmented | Increase % |
| --------- | -------- | --------- | ---------- |
| Cardboard | 259      | 988       | **281.5%** |
| Glass     | 401      | 1,540     | **284.0%** |
| Metal     | 328      | 1,260     | **284.1%** |
| Paper     | 476      | 1,796     | **277.3%** |
| Plastic   | 386      | 1,452     | **276.2%** |
| Trash     | 110      | 424       | **285.5%** |
| **Total**     | **1,960**       | **7,460**       | ***280.6%*** |



### Model Comparison

| Model                | Accuracy |
| -------------------- | -------- |
| Traditional CV + KNN | 86.86%   |
| Traditional CV + SVM | 88.54%   |
| MobileNetV2 + KNN    | 99.73%   |
| MobileNetV2 + SVM    | 99.87%   |

---

## 📝 Conclusion

This project demonstrates how **transfer learning + careful pipeline design** can deliver production-level accuracy using modest hardware.
The combination of data augmentation, MobileNetV2 feature extraction, PCA compression, and classical classifiers results in a **robust, efficient, and deployable waste classification system** suitable for real-world applications.

--- 
