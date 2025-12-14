# MSI_PROJECT – Image Dataset Augmentation for Waste Classification

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
│
├── *.py                        # Python scripts for the project
├── *.pkl                       # Trained ML models serialized
│
└── venv/                       # (Optional) Python virtual environment
```
---

# 📘 Project Documentation: Image Classification Pipeline

This project implements a complete Machine Learning pipeline for image classification. It consists of tree main stages: **Data Augmentation** to increase dataset diversity and **Feature Extraction** to transform raw images into numerical vectors suitable for training ML models (SVM/KNN), **Model Training** to be used in a real-world application with live camera feed.

## 📂 File 1: `data_augmentation.py`

### 1. Overview & Purpose
**Goal:** To artificially expand the size of the dataset and introduce variance to prevent the machine learning model from overfitting.
**Logic:**
The script iterates through every image in the `images/` directory. For every single original image, it generates **3 additional variations** using safe geometric and photometric transformations. This ensures the model learns to recognize objects even if they are flipped, slightly rotated, or viewed under different lighting conditions.

### 2. Workflow
1.  **Validation:** Checks if input folders exist and prepares a matching output structure in `augmented/`.
2.  **Processing Loop:** Iterates through every class folder and every image file.
3.  **Resize:** Instantly resizes the image to `128x128` to ensure consistency before any processing.
4.  **Transformation:** Passes the image to `augment_image()` which returns a list containing:
    * Horizontal Flip
    * Random Rotation
    * Brightness Adjustment
5.  **Saving:** Writes both the original (resized) and the 3 new variants to the disk.

### 3. Key Functions & Logic

#### `augment_image(img)`
* **Purpose:** The core engine that generates variations of a single input image.
* **Logic:**
    1.  **Flip:** Uses `cv2.flip` with code `1` (horizontal). *Why?* Objects like "metal" or "glass" are vertically symmetrical; a flipped bottle is still a bottle.
    2.  **Rotate:** Calculates a rotation matrix for a random angle between -15° and +15°. *Why?* Small rotations simulate camera tilt. We avoid large rotations (e.g., 90°) because some classes might rely on orientation, and to not distort the image too much.
    3.  **Brightness:** Converts the image from BGR to **HSV** color space. It scales the **V (Value)** channel by a factor of 0.8 to 1.2, then converts back. *Why?* Changing brightness in RGB space often washes out colors. Changing the V channel preserves the *color* (Chrominance) while only affecting lightness.

#### `augment_dataset()`
* **Purpose:** Manages file I/O and progress tracking.
* **Logic:** It mirrors the folder structure of the input directory into the output directory to ensure the labels (folder names) remain correct for the next stage.

### 4. Variables & Parameters

| Variable/Parameter | Value | Explanation & Reasoning |
| :--- | :--- | :--- |
| `IMG_SIZE` | `(128, 128)` | **Standardization.** High enough resolution to see textures (like rust on metal), but small enough to keep processing speed fast. |
| `ROTATION_RANGE` | `(-15, 15)` | **Safety.** Limits rotation to ±15 degrees. Large rotations (e.g., 180°) might introduce black borders or distort the object context too much. |
| `brightness_factor` | `0.8 - 1.2` | **Lighting Variance.** Randomly darkens image by 20% or brightens by 20% to simulate different times of day/lighting conditions. |
| `cv2.INTER_LINEAR` | (OpenCV Flag) | **Interpolation.** Used during rotation. Linear interpolation is a balance between speed and quality (avoids jagged edges). |
| `cv2.BORDER_REFLECT` | (OpenCV Flag) | **Edge Handling.** When rotating, empty space appears at corners. This fills that space by mirroring the image content, preventing black artifacts that could confuse the ML model. |

---

## 📂 File 2: `feature_extraction.py`

### 1. Overview & Purpose
**Goal:** To convert raw images (pixels) into a meaningful set of numbers (feature vector) that describes the content.
**Logic:**
Raw pixels are poor inputs for standard classifiers. This script extracts specific "descriptors" focusing on three main properties: **Shape** (HOG), **Texture** (LBP, GLCM), and **Color** (LAB stats). It also uses **Bag of Visual Words (BoVW)** to recognize local features. Finally, it uses **PCA** to reduce the data size while keeping the most important information.

### 2. Workflow
1.  **Data Loading:** Reads all images from `augmented/` and assigns numeric labels based on folder names.
2.  **BoVW Training:**
    * Extracts ORB keypoints from all images.
    * Uses **K-Means Clustering** to find 256 common "visual patterns."
    * Saves this "vocabulary."
3.  **Feature Extraction:** Loops through images again, extracting HOG, LBP, GLCM, LAB, and the BoVW histogram for every single image.
4.  **Post-Processing:**
    * **Scaling:** Standardizes data (mean=0, variance=1).
    * **PCA:** Compresses the feature vector to the top 200 components.
5.  **Artifact Saving:** Saves the features (`X.npy`), labels (`y.npy`), and the fitted models (Scaler, PCA, BoVW) for later use.

### 3. Key Functions & Logic

#### `extract_hog(gray)` (Histogram of Oriented Gradients)
* **What it does:** Detects object shapes and edges.
* **Library:** `skimage.feature.hog`
* **Parameters:**
    * `orientations=9`: The code looks for edges in 9 specific directions (0°, 20°, 40°...). *Why?* 9 is the industry standard for capturing curvature.
    * `pixels_per_cell=(8, 8)`: Checks 8x8 pixel blocks at a time. *Why?* Captures small details without being too noisy.

#### `extract_lbp(gray)` (Local Binary Patterns)
* **What it does:** Detects fine surface textures (rough vs smooth).
* **Library:** `skimage.feature.local_binary_pattern`
* **Parameters:**
    * `P=8, R=1`: Compares a pixel to its 8 immediate neighbors (radius 1).
    * `method='uniform'`: **Crucial.** This merges similar patterns (like a rotated edge) into one. It reduces the output histogram size to just 10 bins, making the feature vector efficient and rotation-invariant.

#### `extract_glcm(gray)` (Gray-Level Co-occurrence Matrix)
* **What it does:** Calculates statistical texture properties.
* **Logic:** It looks at how often a pixel of brightness *X* is right next to a pixel of brightness *Y*. From this, it calculates:
    * **Contrast:** Intensity of edges.
    * **Correlation:** Linear dependency of gray levels.
    * **Energy:** Orderliness of the texture.
    * **Homogeneity:** How close the distribution is to a diagonal (smoothness).

#### `build_bovw(images)` & `extract_features(...)`
* **What it does:** Implements the Bag of Visual Words model.
* **Logic:**
    1.  **ORB (Oriented FAST and Rotated BRIEF):** Detects "keypoints" (corners/blobs). We extract 500 per image.
    2.  **K-Means:** We throw *all* keypoints from *all* images into a pile and sort them into `BOVW_K=256` clusters. Each cluster center is a "Visual Word" (e.g., "a sharp corner", "a white circle").
    3.  **Histogram:** For every image, we count how many times each "Visual Word" appears.

### 4. Variables & Parameters

| Variable/Parameter | Value | Explanation & Reasoning |
| :--- | :--- | :--- |
| `BOVW_K` | `256` | **Vocabulary Size.** Defines how many "visual words" we look for. 256 is a power of 2, providing enough detail to distinguish objects without creating a massive, sparse vector. |
| `IMG_SIZE` | `(128, 128)` | **Consistency.** Must match the augmentation output so pixel-based extractors (like HOG) work consistently. |
| `StandardScaler()` | `Default` | **Normalization.** SVM and KNN calculate "distances" between data points. If one feature is 0-1 (LBP) and another is 0-255 (Color), the larger one dominates. Scaling makes all features contribute equally. |
| `PCA(n_components=150)` | `150` | **Dimensionality Reduction.** The raw feature vector is huge (~2000+ numbers). PCA condenses this to the 200 most mathematically significant directions. This removes noise and speeds up training significantly. |
| `MiniBatchKMeans` | `batch_size=1000` | **Optimization.** Standard K-Means is slow on large datasets. MiniBatch processes data in chunks of 1000, which is much faster with very little loss in accuracy. |

### 5. Why this Pipeline?
This specific combination of features was chosen to maximize accuracy for general object classification:
* **HOG** handles the **Shape** (e.g., is it a bottle or a can?).
* **LBP/GLCM** handles the **Texture** (e.g., is it smooth glass or crinkled trash?).
* **LAB Color** handles the **Material** (e.g., is it brown cardboard or transparent plastic?).
* **BoVW** handles **Local Features** (e.g., specific logos or text on the object).

