# MSI_PROJECT – Image processing for Waste Classification

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
* **BoVW** handles **Local Features** (e.g., specific logos, sharp corners or text on the object).

---

## 📂 File 3: `knn_training.py`

### 1. Overview & Purpose
**Goal:** To train a K-Nearest Neighbors classifier on the extracted features and evaluate its performance on waste classification.
**Logic:**
KNN is a simple yet powerful algorithm that classifies samples based on the majority vote of their nearest neighbors in feature space. The script loads pre-processed features, splits the data strategically, trains the model, and generates comprehensive visualizations to understand model performance.

### 2. Workflow
1.  **Data Loading:** Reads the pre-processed features (`X.npy`), labels (`y.npy`), and class names from `extracted_features/`.
2.  **Stratified Split:** Divides data into 80% training and 20% testing while maintaining class proportions.
3.  **Model Training:** Either loads an existing model or trains a new KNN classifier with optimized hyperparameters.
4.  **Evaluation:** Predicts on test set and calculates overall and per-class accuracy.
5.  **Visualization:** Generates 6 comprehensive plots showing data distribution, PCA projections, confusion matrix, and predictions.

### 3. Key Functions & Logic

#### `read_data()`
* **Purpose:** Safely loads the numpy arrays containing features and labels.
* **Logic:** Uses try-catch to handle missing files gracefully and reports the shape of loaded data for verification.

#### `load_class_map()`
* **Purpose:** Reads human-readable class names from `class_map.txt`.
* **Logic:** Parses the file line-by-line to create a dictionary mapping numeric labels to class names (e.g., `0: cardboard`). Falls back to numeric labels if file is missing.

#### `StratifiedShuffleSplit`
* **What it does:** Ensures the 80/20 split maintains the same class distribution in both training and test sets.
* **Why it matters:** If you have 100 cardboard images, this guarantees exactly 80 go to training and 20 to testing. Regular random split could accidentally put 85 in training and 15 in testing, which would bias results.

#### `plot_visualizations()`
* **What it does:** Creates a 6-panel figure showing:
    1.  **Class Distribution:** Bar chart of sample counts per class
    2.  **PCA 2D - All Data:** Projects all features onto 2 dimensions to visualize separability
    3.  **PCA 3D:** 3D scatter plot showing feature space structure
    4.  **Train/Test Split:** Shows which samples were used for training (circles) vs testing (squares)
    5.  **Confusion Matrix:** Heatmap showing where the model makes mistakes
    6.  **Predictions:** Highlights correctly classified (green border) vs misclassified (red X) samples

### 4. Hyperparameters & Reasoning

#### KNN_PARAMS Configuration

| Parameter | Value | Explanation & Reasoning |
| :--- | :--- | :--- |
| `n_neighbors` | `7` | **Optimal K value.** Tested range: [3, 5, 7, 9, 11, 15, 21]. **Why 7?** Small K (3) is too sensitive to noise/outliers. Large K (15+) averages too many samples and loses local patterns. 7 provides the best balance, achieving **86.86% accuracy**. |
| `weights` | `'distance'` | **Inverse distance weighting.** Closer neighbors have more influence than distant ones. **Why?** A sample 0.1 units away is more relevant than one 5 units away. This prevents distant outliers from affecting classification. |
| `metric` | `'cosine'` | **Similarity measure.** Tested: euclidean, manhattan, cosine. **Why cosine?** Our features are histograms and frequency distributions (LBP, BoVW, GLCM). Cosine measures the *angle* between vectors, making it robust to magnitude differences. A bright image and dark image of the same object will have similar angles but different euclidean distances. |
| `algorithm` | `'auto'` | **Optimization.** Lets scikit-learn choose the fastest search algorithm (ball_tree, kd_tree, or brute force) based on data characteristics. |
| `n_jobs` | `-1` | **Parallelization.** Uses all available CPU cores to speed up distance calculations during prediction. |

### 5. Why These Metrics?

**Confusion Matrix:** Shows *where* the model fails. If metal is often confused with glass, this indicates those classes have similar features (e.g., both are shiny/smooth).

**Per-Class Accuracy:** Overall accuracy can be misleading if classes are imbalanced. If you have 1000 plastic samples but only 50 trash samples, 95% overall accuracy might hide that the model never correctly predicts trash.

**PCA Visualizations:** High-dimensional feature space (200 dimensions) is impossible to visualize. PCA projects it to 2D/3D while keeping the most important variance, letting us see if classes form distinct clusters.

---

## 📂 File 4: `svm_training.py`

### 1. Overview & Purpose
**Goal:** To train a Support Vector Machine classifier, which is typically more powerful than KNN for complex decision boundaries.
**Logic:**
SVM finds the optimal hyperplane that maximizes the margin between classes. Unlike KNN which memorizes training data, SVM learns a decision function. The RBF (Radial Basis Function) kernel allows it to create non-linear boundaries, making it ideal for image classification where class boundaries are rarely straight lines.

### 2. Workflow
The workflow is **identical** to `knn_training.py`:
1.  Load data and class map
2.  Stratified 80/20 split
3.  Train or load SVM model
4.  Evaluate and generate visualizations

The only differences are in the model type and hyperparameters.

### 3. Key Functions & Logic

All functions (`read_data()`, `load_class_map()`, `plot_visualizations()`) are identical to KNN implementation, ensuring consistent evaluation methodology.

### 4. Hyperparameters & Reasoning

#### SVM_PARAMS Configuration

| Parameter | Value | Explanation & Reasoning |
| :--- | :--- | :--- |
| `kernel` | `'rbf'` | **Radial Basis Function kernel.** Tested: [rbf, linear, poly, sigmoid]. **Why RBF?** Image features rarely form linearly separable clusters. RBF creates circular/elliptical decision boundaries by measuring similarity using Gaussian functions. It can approximate any decision boundary given enough training data. Achieved **88.54% accuracy** (best of all kernels). |
| `C` | `10` | **Regularization parameter.** Tested: [0.1, 1, 10, 50, 100, 200]. **Why 10?** Controls the trade-off between smooth decision boundary and classifying training points correctly. Low C (0.1) = smooth boundary but many errors. High C (200) = fits training data perfectly but overfits. C=10 balances generalization and accuracy. |
| `gamma` | `'scale'` | **RBF kernel coefficient.** Options: ['scale', 'auto', 0.001, 0.01, 0.1, 1]. **Why 'scale'?** Defines how far the influence of a single training sample reaches. 'scale' uses `1 / (n_features × X.var())` which automatically adapts to the data's spread. High gamma = only nearby points matter (risk of overfitting). Low gamma = all points matter equally (underfitting). |
| `probability` | `True` | **Enable probability estimates.** Required for `predict_proba()` in the real-time inference script. Adds calibration step to convert SVM decision values into probabilities. |
| `random_state` | `42` | **Reproducibility.** Ensures the same train/test split and model initialization across runs for consistent comparison. |

### 5. SVM vs KNN: Why Both?

| Aspect | KNN | SVM |
| :--- | :--- | :--- |
| **Training Time** | Instant (just stores data) | Slower (finds optimal hyperplane) |
| **Prediction Time** | Slow (calculates distance to all samples) | Fast (evaluates decision function) |
| **Memory Usage** | High (stores all training data) | Low (stores only support vectors) |
| **Decision Boundary** | Piecewise linear (votes from neighbors) | Smooth non-linear (RBF kernel) |
| **Best For** | Small datasets, simple patterns | Large datasets, complex patterns |
| **Accuracy (this project)** | 86.86% | **88.54%** ✓ |

**Why SVM wins here?** Waste classification has overlapping features (e.g., metal and glass both have smooth textures and reflective surfaces). SVM's soft margin and RBF kernel can create more nuanced decision boundaries to separate these ambiguous cases.

---

## 📂 File 5: `input.py`

### 1. Overview & Purpose
**Goal:** Real-time waste classification using live webcam feed.
**Logic:**
This script bridges the gap between trained models and practical application. It captures frames from a webcam, applies the *exact same preprocessing pipeline* used during training, and displays predictions in real-time. The key challenge is maintaining perfect consistency with the training pipeline.

### 2. Workflow
1.  **Model Loading:** Loads both KNN and SVM models, plus all preprocessing artifacts (BoVW, Scaler, PCA).
2.  **Camera Initialization:** Opens the default webcam (device 0).
3.  **Frame Loop:** Continuously captures frames and processes them.
4.  **Feature Extraction:** For each frame, extract the same features used in training.
5.  **Prediction:** Run both models and display results with confidence threshold.
6.  **Display:** Show annotated frame with predictions until user presses 'q'.

### 3. Key Functions & Logic

#### Model Loading Section
```python
knn = joblib.load("knn_model.pkl")
bovw = joblib.load("extracted_features/bovw.pkl")  # FIX 1: Load actual BoVW, not KNN again
svm = joblib.load("svm_model.pkl")
scaler = joblib.load("extracted_features/scaler.pkl")
pca = joblib.load("extracted_features/pca.pkl")  # FIX 2: Required for dimension matching
```

**Critical Detail:** The original code had a bug where `bovw` was accidentally loading the KNN model twice. This would cause feature extraction to fail.

#### `process_frame(frame, model)`
* **Purpose:** The core inference engine that converts raw pixels to prediction.
* **Logic (step-by-step):**
    1.  **Color Conversion:** `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` - OpenCV uses BGR, but feature extraction was trained on RGB.
    2.  **Feature Extraction:** Calls `extract_features(image_rgb, bovw)` which internally:
        * Resizes to 128×128
        * Extracts HOG, LBP, GLCM, LAB stats, BoVW histogram
        * Returns concatenated feature vector (~2000 dimensions)
    3.  **Scaling:** `scaler.transform(X)` - **FIX 3: Uses `.transform()` NOT `.fit_transform()`**. During inference, we must use the *same* scaling parameters learned from training data. `.fit_transform()` would recalculate mean/std from a single frame, giving wrong results.
    4.  **PCA:** `pca.transform(X_Scaled)` - **FIX 4: Reduces from ~2000 to 200 dimensions** to match what models were trained on.
    5.  **Prediction:** `model.predict_proba(X_PCA)[0]` - Returns probability distribution over 6 classes.
    6.  **Thresholding:** If max probability < 60%, classify as "unknown" to avoid false confidence.

### 4. Critical Implementation Details

#### Why `.transform()` vs `.fit_transform()`?

| Method | When to Use | What It Does |
| :--- | :--- | :--- |
| `.fit_transform()` | Training phase | Learns parameters from data AND applies them |
| `.transform()` | Inference phase | Applies previously learned parameters |

**Example:** If training data had mean=100, std=20, and a test pixel has value=140:
* **Correct** (`.transform()`): `(140 - 100) / 20 = 2.0` ✓
* **Wrong** (`.fit_transform()`): `(140 - 140) / 0 = NaN` ✗ (would learn mean=140 from single frame)

#### Confidence Threshold (0.6)

```python
if max_prob < 0.6:
    return class_names[-1]  # "unknown"
```

**Why 60%?** 
* Below 60% indicates the model is "guessing" rather than confidently classifying.
* This prevents showing incorrect predictions when the input is ambiguous or outside the training distribution.
* Example: If probabilities are [0.18, 0.19, 0.21, 0.15, 0.17, 0.10], the model is essentially saying "I have no idea" - showing "plastic" (21%) would mislead the user.

#### Dual Model Display

```python
cv2.putText(frame, f"KNN: {predicted_class_knn}", (10, 30), ...)
cv2.putText(frame, f"SVM: {predicted_class_svm}", (10, 70), ...)
```

**Why show both?**
* **Comparison:** If both agree, high confidence. If they disagree, the object is ambiguous.
* **Transparency:** Users can see both models' opinions rather than a single "black box" answer.
* **Debugging:** Helps developers understand which model is more reliable for which classes.

### 5. Variables & Parameters

| Variable/Parameter | Value | Explanation & Reasoning |
| :--- | :--- | :--- |
| `cv2.VideoCapture(0)` | `0` | **Camera index.** 0 = default webcam. Change to 1, 2, etc. for external cameras. |
| `confidence_threshold` | `0.6` | **Minimum probability.** Prevents showing low-confidence predictions. Tuned empirically - lower (0.4) shows too many false positives, higher (0.8) shows "unknown" too often. |
| `class_names` | 6 classes + "unknown" | **Output labels.** Must match training order: [cardboard, glass, metal, paper, plastic, trash, unknown]. |
| `cv2.waitKey(1)` | `1` millisecond | **Frame delay.** Controls frame rate. 1ms = ~1000 FPS (capped by camera). Higher values slow down video. |

### 6. Testing Without Webcam

```python
# Commented code at bottom:
frame = cv2.imread("test_images/zz.jpg")
predicted_class_knn = process_frame(frame, knn)
```

**Purpose:** Allows testing the pipeline on static images without a webcam. Useful for:
* Debugging feature extraction issues
* Benchmarking prediction speed
* Testing on specific difficult cases

### 7. Common Issues & Solutions

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| "Cannot open camera" | Webcam in use or missing | Check camera permissions, try index 1 instead of 0 |
| Wrong predictions | Wrong color space | Ensure BGR→RGB conversion |
| "Shape mismatch" error | Missing PCA transform | Verify PCA is loaded and applied |
| All predictions "unknown" | Threshold too high OR scaler issue | Check that scaler.transform() is used (not fit_transform) |
| Slow frame rate | Heavy processing | Resize frame before processing, use GPU acceleration |

---

## 🎯 Complete Pipeline Summary

### Data Flow

```
Raw Images (images/)
    ↓
[data_augmentation.py] → Resize + Flip + Rotate + Brightness
    ↓
Augmented Images (augmented/) - 4× larger dataset
    ↓
[feature_extraction.py] → HOG + LBP + GLCM + LAB + BoVW → Scale → PCA
    ↓
Feature Vectors (X.npy, y.npy) - 200 dimensions per sample
    ↓
[knn_training.py / svm_training.py] → Train classifiers
    ↓
Trained Models (*.pkl files)
    ↓
[input.py] → Real-time webcam → Feature extraction → Prediction
    ↓
Live Classification Display
```

### Key Design Decisions

1.  **Why data augmentation?** → Prevents overfitting, simulates real-world variance
2.  **Why multiple features?** → Each captures different aspects (shape, texture, color)
3.  **Why PCA?** → Reduces noise, speeds up training, prevents overfitting
4.  **Why both KNN and SVM?** → KNN is simple/interpretable, SVM is more accurate
5.  **Why confidence threshold?** → Avoids misleading predictions on ambiguous inputs

### Performance Summary

| Component | Metric | Value |
| :--- | :--- | :--- |
| **Dataset** | Images after augmentation | ~2000-3000 |
| **Features** | Dimensions after PCA | 200 |
| **KNN** | Accuracy | 86.86% |
| **SVM** | Accuracy | **88.54%** |
| **Inference** | Processing time | ~100-150ms per frame |

---

## 🚀 Usage Instructions

### 1. Initial Setup
```bash
# Install dependencies
pip install opencv-python numpy scikit-learn scikit-image joblib matplotlib seaborn tqdm

# Verify folder structure
MSI_PROJECT/
├── images/          # Add your raw images here
├── augmented/       # Will be created automatically
└── extracted_features/  # Will be created automatically
```

### 2. Run Pipeline (In Order)
```bash
# Step 1: Augment dataset
python data_augmentation.py

# Step 2: Extract features
python feature_extraction.py

# Step 3: Train models
python knn_training.py
python svm_training.py

# Step 4: Run live classification
python input.py
```

### 3. Expected Outputs
* `augmented/` - Augmented images (4× original count)
* `extracted_features/` - X.npy, y.npy, scaler.pkl, pca.pkl, bovw.pkl, class_map.txt
* `knn_model.pkl` - Trained KNN classifier
* `svm_model.pkl` - Trained SVM classifier
* `knn_visualization.png` - 6-panel analysis plot
* `svm_visualization.png` - 6-panel analysis plot

---

## 🔧 Troubleshooting

### "No images found in augmented/"
**Solution:** Run `data_augmentation.py` first

### "Shape mismatch" during training
**Solution:** Delete `extracted_features/` folder and re-run `feature_extraction.py`

### Low accuracy (<70%)
**Solution:** 
* Ensure images are clear and well-lit
* Increase dataset size (min 100 images per class recommended)
* Verify class folders are correctly named

### Webcam shows "unknown" for everything
**Solution:**
* Check lighting (avoid shadows/glare)
* Verify scaler/PCA are loaded correctly
* Lower confidence threshold from 0.6 to 0.4

