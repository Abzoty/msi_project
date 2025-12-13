"""
Feature Extraction Pipeline
-----------------------------------
1. Loads images from 'augmented/' directory.
2. Builds Bag of Visual Words (BoVW) vocabulary using ORB.
3. Extracts concatenated features: HOG + LBP + GLCM + LAB + BoVW.
4. Applies StandardScaler and PCA.
5. Saves all outputs to 'extracted_features/'.
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("augmented")
OUTPUT_DIR = Path("extracted_features")
IMG_SIZE = (128, 128)
BOVW_K = 256  # Size of the visual vocabulary

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def extract_hog(gray):
    """Extracts Histogram of Oriented Gradients (shape)."""
    return hog(gray, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), feature_vector=True)

def extract_lbp(gray):
    """Extracts Local Binary Patterns (texture)."""
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    return hist / (hist.sum() + 1e-6)

def extract_glcm(gray):
    """Extracts GLCM properties (Haralick texture)."""
    glcm = graycomatrix(gray, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    feats = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
    ]
    return np.array(feats)

def extract_lab_stats(img):
    """Extracts Mean and Std Dev of LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mean = lab.mean(axis=(0, 1))
    std = lab.std(axis=(0, 1))
    return np.concatenate([mean, std])

def extract_orb_descriptors(img):
    """Extracts raw ORB descriptors for BoVW construction."""
    orb = cv2.ORB_create(500)
    kp, des = orb.detectAndCompute(img, None)
    return des

def extract_features(img, bovw_model=None):
    """
    Main function to combine all features.
    Resizes image, converts to grayscale, and aggregates feature vectors.
    """
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = [
        extract_hog(gray),
        extract_lbp(gray),
        extract_glcm(gray),
        extract_lab_stats(img)
    ]

    # Add BoVW histogram if model is provided
    if bovw_model is not None:
        des = extract_orb_descriptors(gray)
        if des is None:
            bovw_hist = np.zeros(BOVW_K)
        else:
            words = bovw_model.predict(des)
            bovw_hist, _ = np.histogram(words, bins=BOVW_K, range=(0, BOVW_K))
            bovw_hist = bovw_hist / (bovw_hist.sum() + 1e-6)
        features.append(bovw_hist)

    return np.concatenate(features)


# =============================================================================
# Pipeline Logic
# =============================================================================

def load_images():
    """Loads images and labels from the 'augmented' directory."""
    print("\n" + "="*70)
    print("📂 LOADING IMAGES FROM 'augmented/' DIRECTORY")
    print("="*70)
    
    if not DATA_DIR.exists():
        print(f"❌ ERROR: Data directory '{DATA_DIR}' not found.")
        print(f"   Please run data augmentation first (1_data_augmentation.py)")
        return [], [], {}
    
    # Discover classes
    class_dirs = [cls for cls in DATA_DIR.iterdir() if cls.is_dir()]
    
    if not class_dirs:
        print(f"❌ ERROR: No class folders found in '{DATA_DIR}'.")
        print(f"   Expected folders like: cardboard, glass, metal, paper, plastic, trash")
        return [], [], {}
    
    class_map = {cls.name: i for i, cls in enumerate(sorted(class_dirs))}
    
    print(f"✅ Found {len(class_map)} classes: {list(class_map.keys())}")
    print()
    
    # Count images per class first
    print("📊 Counting images per class:")
    class_counts = {}
    for cls in class_map.keys():
        class_path = DATA_DIR / cls
        count = len([f for f in class_path.glob("*") 
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        class_counts[cls] = count
        print(f"   {cls:12} | {count:5} images")
    
    total_images = sum(class_counts.values())
    print(f"\n📊 Total images to load: {total_images}")
    print("="*70 + "\n")
    
    if total_images == 0:
        print("❌ ERROR: No images found!")
        print("   Check that 'augmented/' contains valid image files")
        return [], [], {}
    
    # Load images with progress bar
    X, y = [], []
    print("⏳ Loading images into memory...")
    
    for cls, label in tqdm(class_map.items(), desc="Loading classes", ncols=70):
        class_path = DATA_DIR / cls
        for img_path in class_path.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                img = cv2.imread(str(img_path))
                if img is not None:
                    X.append(img)
                    y.append(label)
    
    print(f"✅ Successfully loaded {len(X)} images")
    print("="*70 + "\n")
    
    return X, np.array(y), class_map

def build_bovw(images):
    """Builds the Bag of Visual Words vocabulary using K-Means."""
    print("="*70)
    print("🔨 BUILDING BAG OF VISUAL WORDS (BoVW) MODEL")
    print("="*70)
    
    if len(images) == 0:
        raise ValueError("❌ Cannot build BoVW: No images provided.")
    
    print(f"📊 Parameters:")
    print(f"   Vocabulary size (K): {BOVW_K}")
    print(f"   ORB keypoints per image: 500")
    print(f"   Total images: {len(images)}")
    print()
    
    # Collect descriptors with progress bar
    descriptors = []
    print("⏳ Step 1/2: Extracting ORB descriptors from all images...")
    
    failed_count = 0
    for img in tqdm(images, desc="Extracting ORB", ncols=70):
        img_resized = cv2.resize(img, IMG_SIZE)
        des = extract_orb_descriptors(img_resized)
        if des is not None:
            descriptors.append(des)
        else:
            failed_count += 1
    
    if len(descriptors) == 0:
        raise ValueError("❌ No ORB descriptors found. Images might be too small or blank.")
    
    print(f"✅ Collected descriptors from {len(descriptors)} images")
    if failed_count > 0:
        print(f"⚠️  {failed_count} images had no detectable keypoints (skipped)")
    
    # Stack all descriptors
    descriptors = np.vstack(descriptors)
    print(f"📊 Total descriptors: {len(descriptors):,}")
    print()
    
    # Cluster into visual words
    print(f"⏳ Step 2/2: Clustering descriptors into {BOVW_K} visual words...")
    print(f"   Using MiniBatchKMeans (batch_size=1000)...")
    
    kmeans = MiniBatchKMeans(n_clusters=BOVW_K, random_state=42, 
                                batch_size=1000, verbose=0)
    kmeans.fit(descriptors)
    
    print(f"✅ BoVW model trained successfully")
    print(f"💾 Saving model to: {OUTPUT_DIR / 'bovw.pkl'}")
    joblib.dump(kmeans, OUTPUT_DIR / "bovw.pkl")
    print("="*70 + "\n")
    
    return kmeans

def main():
    print("\n" + "="*70)
    print("🚀 STARTING FEATURE EXTRACTION PIPELINE")
    print("="*70)
    print(f"Input directory: {DATA_DIR}/")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Image size: {IMG_SIZE}")
    print("="*70)

    # Step 1: Load Images
    images, labels, class_map = load_images()

    if len(images) == 0:
        print("\n❌ PIPELINE ABORTED: No data found.")
        print("   Please ensure 'augmented/' folder exists and contains images.")
        return

    # Step 2: Build BoVW Model
    try:
        bovw_model = build_bovw(images)
    except Exception as e:
        print(f"\n❌ ERROR building BoVW model: {str(e)}")
        return

    # Step 3: Extract Features
    print("="*70)
    print("🔬 EXTRACTING MULTI-MODAL FEATURES")
    print("="*70)
    print("Feature components:")
    print("   1. HOG (Histogram of Oriented Gradients) - shape/edges")
    print("   2. LBP (Local Binary Patterns) - texture")
    print("   3. GLCM (Gray-Level Co-occurrence Matrix) - texture statistics")
    print("   4. LAB color statistics - color distribution")
    print("   5. BoVW (Bag of Visual Words) - visual vocabulary histogram")
    print()
    
    features = []
    print(f"⏳ Processing {len(images)} images...")
    
    for img in tqdm(images, desc="Extracting features", ncols=70):
        features.append(extract_features(img, bovw_model))
    
    features = np.array(features)
    print(f"✅ Feature extraction complete")
    print(f"📊 Raw feature shape: {features.shape}")
    print(f"   ({features.shape[0]} samples × {features.shape[1]} features)")
    print("="*70 + "\n")

    # Step 4: Scaling and PCA
    print("="*70)
    print("⚙️  SCALING AND DIMENSIONALITY REDUCTION")
    print("="*70)
    
    # Scaling
    print("⏳ Step 1/2: Applying StandardScaler...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"✅ Features scaled (mean=0, std=1)")
    
    # PCA
    n_components = min(150, len(features_scaled), features_scaled.shape[1])
    print(f"\n⏳ Step 2/2: Applying PCA (reducing to {n_components} components)...")
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features_scaled)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"✅ PCA applied successfully")
    print(f"📊 Final feature shape: {features_reduced.shape}")
    print(f"📊 Variance explained: {variance_explained:.2%}")
    print("="*70 + "\n")

    # Step 5: Save Artifacts
    print("="*70)
    print("💾 SAVING ARTIFACTS")
    print("="*70)
    
    print(f"Saving to: {OUTPUT_DIR}/")
    
    # Save models
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    print(f"✅ Saved: scaler.pkl")
    
    joblib.dump(pca, OUTPUT_DIR / "pca.pkl")
    print(f"✅ Saved: pca.pkl")
    
    joblib.dump(bovw_model, OUTPUT_DIR / "bovw.pkl")
    print(f"✅ Saved: bovw.pkl")
    
    # Save features and labels
    np.save(OUTPUT_DIR / "X.npy", features_reduced)
    print(f"✅ Saved: X.npy ({features_reduced.shape[0]} samples)")
    
    np.save(OUTPUT_DIR / "y.npy", labels)
    print(f"✅ Saved: y.npy ({len(labels)} labels)")
    
    # Save class map
    with open(OUTPUT_DIR / "class_map.txt", "w") as f:
        for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {name}\n")
    print(f"✅ Saved: class_map.txt")
    
    print("="*70)
    
    # Summary
    print("\n" + "="*70)
    print("🎉 FEATURE EXTRACTION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"📊 Summary:")
    print(f"   Total samples: {features_reduced.shape[0]}")
    print(f"   Feature dimensions: {features_reduced.shape[1]}")
    print(f"   Classes: {len(class_map)}")
    print(f"   Output directory: {OUTPUT_DIR}/")
    print("="*70)
    print("\n💡 Next step: Run model training scripts")
    print("   - python 3_train_svm.py")
    print("   - python 4_train_knn.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()