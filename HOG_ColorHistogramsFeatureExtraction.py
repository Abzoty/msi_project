"""
Minimal Feature Extraction with Dimensionality Reduction for KNN
Reduces features from ~8000 to ~100-300 dimensions using PCA
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def extract_features(image_path, img_size=(64, 64)):
    """
    Extract reduced multi-modal features from an image.
    Reduced image size from 128x128 to 64x64 to cut features by ~75%
    
    Args:
        image_path: Path to image file or numpy array (BGR image)
        img_size: Target size for resizing (height, width)
    
    Returns:
        Combined feature vector or None if extraction fails
    """
    try:
        # Handle both file paths and numpy arrays
        if isinstance(image_path, (str, Path)):
            img = cv2.imread(str(image_path))
            if img is None:
                return None
        else:
            img = image_path
        
        # Resize to smaller size (64x64 instead of 128x128)
        img = cv2.resize(img, img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. HOG features (reduced parameters)
        hog_feat = hog(gray, orientations=6, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        
        # 2. Color histogram features (reduced bins)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        color_feat = np.concatenate([hist_h, hist_s, hist_v])
        
        # 3. LBP features (texture)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        n_bins = 8 * (8 - 1) + 3
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        lbp_feat = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-7)
        
        # 4. Color statistics
        stat_feat = np.concatenate([img.mean(axis=(0, 1)), img.std(axis=(0, 1))])
        
        # Combine all features
        return np.concatenate([hog_feat, color_feat, lbp_feat, stat_feat])
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def apply_pca_reduction(features, n_components=100, scaler=None, pca=None, fit=True):
    """
    Apply StandardScaler + PCA to reduce dimensionality.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        n_components: Target number of dimensions (100-300 recommended)
        scaler: Pre-fitted scaler (for inference)
        pca: Pre-fitted PCA (for inference)
        fit: Whether to fit scaler/PCA or just transform
    
    Returns:
        reduced_features, scaler, pca
    """
    if fit:
        # Fit and transform
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        pca = PCA(n_components=n_components)
        features_reduced = pca.fit_transform(features_scaled)
        
        print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        # Just transform (for inference)
        features_scaled = scaler.transform(features)
        features_reduced = pca.transform(features_scaled)
    
    return features_reduced, scaler, pca


def process_dataset(augmented_dir='augmented', output_dir='extracted_features', 
                    n_components=100):
    """
    Process entire dataset, extract features, and apply PCA reduction.
    
    Args:
        n_components: Number of PCA components (50-300 range)
                        - 50: Very fast, may lose info
                        - 100: Good balance (RECOMMENDED)
                        - 150: Better accuracy, slower
                        - 200-300: Diminishing returns
    """
    aug_path = Path(augmented_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Class mapping
    class_map = {'glass': 0, 'paper': 1, 'cardboard': 2, 'plastic': 3, 'metal': 4, 'trash': 5}
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Collect all image paths with labels
    img_label_pairs = []
    for cls, label in class_map.items():
        cls_path = aug_path / cls
        if cls_path.exists():
            imgs = [f for f in cls_path.iterdir() if f.suffix.lower() in valid_ext]
            img_label_pairs.extend([(img, label) for img in imgs])
    
    print(f"Processing {len(img_label_pairs)} images...")
    
    # Extract features from all images
    features = []
    labels = []
    for img_path, label in img_label_pairs:
        feat = extract_features(img_path)
        if feat is not None:
            features.append(feat)
            labels.append(label)
    
    # Convert to numpy arrays
    feat_matrix = np.array(features, dtype=np.float32)
    label_array = np.array(labels, dtype=np.int32)
    
    print(f"Raw features shape: {feat_matrix.shape}")
    
    # Apply PCA reduction
    print(f"Applying PCA reduction to {n_components} components...")
    feat_reduced, scaler, pca = apply_pca_reduction(feat_matrix, n_components=n_components)
    
    print(f"Reduced features shape: {feat_reduced.shape}")
    
    # Save reduced features and preprocessing objects
    np.save(out_path / 'features.npy', feat_reduced)
    np.save(out_path / 'labels.npy', label_array)
    joblib.dump(scaler, out_path / 'scaler.pkl')
    joblib.dump(pca, out_path / 'pca.pkl')
    
    print(f"Saved to {output_dir}/")
    print(f"  - features.npy: {feat_reduced.shape}")
    print(f"  - scaler.pkl & pca.pkl: for inference")
    
    return feat_reduced, label_array


def extract_features_for_inference(image, scaler_path='extracted_features/scaler.pkl',
                                    pca_path='extracted_features/pca.pkl'):
    """
    Extract features from a single image for real-time inference.
    Uses pre-fitted scaler and PCA.
    
    Args:
        image: BGR image array (from video frame)
        scaler_path: Path to saved scaler
        pca_path: Path to saved PCA
    
    Returns:
        Reduced feature vector ready for KNN prediction
    """
    # Load preprocessing objects
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    
    # Extract raw features
    raw_features = extract_features(image)
    if raw_features is None:
        return None
    
    # Apply same preprocessing
    features_scaled = scaler.transform(raw_features.reshape(1, -1))
    features_reduced = pca.transform(features_scaled)
    
    return features_reduced[0]


def main():
    """
    Main function to run feature extraction with PCA reduction.
    
    Recommended n_components values:
    - 50: Very fast, lower accuracy (~60-70%)
    - 100: RECOMMENDED - good balance (~70-80%)
    - 150: Better accuracy, slower (~75-85%)
    - 200-300: Marginal gains, not worth it
    """
    process_dataset(
        augmented_dir='augmented', 
        output_dir='extracted_features',
        n_components=200  # Change this value to experiment
    )
    print("Feature extraction complete!")


if __name__ == "__main__":
    main()