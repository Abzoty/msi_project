"""
Minimal Feature Extraction Script for Material Stream Identification (MSI) System
Extracts multi-modal features (HOG, Color Histograms, LBP, Statistics) from images.
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.feature import hog, local_binary_pattern
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def extract_features(image_path, img_size=(128, 128)):
    """
    Extract multi-modal features from an image.
    This function can be reused for inference on new frames.
    
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
        
        # Resize and convert to grayscale
        img = cv2.resize(img, img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. HOG features (shape/edges)
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        
        # 2. Color histogram features (HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
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


def process_dataset(augmented_dir='augmented', output_dir='extracted_features'):
    """Process entire dataset and extract features."""
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
    
    # Convert to numpy arrays and save
    feat_matrix = np.array(features, dtype=np.float32)
    label_array = np.array(labels, dtype=np.int32)
    
    np.save(out_path / 'features.npy', feat_matrix)
    np.save(out_path / 'labels.npy', label_array)
    
    print(f"Extracted features: {feat_matrix.shape}")
    print(f"Saved to {output_dir}/")
    
    return feat_matrix, label_array


def main():
    """Main function to run feature extraction."""
    process_dataset(augmented_dir='augmented', output_dir='extracted_features')
    print("Feature extraction complete!")


if __name__ == "__main__":
    main()