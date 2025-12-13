import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray

DATA_DIR = "augmented"
OUTPUT_FEATURES = "extracted_features/features.npy"
OUTPUT_LABELS = "extracted_features/labels.npy"
OUTPUT_SCALER = "extracted_features/scaler.pkl"

HOG_RESIZE_DIMENSIONS = (32, 32) # smaller size for efficiency
HSV_BINS = (8, 8, 4)  # H, S, V bins
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS  # 8 points
LBP_METHOD = 'uniform'  # Produces 26 bins for 8 neighbors


def load_image(augmented_dir: str = DATA_DIR):

    images = [] # list of images, numpy array represent each image
    labels = [] # list of labels, numpy array represent each label
    class_names = [] # list of class names

    # checking input images directory validity
    if not os.path.exists(augmented_dir):
        print("AUGMENTED DIRECTORY NOT FOUND")
        return
    
    # gets the classes names in order
    class_names = sorted(os.listdir(augmented_dir))
    if len(class_names) == 0:
        print("AUGMENTED DIRECTORY IS EMPTY")
        return
    
    for class_id, class_name in enumerate(class_names):
        class_path = os.path.join(augmented_dir, class_name) # gets the class path

        # gets all images in this class
        images_files = [i for i in os.listdir(class_path)
                        if i.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        print(f"class {class_name} with Id = {class_id} loads {len(images_files)} images") 

        # getting the path for every images in all classes we hav
        for image_name in images_files:
            image_path = os.path.join(class_path, image_name)

            image = cv2.imread(image_path)
            if image is None:
                print("CANNOT LOAD THIS IMAGE: ", image_path)
                continue

            # convert to RGB (all featuers extractors expect RGB images)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image) # appending the prepared image to our list with its label
            labels.append(class_id)

    return images, labels, class_names


# Takes RGB image as numpy array, and HSV bins (8,8,4)
def extract_hsv_histogram(image: np.ndarray, bins: tuple = HSV_BINS):
    # Hue: represents the pure color (independent of light)
    # Saturation: represents color intensity
    # Value: represents the brightness

    if image is None or image.size == 0:
        print("HSV_ERROR: Input image is empty or None.")
        return
    
    # Resize for efficiency
    resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Convert RGB to HSV
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    
    # Calculate 3D histogram
    hist = cv2.calcHist(
        [hsv],                           # Image
        [0, 1, 2],                       # Channels (H, S, V)
        None,                            # No mask
        bins,                            # Bins for each channel
        [0, 180, 0, 256, 0, 256]        # Ranges: H[0,180], S[0,256], V[0,256]
    )
    
    # Flatten and normalize
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-7)  # Normalize to sum to 1
    
    return hist

# better for distinguishing materials with different surfaces like papers, metals, and so on
def extract_lbp_features(image: np.ndarray, radius: int = LBP_RADIUS, points: int = LBP_POINTS, method: str = LBP_METHOD):

    # radius: Radius of circle (default: 1)
    # points: Number of circularly symmetric neighbor points (default: 8)
    # method: 'uniform' produces 26 bins for P=8 (more robust)

    if image is None or image.size == 0:
        print("LBP ERROR: Input image is empty or None.")
        return
    
    # Convert to grayscale
    if image.ndim == 3:
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_grayscale = image
    
    # Resize for consistency
    resized = cv2.resize(image_grayscale, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Compute LBP
    lbp = local_binary_pattern(resized, points, radius, method=method)
    
    # Calculate histogram
    # For 'uniform' method with P=8, we get 10 bins (8+2)
    # But sklearn returns values up to P+1, so we use bins accordingly
    if method == 'uniform':
        n_bins = points + 2  # 10 bins for P=8
    else:
        n_bins = 2 ** points
    
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

def extract_edge_histogram(image: np.ndarray):
    """
    Extracts simple edge density features.
    
    Edge information helps distinguish materials with different structural properties:
    - Glass: Often has clear, defined edges
    - Paper: Soft edges
    - Metal: Sharp, reflective edges
    - Cardboard: Textured edges
    
    Returns a 2-element feature vector:
    - Mean edge magnitude
    - Edge density (percentage of edge pixels)
    """
    if image is None or image.size == 0:
        print("RDGE ERROR: Input image is empty or None.")
        return
    
    # Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Resize for consistency
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Apply Canny edge detection
    edges = cv2.Canny(resized, threshold1=50, threshold2=150)
    
    # Compute edge features
    edge_density = np.sum(edges > 0) / edges.size  # Percentage of edge pixels
    
    # Compute Sobel gradients for edge magnitude
    sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    mean_edge_magnitude = np.mean(edge_magnitude) / 255.0  # Normalize
    
    return np.array([edge_density, mean_edge_magnitude])

def extract_hog_features(image: np.ndarray):
    
    if image is None or image.size == 0:
        print("Input image is empty or None.")
        return

    # Convert to grayscale
    if image.ndim == 3:
        gray_image = rgb2gray(image)
    elif image.ndim == 2:
        gray_image = image.astype(np.float64)
        if gray_image.max() > 1.0:
            gray_image = gray_image / 255.0
    else:
        print(f"Unexpected image dimensions: {image.ndim}")
        return
        
    # Resize to 32×32 for efficiency
    resized_image = cv2.resize(
        (gray_image * 255).astype(np.uint8),
        HOG_RESIZE_DIMENSIONS,
        interpolation=cv2.INTER_AREA
    )
    
    # Re-normalize to 0-1 range
    resized_image = resized_image.astype(np.float64) / 255.0

    # Extract HOG Features
    try:
        hog_features = hog(
            resized_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True,
            visualize=False
        )
    except Exception as e:
        print(f"Error extracting HOG features: {e}")
        print(f"Image shape: {resized_image.shape}, dtype: {resized_image.dtype}")
        raise
    
    return hog_features

def extract_features(image: np.ndarray):
    
    if image is None or image.size == 0:
        print("EXTRACT FEATURES: Input image is empty or None.")
        return
    
    try:
        # Extract all feature types
        hsv_hist = extract_hsv_histogram(image, bins=HSV_BINS)
        lbp_hist = extract_lbp_features(image, radius=LBP_RADIUS, points=LBP_POINTS, method=LBP_METHOD)
        edge_hist = extract_edge_histogram(image)
        hog_feat = extract_hog_features(image)
        
        # Combine all features
        combined_features = np.concatenate([
            hsv_hist,    # Color
            lbp_hist,    # Texture
            edge_hist,   # Edges
            hog_feat     # Shape/Gradients
        ])
        
        return combined_features
        
    except Exception as e:
        print(f"ERROR EXTRACT FEATURES: {e}")
        raise

def build_features_set():
    """
    Main function to build the feature dataset from preprocessed images.
    Extracts features, normalizes them, and saves to disk.
    """
    print("=" * 70)
    print("FEATURE EXTRACTION PIPELINE - Option A (Balanced)")
    print("=" * 70)
    
    # Load images
    print("\n Loading preprocessed images...")
    try:
        images, labels, class_names = load_image(DATA_DIR)
    except Exception as e:
        print(f" Error loading images: {e}")
        return

    print(f"\n Loaded {len(images)} images from {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Check for empty dataset
    if len(images) == 0:
        print(" No images found. Please check your data directory.")
        return

    # Extract features
    X = []  # Feature vectors
    y = []  # Labels
    failed_count = 0

    
    for idx, (img, label) in enumerate(zip(images, labels)):
        try:
            features = extract_features(img)
            X.append(features)
            y.append(label)
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(images)} images...")
                
        except Exception as e:
            failed_count += 1
            print(f"  Failed to extract features from image {idx}: {e}")

    if failed_count > 0:
        print(f"\  Warning: {failed_count} images failed feature extraction")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print(f"\n Dataset Statistics:")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Actual feature vector length: {X.shape[1]}")
    print(f"   Feature breakdown:")
    print(f"   └─ HSV Color: {np.prod(HSV_BINS)} dimensions")
    print(f"   └─ LBP Texture: ~10 dimensions")
    print(f"   └─ Edge Features: 2 dimensions")
    print(f"   └─ HOG Shape: ~{X.shape[1] - np.prod(HSV_BINS) - 12} dimensions")
    
    # Print class distribution
    print(f"\ Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"   Class {class_id} ({class_names[class_id]}): {count} samples")

    # Normalize features
    print("\ Normalizing feature vectors using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   Mean: {X_scaled.mean():.6f}")
    print(f"   Std: {X_scaled.std():.6f}")

    # Save outputs
    print("\n Saving features, labels, and scaler...")
    try:
        np.save(OUTPUT_FEATURES, X_scaled)
        np.save(OUTPUT_LABELS, y)
        joblib.dump(scaler, OUTPUT_SCALER)
        
        print(f"   Features saved to: {OUTPUT_FEATURES}")
        print(f"   Labels saved to: {OUTPUT_LABELS}")
        print(f"   Scaler saved to: {OUTPUT_SCALER}")
        
    except Exception as e:
        print(f"    Error saving files: {e}")
        return

    print("\n" + "=" * 70)
    print(" FEATURE DATASET CREATED SUCCESSFULLY!")
    print(" Feature matrix shape:", X_scaled.shape)
    print("=" * 70)



if __name__ == "__main__":
    build_features_set()