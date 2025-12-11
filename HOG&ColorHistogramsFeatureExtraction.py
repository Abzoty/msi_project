import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from skimage.feature import hog
from skimage.color import rgb2gray

# all constants I need
DATA_DIR = fr"msi_project/augmented/"
OUTPUT_FEATURES = "features_X.npy"
OUTPUT_LABELS = "labels_y.npy"
OUTPUT_SCALER = "scaler.pkl"
HOG_RESIZE_DIMENSIONS = (64, 128)  #(512, 384) standard image size for HOG
COLOR_HIST_BINS = 32               # standard bins number per channel for Color histograms


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

        # getting the path for every images in all classes we have
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

# extracts color histograms from RGB images takes the image as numpy array and the number of bins per color channel
def extract_color_histograms(image, bins: int=32):

    # checking the validity of the input image
    if image is None or image.size == 0:
        print("extract color: INPUT IMAGE IS EMPTY")
        return
    
    # check if it is really an RGB image (has the 3 channels)
    if image.ndim != 3 or image.shape[2] != 3:
        print("extract color: INPUT IMAGE IS NOT RGB")
        return

    # resizing the image to be more smaller to improve the performance
    # since extracting color features doesn't depend on the image size
    resized_image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)

    # preparing a list that will contain the color histogram features
    # 1 hist for each color channel
    hist_features = []

    for channel in range(3):
        # extracting the color histogram, calcHist return 2d array (bins,1)
        hist = cv2.calcHist(
            [resized_image], # input resized image
            [channel],  # 0, 1, 3
            None,  ## no mask
            [bins], # 32 bins (the default value)
            [0, 256] # intensity range
        )

        hist = hist.flatten() # making it 1D (32,)
        hist = hist / (hist.sum()+ 1e-7) # normalizing the histogram
        hist_features.extend(hist) # appending the histogram to the list

    return np.array(hist_features)

# takes an image as numpy array (RGB), works only on grayscale images
def extract_hog_features(image):

    if image is None or image.size == 0:
        print("extract hog: INPUT IMAGE IS EMPTY")
        return

    if image.ndim == 3:
        image_grayscale = rgb2gray(image) # convert to grayscale (float values in [0,1])
    elif image.ndim == 2:
        image_grayscale = image.astype(np.float64) 
        if image_grayscale.max() > 1.0:
            image_grayscale = image_grayscale / 255.0 # if the image is already in grayscale we ensure its values are in range [0,1]
    else:
        print("extract hog: INPUT IMAGE IS NOT GRAYSCALE OR RGB")
        return
    
    # resizing the image since HOG requires a fixed size (the standard size is 64x128)
    resized_image = cv2.resize(
        (image_grayscale * 255).astype(np.uint8),
        HOG_RESIZE_DIMENSIONS,
        interpolation=cv2.INTER_AREA
    )

    # normalizing the resized image to be in range [0,1]
    resized_image = resized_image.astype(np.float64) / 255.0

    try:
        hog_features = hog(
            resized_image,
            orientations=9, # divide the fragment dir into 9 orientation bins (edge direction groups) 0-180 degrees
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys", # standard HOG normalization method
            visualize=False,
            feature_vector=True
        )
    except:
        print("extract hog: ERROR COMPUTING HOG FEATURES")
        return

    return hog_features

# function to combine the two feature extractors, receives image as numpy array
def extract_features(image):

    # checking the validity of the input image
    if image is None or image.size == 0:
        print("extract features: INPUT IMAGE IS EMPTY")
        return

    hog_features = extract_hog_features(image)
    color_histogram_features = extract_color_histograms(image)
    
    # IMPORTANT: Normalize each feature type separately before combining
    # This gives both feature types equal weight
    hog_normalized = (hog_features - hog_features.mean()) / (hog_features.std() + 1e-7)
    color_normalized = (color_histogram_features - color_histogram_features.mean()) / (color_histogram_features.std() + 1e-7)
    
    combined_features = np.concatenate([hog_normalized, color_normalized])

    return combined_features

# main function to build the features set from preprocessed image and save them to the disk
def build_features_set():
    print(" =========== Loading preprocessed images... =========")
    try:
        images, labels, class_names = load_image(DATA_DIR)
    except Exception as e:
        print(f"ERROR LOADING IMAGE: {e}")
        return
    
    if len(images) == 0:
        print("No images found. Please check your data directory.")
        return
    
    print(f"{len(images)} Images loaded from {len(class_names)} classes successfully.")
    print(f"Classes: {class_names}")


    # extracting features
    X = [] # feature vectors
    y = [] # labels
    failed_count = 0 # counter for failed images

    print(" =========== Extracting features from images... =========")
    for index, (image, label) in enumerate(zip(images, labels)):
        try:
            features = extract_features(image)
            X.append(features)
            y.append(label)

            if (index + 1)%100 == 0:
                print(f"Processed {index + 1} images from {len(images)}...")
        except Exception as e:
            failed_count += 1
            print(f"Failed to extract features from image {index + 1}: {e}")
    
    if failed_count > 0:
        print(f"Failed to extract features from {failed_count} images.")

    # convert features and labels to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print("============== class distribution: =============")
    unique, counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"Class {class_id} ({class_names[class_id]}): {count} samples")
    
    # Check for class imbalance
    min_samples = counts.min()
    max_samples = counts.max()
    imbalance_ratio = max_samples / min_samples
    if imbalance_ratio > 2.0:
        print(f"\n⚠️  WARNING: Class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
        print("Consider using class_weight='balanced' in your classifier or collecting more data for underrepresented classes.")

    print("\nNormalize feature vector using standardScaller ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features normalized successfully.")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    print(f"Total features per image: {X_scaled.shape[1]}")
    print(f"HOG features: ~3780, Color histogram features: 96")
    print(f"Feature mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")

    print("\nSaving features to disk...")
    try:
        np.save(OUTPUT_FEATURES, X_scaled)
        np.save(OUTPUT_LABELS, y)
        joblib.dump(scaler, OUTPUT_SCALER)
        print("Features and labels saved to disk successfully.")
    except Exception as e:
        print(f"ERROR SAVING FEATURES: {e}")



if __name__ == "__main__":
    build_features_set()