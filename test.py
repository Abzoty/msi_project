import os
from pathlib import Path
import numpy as np
import joblib
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preprocess


def predict(dataFilePath, bestModelPath, scaler_path, pca_path):

    IMG_SIZE = (224, 224)
    APPEND_COLOR_STATS = True
    
    # Convert to Path objects
    data_dir = Path(dataFilePath)
    model_path = Path(bestModelPath)
    
    # Validate paths
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    classifier = joblib.load(str(model_path))
    scaler = joblib.load(str(scaler_path))
    pca = joblib.load(str(pca_path))
    
    # Load MobileNetV2 for feature extraction
    mobilenet = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg"
    )
    mobilenet.trainable = False
    
    # Get list of image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(sorted(data_dir.glob(f"*{ext}")))
        image_files.extend(sorted(data_dir.glob(f"*{ext.upper()}")))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Found {len(image_files)} images to predict")
    
    # Extract features for all images
    features_list = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: couldn't read {img_path}")
            continue
        
        # Preprocess for MobileNetV2
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img_array = img_rgb.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = mb_preprocess(img_array)
        
        # Extract MobileNetV2 features
        features = mobilenet.predict(img_array, verbose=0)[0]
        
        # Append LAB color stats if enabled
        if APPEND_COLOR_STATS:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            mean = lab.mean(axis=(0, 1))
            std = lab.std(axis=(0, 1))
            color_feats = np.concatenate([mean, std])
            features = np.concatenate([features, color_feats])
        
        features_list.append(features)
    
    # Stack features into array
    X_raw = np.stack(features_list, axis=0)
    
    # Apply scaling and PCA transformation
    X_scaled = scaler.transform(X_raw)
    X_pca = pca.transform(X_scaled)
    
    # Make predictions
    predictions = classifier.predict(X_pca)
    predictions_list = predictions.tolist()
    
    print(f"Predictions complete: {len(predictions_list)} samples")
    
    return predictions_list


if __name__ == "__main__":
    data_path = str(input("Enter data directory path: "))
    model_path = str(input("Enter model file path: "))
    scaler_path = str(input("Enter scaler file path: "))
    pca_path = str(input("Enter PCA file path: "))
    
    try:
        predictions = predict(data_path, model_path, scaler_path, pca_path)
        print(f"Predictions: {predictions}")
    except Exception as e:
        print(f"Error: {e}")