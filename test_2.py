import cv2
import numpy as np
import joblib
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preprocess
import pandas as pd
from datetime import datetime

# -----------------------
# GLOBAL CONFIG
# -----------------------
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
    "unknown"
]

# Load Model scaler + PCA (shared)
scaler = joblib.load("extracted_features/scaler.pkl")  # Path to the trained scaler for features scaling
pca = joblib.load("extracted_features/pca.pkl")        # Path to the trained PCA model for dimensionality reduction
images_path = "test_images"         # Path to the folder containing test images
svm_model_path = "svm_model.pkl"    # Path to the trained SVM model  

# Load MobileNetV2 backbone
mobilenet = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="avg"
)
mobilenet.trainable = False


# -----------------------
# Prediction function
# -----------------------
def predict(dataFilePath, bestModelPath):

    data_dir = Path(dataFilePath)
    model_path = Path(bestModelPath)

    if not data_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {data_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"SVM model not found: {model_path}")

    # Load SVM model
    svm = joblib.load(model_path)

    # Collect image files
    image_files = [
        p for p in sorted(data_dir.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    if not image_files:
        raise ValueError(f"No images found in {data_dir}")

    predictions = []

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠ Could not read {img_path.name}, skipping")
            continue

        # 1️⃣ Extract MobileNetV2 features
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_rgb = img_rgb.astype(np.float32)

        x = np.expand_dims(img_rgb, axis=0)
        x = mb_preprocess(x)

        embedding = mobilenet.predict(x, verbose=0)

        # Append LAB stats
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean = lab.mean(axis=(0, 1))
        std = lab.std(axis=(0, 1))
        color = np.concatenate([mean, std]).reshape(1, -1)  # (1, 6)
        feats = np.concatenate([embedding, color], axis=1)

        # 2️⃣ Scale
        feats_scaled = scaler.transform(feats)

        # 3️⃣ PCA
        feats_pca = pca.transform(feats_scaled)

        # 4️⃣ Predict
        pred_idx = svm.predict(feats_pca)[0]
        pred_class = CLASS_NAMES[pred_idx]

        predictions.append((img_path.name, pred_class))

    return predictions


# -----------------------
# Save to Excel function
# -----------------------
def save_to_excel(predictions, output_file="predictions_results.xlsx"): 
    
    # Create DataFrame
    df = pd.DataFrame(predictions, columns=["Image Name", "Predicted Class"])
    
    # Save to Excel with formatting
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Predictions']
        
        # Adjust column widths
        worksheet.column_dimensions['A'].width = 40
        worksheet.column_dimensions['B'].width = 20
    
    print(f"\n✅ Results saved to: {output_file}")
    return output_file


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":

    print("🔍 Starting prediction process...")
    print(f"📁 Loading images from: {images_path}")
    print(f"🤖 Using model: {svm_model_path}\n")
    
    results = predict(images_path, svm_model_path)
    
    print("=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    for idx, (name, pred) in enumerate(results, 1):
        print(f"{idx:3d}. {name:40s} -> {pred}")
    
    print("=" * 60)
    print(f"Total images processed: {len(results)}")
    print("=" * 60)
    
    # Save to Excel
    excel_file = save_to_excel(results)