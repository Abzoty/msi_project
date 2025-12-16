"""
input_cnn.py

Live camera inference using:
- MobileNetV2 feature extractor (CPU-friendly)
- Pretrained scaler + PCA
- KNN and SVM classifiers

Must match:
- feature_extraction_mobilenet.py EXACTLY

Dependencies:
    pip install tensorflow-cpu opencv-python scikit-learn joblib numpy
"""

import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preprocess

# -----------------------
# CONFIG
# -----------------------
IMG_SIZE = (224, 224)
FEATURE_DIR = "extracted_features"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
    "unknown"
]

UNKNOWN_THRESHOLD = 0.60   # confidence threshold

# -----------------------
# Load models
# -----------------------
print("Loading models...")

knn = joblib.load("knn_model.pkl")
svm = joblib.load("svm_model.pkl")

scaler = joblib.load(f"{FEATURE_DIR}/scaler.pkl")
pca = joblib.load(f"{FEATURE_DIR}/pca.pkl")

# -----------------------
# Load MobileNetV2
# -----------------------
print("Loading MobileNetV2 backbone...")
mobilenet = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg"
)
mobilenet.trainable = False

print("Models loaded successfully.\n")

# -----------------------
# Feature extraction
# -----------------------
def lab_stats(img_bgr):
    """
    Extract LAB color statistics (mean and std per channel).
    Matches training pipeline in feature_extraction_mobilenet.py
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean = lab.mean(axis=(0, 1))
    std = lab.std(axis=(0, 1))
    return np.concatenate([mean, std])  # shape (6,)

def extract_cnn_features(frame_bgr):
    """
    Extract CNN features from a single frame using MobileNetV2.
    Output shape matches training pipeline.
    """

    # BGR → RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Resize
    img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    # Convert to float32
    img_rgb = img_rgb.astype(np.float32)

    # Expand batch dimension
    x = np.expand_dims(img_rgb, axis=0)

    # MobileNet preprocessing
    x = mb_preprocess(x)

    # Extract embedding
    embedding = mobilenet.predict(x, verbose=0)
    
    # Extract LAB color stats (CRITICAL: must match training!)
    color_stats = lab_stats(frame_bgr)
    color_stats = color_stats.reshape(1, -1)  # shape: (1, 6)
    
    # Concatenate MobileNet features + color stats
    features = np.concatenate([embedding, color_stats], axis=1)

    return features  # shape: (1, 1286) = 1280 + 6

# -----------------------
# Prediction
# -----------------------
def predict_frame(frame, model):
    """
    Predict class name for a single frame using given model (KNN or SVM).
    """

    # 1️⃣ CNN feature extraction
    features = extract_cnn_features(frame)

    # 2️⃣ Scale (NO fitting!)
    features_scaled = scaler.transform(features)

    # 3️⃣ PCA reduction
    features_pca = pca.transform(features_scaled)

    # 4️⃣ Predict probabilities
    probs = model.predict_proba(features_pca)[0]

    max_prob = probs.max()
    pred_idx = probs.argmax()

    # Unknown handling
    if max_prob < UNKNOWN_THRESHOLD:
        return CLASS_NAMES[-1], max_prob

    return CLASS_NAMES[pred_idx], max_prob

# -----------------------
# Main camera loop
# -----------------------
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("🎥 Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        try:
            knn_pred, knn_conf = predict_frame(frame, knn)
            svm_pred, svm_conf = predict_frame(frame, svm)

            # Console output
            print(f"KNN: {knn_pred} ({knn_conf:.2f}) | SVM: {svm_pred} ({svm_conf:.2f})")

            # Overlay on frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"KNN: {knn_pred}", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"SVM: {svm_pred}", (10, 70), font, 1, (0, 0, 255), 2)

            cv2.imshow("Recycle Smart (CNN)", frame)

        except Exception as e:
            print(f"⚠ Error processing frame: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------
if __name__ == "__main__":
    main()