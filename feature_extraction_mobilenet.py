"""
feature_extraction_mobilenet.py

CNN-based feature extractor using MobileNetV2 (pretrained on ImageNet).

- Reads images from augmented dataset directory (default: "augmented_balanced/")
    where each class has its own subfolder.
- Extracts MobileNetV2 pooled embeddings (global average pooling).
- Optionally appends simple LAB color stats (mean/std per channel).
- StandardScaler -> PCA applied and saved.
- Outputs into 'extracted_features/':
    - X.npy (PCA-reduced features)
    - X_raw.npy (raw embeddings before scaling/PCA)
    - y.npy (labels)
    - scaler.pkl
    - pca.pkl
    - class_map.txt

Why MobileNetV2?
- Lightweight and fast on CPU / low-power laptops, while still giving strong embeddings.

Dependencies:
    pip install tensorflow opencv-python scikit-learn joblib numpy tqdm

Run:
    python feature_extraction_mobilenet.py

"""

import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import joblib
import cv2

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preprocess

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path("augmented")    # <-- matches augment_and_balance.py default
OUTPUT_DIR = Path("extracted_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE = "mobilenet"   # fixed to MobileNetV2 per your request
IMG_SIZE = (224, 224)    # MobileNetV2 expected input size
APPEND_COLOR_STATS = True
PCA_TARGET = 200
BATCH_SIZE = 16          # small batch size for low-end laptop
RANDOM_STATE = 42

# -----------------------
# Build backbone
# -----------------------
def build_mobilenet(input_shape=(224,224,3)):
    model = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    model.trainable = False
    preprocess_fn = mb_preprocess
    feature_dim = model.output_shape[1]
    return model, preprocess_fn, feature_dim

# -----------------------
# LAB color stats helper
# -----------------------
def lab_stats(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean = lab.mean(axis=(0,1))
    std = lab.std(axis=(0,1))
    return np.concatenate([mean, std])   # shape (6,)

# -----------------------
# Load images & labels
# -----------------------
def load_images_from_dir(data_dir=DATA_DIR):
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Run augmentation first.")
    class_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class subfolders in {data_dir}. Expected structure like '{data_dir}/class_name/*.jpg'")

    class_map = {cls.name: i for i, cls in enumerate(class_dirs)}
    X_imgs = []
    y = []
    print(f"Found classes: {list(class_map.keys())}")
    for cls_name, idx in class_map.items():
        cls_path = data_dir / cls_name
        img_files = [p for p in sorted(cls_path.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        print(f"  {cls_name}: {len(img_files)} images")
        for p in img_files:
            img = cv2.imread(str(p))
            if img is None:
                print(f"Warning: couldn't read {p}")
                continue
            X_imgs.append(img)
            y.append(idx)
    return X_imgs, np.array(y), class_map

# -----------------------
# Main routine
# -----------------------
def extract_and_save(img_size=IMG_SIZE, batch_size=BATCH_SIZE, append_color=APPEND_COLOR_STATS):
    print("Starting MobileNetV2 feature extraction...")
    images, labels, class_map = load_images_from_dir(DATA_DIR)
    n_samples = len(images)
    if n_samples == 0:
        raise ValueError("No images found.")

    model, preprocess_fn, feat_dim = build_mobilenet(input_shape=(img_size[0], img_size[1], 3))
    print(f"MobileNetV2 loaded. Embedding dim = {feat_dim}")

    extra_dim = 6 if append_color else 0
    embeddings = np.zeros((n_samples, feat_dim + extra_dim), dtype=np.float32)

    def preprocess_img(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_LINEAR)
        return img_rgb.astype(np.float32)

    print("Extracting embeddings in batches...")
    for start in tqdm(range(0, n_samples, batch_size), desc="Batches", ncols=80):
        end = min(n_samples, start + batch_size)
        batch_imgs = [preprocess_img(im) for im in images[start:end]]
        batch_arr = np.stack(batch_imgs, axis=0)
        batch_arr = preprocess_fn(batch_arr)
        batch_feats = model.predict(batch_arr, verbose=0)
        if append_color:
            batch_color = []
            for im in images[start:end]:
                batch_color.append(lab_stats(im))
            batch_color = np.stack(batch_color, axis=0).astype(np.float32)
            batch_out = np.concatenate([batch_feats, batch_color], axis=1)
        else:
            batch_out = batch_feats
        embeddings[start:end, :] = batch_out

    print(f"Raw embeddings shape: {embeddings.shape}")
    np.save(OUTPUT_DIR / "X_raw.npy", embeddings)

    # Scaling
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    print(f"Saved scaler -> {OUTPUT_DIR/'scaler.pkl'}")

    # PCA
    n_components = min(PCA_TARGET, embeddings_scaled.shape[0] - 1, embeddings_scaled.shape[1])
    n_components = max(1, n_components)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    embeddings_reduced = pca.fit_transform(embeddings_scaled)
    joblib.dump(pca, OUTPUT_DIR / "pca.pkl")
    print(f"Saved pca -> {OUTPUT_DIR/'pca.pkl'}")
    print(f"Final features shape: {embeddings_reduced.shape}")
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"Variance explained by PCA: {var_explained:.2%}")

    # Save outputs
    np.save(OUTPUT_DIR / "X.npy", embeddings_reduced)
    np.save(OUTPUT_DIR / "y.npy", labels)
    with open(OUTPUT_DIR / "class_map.txt", "w") as f:
        for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {name}\n")

    print("Saved X.npy, y.npy, class_map.txt to", OUTPUT_DIR)
    print("Feature extraction finished. Now run your KNN and SVM training scripts (unchanged).")

if __name__ == "__main__":
    extract_and_save()
