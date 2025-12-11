# aug_albumentations.py
import os
import cv2
import random
import numpy as np
from glob import glob
import albumentations as A

# ------------ Config -------------
CLASS = "cardboard"                 # change this to change both input and output paths (make sure to name your folders accordingly)
SRC_DIR = fr"images/{CLASS}"        # source folder with original images
OUT_DIR = fr"augmented/{CLASS}"     # where augmented images will be saved
IMG_SIZE = (512, 384)               # final image size (width, height)
RANDOM_SEED_1 = 42
RANDOM_SEED_2 = 73
num_images = 5                      # set to None to process all images
# ----------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# --- Utility: set all seeds for reproducible transforms ---
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)

# --- Albumentations pipelines ---

# Variant 1: mild augmentations (small rotation/scale & color jitter)
transform_variant_1 = A.Compose([
    # Geometric: combine shift/scale/rotate (scale_limit controls zoom in/out)
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.12, rotate_limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    A.HorizontalFlip(p=0.5),
    # Color / illumination
    A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.9),
    A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=20, val_shift_limit=10, p=0.9),
    A.GaussNoise(var_limit=(1.0, 5.0), p=0.2),
    # Ensure final size (some transforms can change size if you use crop)
    A.Resize(height=IMG_SIZE[1], width=IMG_SIZE[0], interpolation=cv2.INTER_LINEAR)
], p=1.0)

# Variant 2: stronger augmentations (wider rotation/scale & color changes)
transform_variant_2 = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.05, 
        scale_limit=0.15,  # Changed from (0.4, 0.4) to 0.15
        rotate_limit=30,    # Changed from 50 to 30
        interpolation=cv2.INTER_LINEAR, 
        border_mode=cv2.BORDER_REFLECT_101, 
        p=1.0
    ),
    A.HorizontalFlip(p=0.5),
    # stronger brightness/contrast and hue/saturation
    A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.95),
    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=40, val_shift_limit=20, p=0.95),
    A.GaussNoise(var_limit=(4.0, 8.0), p=0.3),
    A.Resize(height=IMG_SIZE[1], width=IMG_SIZE[0], interpolation=cv2.INTER_LINEAR)
], p=1.0)

# --- Main processing ---
def process_folder(src_dir, out_dir, n_images=None):
    # Collect supported image files
    img_paths = sorted(
        glob(os.path.join(src_dir, "*.jpg")) +
        glob(os.path.join(src_dir, "*.jpeg")) +
        glob(os.path.join(src_dir, "*.png"))
    )
    if not img_paths:
        print("No images found in", src_dir)
        return

    if n_images is not None:
        img_paths = img_paths[:n_images]

    print(f"Processing {len(img_paths)} images from {src_dir} -> {out_dir}")

    for i, img_path in enumerate(img_paths):
        base = os.path.splitext(os.path.basename(img_path))[0]
        # Read image in BGR (OpenCV)
        img = cv2.imread(img_path)
        if img is None:
            print("Could not read", img_path)
            continue

        # Ensure consistent size first (optional).
        img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

        # Save original resized copy (optional)
        orig_out_path = os.path.join(out_dir, f"{base}_orig.jpg")
        cv2.imwrite(orig_out_path, img_resized)

        # Variant 1: use seed1 + image index to get reproducible but varied results per-image
        set_seeds(RANDOM_SEED_1 + i)
        augmented1 = transform_variant_1(image=img_resized)["image"]

        # Variant 2: use seed2 + image index
        set_seeds(RANDOM_SEED_2 + i)
        augmented2 = transform_variant_2(image=img_resized)["image"]

        # Save augmented outputs
        out1 = os.path.join(out_dir, f"{base}_aug1.jpg")
        out2 = os.path.join(out_dir, f"{base}_aug2.jpg")
        cv2.imwrite(out1, augmented1)
        cv2.imwrite(out2, augmented2)

        print(f"[{i+1}/{len(img_paths)}] Saved: {os.path.basename(out1)}, {os.path.basename(out2)}")

    print("Done. Check:", out_dir)


if __name__ == "__main__":
    process_folder(SRC_DIR, OUT_DIR, n_images=num_images)
