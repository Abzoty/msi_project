import os
import cv2
import numpy as np
import random
from glob import glob

# ------------ Config -------------
SRC_DIR = "images\cardboard"          # <-- change to your source class folder
OUT_DIR = "augmented\cardboard" # <-- where augmented images will be saved
IMG_SIZE = (512, 384)           # original images already 512x384; you can resize if you want
RANDOM_SEED_1 = 42                # for reproducibility of random augmentations, helpful for debugging
RANDOM_SEED_2 = 73                # for reproducibility of random augmentations, helpful for debugging
num_images = 5                 # set to None to process all images
# ----------------------------------

# Create two independent RNGs for Python-level randomness
rng1 = random.Random(RANDOM_SEED_1)
rng2 = random.Random(RANDOM_SEED_2)

# Create two independent NumPy RNGs for NumPy randomness (Gaussian noise)
np_rng1 = np.random.default_rng(RANDOM_SEED_1)
np_rng2 = np.random.default_rng(RANDOM_SEED_2)

os.makedirs(OUT_DIR, exist_ok=True) # ensure output directory exists

# ---------- Geometric transforms ----------
def flip_horizontal(img):
    return cv2.flip(img, 1)

def flip_vertical(img):
    return cv2.flip(img, 0)

def rotate(img, angle):
    h, w = img.shape[:2] # height, width
    # create rotation matrix M
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0) # rotation matrix, w/2 and h/2 are center coordinates, angle in degrees, scale=1.0 to keep same size
    #"flags": How to calculate pixel values for new positions, Bilinear (default, good quality)
    #"borderMode": What to do with pixels outside the image boundary, REFLECT to avoid black borders
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def scale(img, scale_factor):
    h, w = img.shape[:2] # height, width
    new_w, new_h = int(w * scale_factor), int(h * scale_factor) # new dimensions

    # 1. Resize image
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 2. CASE A: If scaled image is larger → center-crop
    if new_w >= w and new_h >= h:
        x0 = (new_w - w) // 2
        y0 = (new_h - h) // 2
        cropped = scaled[y0:y0 + h, x0:x0 + w]
        return cropped

    # 3. CASE B: If scaled image is smaller → pad symmetrically
    # Calculate padding amounts
    pad_left   = (w - new_w) // 2
    pad_right  = w - new_w - pad_left
    pad_top    = (h - new_h) // 2
    pad_bottom = h - new_h - pad_top

    # Pad image
    padded = cv2.copyMakeBorder(
        scaled,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REFLECT_101
    )

    return padded


# ---------- Color / illumination transforms ----------
def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    # alpha: contrast (1.0 = no change), beta: brightness added (0 = no change)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out

def add_gaussian_noise(img, mean=0, sigma=10):
    #mean: Mean of Gaussian distribution (default 0), controls brightness shift
    #sigma: Standard deviation of Gaussian (default 10), controls noise intensity
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    # Convert image to float32 for safe addition (prevents overflow), then add generated noise to original image
    noisy = img.astype(np.float32) + gauss
    # Clip values to valid image range (0-255) after noise addition, then convert back to uint8 (standard image format)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def color_jitter(img, hue_shift=0, sat_scale=1.0):
    # operate in HSV: hue shift (-180..180), saturation scale
    # Convert from BGR to HSV color space for easier color manipulation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Apply hue shift
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    # Adjust saturation
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    # Convert back to BGR color space for output
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

# ---------- Combined augmentation pipelines ----------
def geom_aug_variant_1(img):
    # Scale (zoom in/out)
    scale_f = rng1.uniform(0.9, 1.1)
    out_1 = scale(img, scale_f)
    # Flip horizontally + small rotation
    out_2 = flip_horizontal(out_1)
    angle = rng1.uniform(-50, 50)
    out_3 = rotate(out_2, angle)
    return out_3

def geom_aug_variant_2(img):
    # Scale (zoom in/out)
    scale_f = rng2.uniform(0.6, 1.4)
    out_1 = scale(img, scale_f)
    # Flip horizontally + small rotation
    out_2 = flip_horizontal(out_1)
    angle = rng2.uniform(-50, 50)
    out_3 = rotate(out_2, angle)
    return out_3

def color_aug_variant_1(img):
    # Brightness and contrast jitter
    alpha = rng1.uniform(0.9, 1.15)   # contrast
    beta = rng1.uniform(-30, 30)      # brightness
    out_1 = adjust_brightness_contrast(img, alpha=alpha, beta=beta)
    # Add gaussian noise + slight saturation change
    sigma = rng1.uniform(5, 20)
    out_2 = add_gaussian_noise(out_1, sigma=sigma)
    sat = rng1.uniform(0.9, 1.2)
    hue = rng1.uniform(-5, 5)  # small hue shift
    out_3 = color_jitter(out_2, hue_shift=hue, sat_scale=sat)
    return out_3

def color_aug_variant_2(img):
    # Brightness and contrast jitter
    alpha = rng2.uniform(0.9, 1.15)   # contrast
    beta = rng2.uniform(-30, 30)      # brightness
    out_1 = adjust_brightness_contrast(img, alpha=alpha, beta=beta)
    # Add gaussian noise + slight saturation change
    sigma = rng2.uniform(5, 20)
    out_2 = add_gaussian_noise(out_1, sigma=sigma)
    sat = rng2.uniform(0.9, 1.2)
    hue = rng2.uniform(-5, 5)  # small hue shift
    out_3 = color_jitter(out_2, hue_shift=hue, sat_scale=sat)
    return out_3

# ---------- Main processing ----------
def process_folder(src_dir, out_dir, num_images=None):
    img_paths = sorted(glob(os.path.join(src_dir, "*.jpg")) + glob(os.path.join(src_dir, "*.jpeg")) + glob(os.path.join(src_dir, "*.png")))
    if not img_paths:
        print("No images found in", src_dir)
        return
    if num_images is not None:
        img_paths = img_paths[:num_images]
    print(f"Processing {len(img_paths)} images from {src_dir}")
    for i, p in enumerate(img_paths):
        base_name = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        if img is None:
            print("Could not read", p)
            continue

        # Save resized original
        cv2.imwrite(os.path.join(out_dir, f"{base_name}_orig.jpg"), img)

        # Geometric Augmentations (2 variants)
        g1 = geom_aug_variant_1(img)
        g2 = geom_aug_variant_2(img)

        # Color / Illumination Augmentations (2 variants)
        c1 = color_aug_variant_1(g1)
        c2 = color_aug_variant_2(g2)
        cv2.imwrite(os.path.join(out_dir, f"{base_name}_aug1.jpg"), c1)
        cv2.imwrite(os.path.join(out_dir, f"{base_name}_aug2.jpg"), c2)

        print(f"[{i+1}/{len(img_paths)}] Saved: {base_name}_orig, aug1, aug2")

if __name__ == "__main__":
    process_folder(SRC_DIR, OUT_DIR, num_images=num_images)
    print("Done. Check:", OUT_DIR)
