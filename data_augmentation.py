import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = Path("images")
OUTPUT_DIR = Path("augmented")

IMG_SIZE = (224, 224)              # MUST match feature extraction pipeline
ROTATION_RANGE = (-15, 15)         # Degrees (safe for object integrity)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================

def augment_image(img):
    """
    Generates safe augmented versions of an image.

    Augmentations are chosen to preserve class semantics while increasing
    intra-class variability for better generalization.

    Parameters:
        img (np.ndarray): Input BGR image resized to IMG_SIZE

    Returns:
        List[np.ndarray]: Augmented images
    """
    augmented = []

    # 1️⃣ Horizontal Flip (safe for most object classes)
    augmented.append(cv2.flip(img, 1))

    # 2️⃣ Small Random Rotation (keeps object recognizable)
    angle = np.random.uniform(*ROTATION_RANGE)
    center = (IMG_SIZE[0] // 2, IMG_SIZE[1] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, IMG_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    augmented.append(rotated)

    # 3️⃣ Brightness Variation (HSV Value channel)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness_factor = np.random.uniform(0.8, 1.2)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness_factor, 0, 255)
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented.append(bright)

    return augmented

# =============================================================================
# MAIN PIPELINE LOGIC
# =============================================================================

def augment_dataset():
    """
    Main data augmentation pipeline.
    """

    print("\n" + "=" * 70)
    print("🚀 STARTING DATA AUGMENTATION PIPELINE")
    print("=" * 70)
    print(f"Input directory : {INPUT_DIR}/")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Image size      : {IMG_SIZE}")
    print(f"Augmentations   : 3 per image")
    print("=" * 70 + "\n")

    # -------------------------------------------------------------------------
    # Step 1: Validate Input & Output Directories
    # -------------------------------------------------------------------------

    if not INPUT_DIR.exists():
        print(f"❌ ERROR: Input directory '{INPUT_DIR}' does not exist.")
        print("   Please create it and add class folders with images.")
        return

    class_dirs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"❌ ERROR: No class folders found in '{INPUT_DIR}'.")
        print("   Expected structure:")
        print("   images/class_name/image.jpg")
        return

    print(f"✅ Found {len(class_dirs)} classes:")
    for cls in class_dirs:
        print(f"   - {cls.name}")
    print()

    # Create output root directory if not present
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Ensured output root exists: {OUTPUT_DIR.resolve()}\n")

    # Ensure matching class folders exist under OUTPUT_DIR
    created_dirs = []
    for cls in class_dirs:
        out_cls_dir = OUTPUT_DIR / cls.name
        if not out_cls_dir.exists():
            out_cls_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.append(out_cls_dir.name)

    if created_dirs:
        print(f"✅ Created {len(created_dirs)} class directories in '{OUTPUT_DIR}':")
        for name in created_dirs:
            print(f"   - {name}")
    else:
        print("ℹ️  All class directories already existed in the output directory.")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Process Each Class
    # -------------------------------------------------------------------------

    class_counts = {}

    for cls_dir in class_dirs:
        print("=" * 70)
        print(f"📂 Processing class: '{cls_dir.name}'")
        print("=" * 70)

        output_class_dir = OUTPUT_DIR / cls_dir.name

        images = [
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in VALID_EXTENSIONS
        ]

        if not images:
            print(f"⚠️  WARNING: No valid images found in '{cls_dir.name}'. Skipping.")
            continue

        saved_count = 0

        for img_path in tqdm(images, desc=f"Augmenting {cls_dir.name}", ncols=70):
            img = cv2.imread(str(img_path))

            # Skip unreadable files safely
            if img is None:
                print(f"⚠ Could not read image: {img_path.name}")
                continue

            # Resize to match feature extraction requirements
            img = cv2.resize(img, IMG_SIZE)

            # Save original image (overwrite is allowed)
            out_orig = output_class_dir / img_path.name
            success = cv2.imwrite(str(out_orig), img)
            if not success:
                print(f"[ERROR] Failed to write original image: {out_orig}")
            else:
                saved_count += 1

            # Generate and save augmented images
            for i, aug_img in enumerate(augment_image(img)):
                aug_name = f"{img_path.stem}_aug{i}.jpg"
                out_aug = output_class_dir / aug_name
                success = cv2.imwrite(str(out_aug), aug_img)
                if not success:
                    print(f"[ERROR] Failed to write augmented image: {out_aug}")
                else:
                    saved_count += 1

        class_counts[cls_dir.name] = saved_count

        print(f"✅ Finished '{cls_dir.name}' → {saved_count} images saved\n")

    # -------------------------------------------------------------------------
    # Step 3: Final Summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("📊 FINAL CLASS DISTRIBUTION AFTER AUGMENTATION")
    print("=" * 70)

    total_images = 0
    for cls, count in class_counts.items():
        print(f"{cls:12} | {count:5} images")
        total_images += count

    print("-" * 70)
    print(f"📊 Total augmented images: {total_images}")
    print("=" * 70)

    print("\n🎉 DATA AUGMENTATION COMPLETED SUCCESSFULLY")
    print("💡 Next step: Run feature extraction pipeline")
    print("   → python feature_extraction.py")
    print("=" * 70 + "\n")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        augment_dataset()
    except KeyboardInterrupt:
        print("\n\n⚠️  Augmentation interrupted by user.")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
