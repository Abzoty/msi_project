"""
Minimal Data Augmentation Script for Material Stream Identification (MSI) System
Balances class distributions by augmenting images to reach target count per class.
"""

import cv2
import numpy as np
from pathlib import Path
import albumentations as A
import random
import shutil


def create_augmentation_pipeline():
    """Create augmentation pipeline with geometric and lighting transforms."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, 
                            border_mode=cv2.BORDER_REPLICATE, p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(0.0, (0.05 * 255) ** 2), p=0.4),
    ])


def count_images(source_dir, class_names):
    """Count images in each class folder."""
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    counts = {}
    for cls in class_names:
        cls_path = Path(source_dir) / cls
        if cls_path.exists():
            counts[cls] = len([f for f in cls_path.iterdir() if f.suffix.lower() in valid_ext])
        else:
            counts[cls] = 0
    return counts


def calculate_augmentation_plan(counts, target_count):
    """Calculate how many originals to copy and augmentations to generate per class."""
    plan = {}
    for cls, count in counts.items():
        if count >= target_count:
            plan[cls] = {'copy': target_count, 'augment': 0}
        else:
            plan[cls] = {'copy': count, 'augment': target_count - count}
    return plan


def augment_image(image, pipeline):
    """Apply augmentation pipeline to image."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = pipeline(image=img_rgb)['image']
    return cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)


def process_class(cls, plan, source_dir, target_dir, pipeline):
    """Copy originals and generate augmentations for a single class."""
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    src_path = Path(source_dir) / cls
    tgt_path = Path(target_dir) / cls
    tgt_path.mkdir(parents=True, exist_ok=True)
    
    # Get original images
    orig_imgs = [f for f in src_path.iterdir() if f.suffix.lower() in valid_ext]
    if not orig_imgs:
        return
    
    # Copy originals
    imgs_to_copy = random.sample(orig_imgs, plan['copy']) if len(orig_imgs) > plan['copy'] else orig_imgs
    for img_file in imgs_to_copy:
        img = cv2.imread(str(img_file))
        if img is not None:
            cv2.imwrite(str(tgt_path / img_file.name), img)
    
    # Generate augmentations
    for i in range(plan['augment']):
        src_img = random.choice(orig_imgs)
        img = cv2.imread(str(src_img))
        if img is not None:
            aug_img = augment_image(img, pipeline)
            new_name = f"{src_img.stem}_aug_{i:04d}{src_img.suffix}"
            cv2.imwrite(str(tgt_path / new_name), aug_img)


def main():
    """Main function to run data augmentation."""
    source_dir = 'images'
    target_dir = 'augmented'
    target_count = 500
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Clean target directory
    if Path(target_dir).exists():
        shutil.rmtree(target_dir)
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate plan and process
    counts = count_images(source_dir, class_names)
    plan = calculate_augmentation_plan(counts, target_count)
    pipeline = create_augmentation_pipeline()
    
    print(f"Augmenting images to {target_count} per class...")
    for cls in class_names:
        print(f"Processing {cls}: copy {plan[cls]['copy']}, augment {plan[cls]['augment']}")
        process_class(cls, plan[cls], source_dir, target_dir, pipeline)
    
    print("Augmentation complete!")


if __name__ == "__main__":
    main()