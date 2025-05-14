import os
import cv2
import glob
import random
import albumentations as A

# --- CONFIGURATION ---

TARGET_SIZE = (640, 640)
SPLIT_RATIOS = {'train': 0.64, 'val': 0.16, 'test': 0.2}
OUTPUT_ROOT = './Road_Damage'
ROTATION_SAMPLES = 3

# Dataset configuration: (image_dir, class_name) - Removed label_dir
DATASETS = [
    (
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Potholes - Annotated/1 - Potholes-20250429T144329Z-001/1 - Potholes',
        'potholes'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Cracks - Annotated/12 - Cracks',
        'crack_issues'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Alligator Cracks - Annotated/2 - Alligator Cracks - IMAGES',
        'alligator_crack_issues'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Ravelling - Annotated/Ravelling - IMAGES',
        'ravelling'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Open Manhole - Annotated/10 - Open Manhole - IMAGES',
        'open_manhole'
    )
]

# Create directory structure (simplified - no labels)
for split in SPLIT_RATIOS:
    for _, class_name in DATASETS:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, class_name, 'images'), exist_ok=True)

# --- SPLIT AND SAVE ORIGINALS ---
for img_dir, class_name in DATASETS:  # Removed lbl_dir
    imgs = glob.glob(os.path.join(img_dir, '*.[jJ][pP][gG]')) + glob.glob(os.path.join(img_dir, '*.[pP][nN][gG]'))
    print(f"[DEBUG] {class_name}: found {len(imgs)} images.")
    items = [img for img in imgs]  # Simplified - no label path

    random.seed(42)
    random.shuffle(items)
    total = len(items)
    test_count = int(total * SPLIT_RATIOS['test'])
    val_count = int(total * SPLIT_RATIOS['val'])

    split_items = {
        'train': items[:total - test_count - val_count],
        'val': items[total - test_count - val_count: total - test_count],
        'test': items[total - test_count:]
    }

    for split, batch in split_items.items():
        for img_path in batch:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {img_path}: cannot read image.")
                continue
            img_res = cv2.resize(img, TARGET_SIZE)
            dst_img = os.path.join(OUTPUT_ROOT, split, class_name, 'images', f"{fname}.jpg")
            cv2.imwrite(dst_img, img_res)

# --- AUGMENTATION CONFIGURATION ---

def sharpen_image(x, **kwargs):
    # Apply sharpening using Laplacian
    laplacian = cv2.Laplacian(x, cv2.CV_16S, ksize=3)
    sharp = cv2.addWeighted(x, 0.7, cv2.convertScaleAbs(laplacian), 0.3, 0)
    return sharp

augmentations = {
    'exposure': A.Compose([
        A.Resize(*TARGET_SIZE),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0)
    ]),  # Removed bbox_params
    'rotation': A.Compose([
        A.Resize(*TARGET_SIZE),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    ]),  # Removed bbox_params
    'scaling': A.Compose([
        A.Resize(*TARGET_SIZE),
        A.RandomScale(scale_limit=0.3, p=1.0)
    ]),  # Removed bbox_params
    'motion_blur': A.Compose([
        A.Resize(*TARGET_SIZE),
        A.MotionBlur(blur_limit=(5, 15), p=1.0)
    ]),  # Removed bbox_params
    'color': A.Compose([
        A.Resize(*TARGET_SIZE),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)
    ]),  # Removed bbox_params
    'sharpen': A.Compose([
        A.Resize(*TARGET_SIZE),
        A.Lambda(image=sharpen_image)
    ])  # Removed bbox_params
}

# --- APPLY AUGMENTATIONS ---
for img_dir, class_name in DATASETS:  # Removed lbl_dir
    print(f"Augmenting: {class_name}")
    img_folder_train = os.path.join(OUTPUT_ROOT, 'train', class_name, 'images')
    img_files_train = glob.glob(os.path.join(img_folder_train, '*.jpg'))
    augmentation_count = 0

    for img_path in img_files_train:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)

        if img is None:
            print(f"[ERROR] Could not read image: {img_path}")
            continue

        try:
            # Exposure
            aug_exposure = augmentations['exposure'](image=img.copy())  # Removed bboxes, class_labels
            cv2.imwrite(os.path.join(img_folder_train, f"{fname}_exp.jpg"), aug_exposure['image'])
            augmentation_count += 1

            # Rotation variants
            for i in range(1, ROTATION_SAMPLES + 1):
                aug_rotation = augmentations['rotation'](image=img.copy())  # Removed bboxes, class_labels
                cv2.imwrite(os.path.join(img_folder_train, f"{fname}_rot{i}.jpg"), aug_rotation['image'])
                augmentation_count += 1

            # Scaling
            aug_scaling = augmentations['scaling'](image=img.copy())  # Removed bboxes, class_labels
            cv2.imwrite(os.path.join(img_folder_train, f"{fname}_scale.jpg"), aug_scaling['image'])
            augmentation_count += 1

            # Motion Blur
            aug_motion_blur = augmentations['motion_blur'](image=img.copy())  # Removed bboxes, class_labels
            cv2.imwrite(os.path.join(img_folder_train, f"{fname}_motion.jpg"), aug_motion_blur['image'])
            augmentation_count += 1

            # Color
            aug_color = augmentations['color'](image=img.copy())  # Removed bboxes, class_labels
            cv2.imwrite(os.path.join(img_folder_train, f"{fname}_color.jpg"), aug_color['image'])
            augmentation_count += 1

            # Sharpen
            aug_sharpen = augmentations['sharpen'](image=img.copy())  # Removed bboxes, class_labels
            cv2.imwrite(os.path.join(img_folder_train, f"{fname}_sharp.jpg"), aug_sharpen['image'])
            augmentation_count += 1

        except Exception as e:
            print(f"[ERROR] Augmentation failed for {img_path}: {e}")

    print(f"{class_name} augmentation complete. Total augmented images: {augmentation_count}")