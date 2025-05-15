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

# Dataset configuration: (image_dir, class_name)
DATASETS = [
    (r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Potholes - Annotated/1 - Potholes-20250429T144329Z-001/1 - Potholes', 'potholes'),
    (r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Cracks - Annotated/12 - Cracks', 'crack_issues'),
    (r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Alligator Cracks - Annotated/2 - Alligator Cracks - IMAGES', 'alligator_crack_issues'),
    (r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Ravelling - Annotated/Ravelling - IMAGES', 'ravelling'),
    (r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Open Manhole - Annotated/10 - Open Manhole - IMAGES', 'open_manhole'),
]

# Create directory structure
for split in SPLIT_RATIOS:
    for _, class_name in DATASETS:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, class_name, 'images'), exist_ok=True)

# Sharpen function

def sharpen_image(image, **kwargs):
    laplacian = cv2.Laplacian(image, cv2.CV_16S, ksize=3)
    sharp = cv2.addWeighted(image, 0.7, cv2.convertScaleAbs(laplacian), 0.3, 0)
    return sharp

# Define augmentations dict (using Affine instead of ShiftScaleRotate)
augmentations = {
    'brightness_contrast': A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.6, p=0.7),
    'hue_saturation': A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.6),
    'shadow': A.RandomShadow(p=0.3),
    'rain': A.RandomRain(p=0.3),
    'horizontal_flip': A.HorizontalFlip(p=0.5),
    'rotate': A.Rotate(limit=25, p=0.5),
    'scale': A.RandomScale(scale_limit=0.4, p=0.5),
    'affine': A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=15, p=0.5),
    'motion_blur': A.MotionBlur(blur_limit=15, p=0.4),
    'sharpen': A.Lambda(image=sharpen_image, p=0.5)
}

# --- SPLIT AND SAVE ORIGINALS ---
def split_and_save():
    for img_dir, class_name in DATASETS:
        imgs = glob.glob(os.path.join(img_dir, '*.[jJ][pP][gG]')) + glob.glob(
            os.path.join(img_dir, '*.[pP][nN][gG]'))
        print(f"[INFO] {class_name}: found {len(imgs)} images.")

        random.seed(42)
        random.shuffle(imgs)
        total = len(imgs)
        test_count = int(total * SPLIT_RATIOS['test'])
        val_count = int(total * SPLIT_RATIOS['val'])

        splits = {
            'train': imgs[: total - test_count - val_count],
            'val': imgs[total - test_count - val_count: total - test_count],
            'test': imgs[total - test_count:]
        }

        for split, batch in splits.items():
            for img_path in batch:
                fname = os.path.splitext(os.path.basename(img_path))[0]
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARN] Skipping {img_path}: cannot read.")
                    continue
                img_res = cv2.resize(img, TARGET_SIZE)
                out_path = os.path.join(OUTPUT_ROOT, split, class_name, 'images', f"{fname}.jpg")
                cv2.imwrite(out_path, img_res)

# --- AUGMENT TRAINING IMAGES ---
def augment_train_set(class_name):
    print(f"[INFO] Augmenting class: {class_name}")
    folder = os.path.join(OUTPUT_ROOT, 'train', class_name, 'images')
    files = glob.glob(os.path.join(folder, '*.jpg'))
    count = 0

    for img_path in files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        fname = os.path.splitext(os.path.basename(img_path))[0]

        for name, aug in augmentations.items():
            if random.random() < aug.p:
                augmented = aug(image=img)['image']
                out_file = os.path.join(folder, f"{fname}_{name}.jpg")
                cv2.imwrite(out_file, augmented)
                count += 1

    print(f"[INFO] {class_name}: generated {count} augmented images.")

if __name__ == '__main__':
    split_and_save()
    for _, class_name in DATASETS:
        augment_train_set(class_name)
