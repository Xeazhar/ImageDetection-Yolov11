import os
import cv2
import glob
import random
import albumentations as A

# --- CONFIGURATION ---
NUM_AUG_PER_IMAGE = 3  # Not used in this script
TARGET_SIZE = (640, 640)
SPLIT_RATIOS = {'train': 0.64, 'val': 0.16, 'test': 0.2}
OUTPUT_ROOT = './Road_Damage'
ROTATION_SAMPLES = 3

# Dataset configuration: (image_dir, label_dir, class_name)
DATASETS = [
    (
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Potholes - Annotated/1 - Potholes-20250429T144329Z-001/1 - Potholes',
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Potholes - Annotated',
        'potholes'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Cracks - Annotated/12 - Cracks',
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Cracks - Annotated',
        'crack_issues'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Alligator Cracks - Annotated/2 - Alligator Cracks - IMAGES',
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Alligator cracks - Annotated',
        'alligator_crack_issues'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Ravelling - Annotated/Ravelling - IMAGES',
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Ravelling - Annotated',
        'ravelling'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Open Manhole - Annotated/10 - Open Manhole - IMAGES',
        r'C:/Users/jazzb/ImageDetection-Yolov11/annotated/Open Manhole - Annotated',
        'open_manhole'
    )
]

# Create directory structure
for split in SPLIT_RATIOS:
    for _, _, class_name in DATASETS:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, class_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, split, class_name, 'labels'), exist_ok=True)

# --- YOLO Label Helpers ---
def read_yolo_labels(path):
    bboxes, classes = [], []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    bboxes.append([float(x) for x in parts[1:]])
                    classes.append(cls)
    return bboxes, classes

def write_yolo_labels(path, bboxes, classes):
    with open(path, 'w') as f:
        for cls, box in zip(classes, bboxes):
            f.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")

# --- SPLIT AND SAVE ORIGINALS ---
for img_dir, lbl_dir, class_name in DATASETS:
    imgs = glob.glob(os.path.join(img_dir, '*.[jJ][pP][gG]')) + glob.glob(os.path.join(img_dir, '*.[pP][nN][gG]'))
    print(f"[DEBUG] {class_name}: found {len(imgs)} images.")
    items = [(img, os.path.join(lbl_dir, os.path.splitext(os.path.basename(img))[0] + '.txt')) for img in imgs]

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
        for img_path, lbl_path in batch:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {img_path}: cannot read image.")
                continue
            img_res = cv2.resize(img, TARGET_SIZE)
            dst_img = os.path.join(OUTPUT_ROOT, split, class_name, 'images', f"{fname}.jpg")
            cv2.imwrite(dst_img, img_res)

            bboxes, classes = read_yolo_labels(lbl_path)
            dst_lbl = os.path.join(OUTPUT_ROOT, split, class_name, 'labels', f"{fname}.txt")
            write_yolo_labels(dst_lbl, bboxes, classes)

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
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)),

    'rotation': A.Compose([
        A.Resize(*TARGET_SIZE),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)),

    'scaling': A.Compose([
        A.Resize(*TARGET_SIZE),
        A.RandomScale(scale_limit=0.3, p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)),

    'motion_blur': A.Compose([
        A.Resize(*TARGET_SIZE),
        A.MotionBlur(blur_limit=(5, 15), p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)),

    'color': A.Compose([
        A.Resize(*TARGET_SIZE),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)),

    'sharpen': A.Compose([
        A.Resize(*TARGET_SIZE),
        A.Lambda(image=sharpen_image)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)),
}

# --- APPLY AUGMENTATIONS ---
for _, _, class_name in DATASETS:
    print(f"Augmenting: {class_name}")
    img_folder = os.path.join(OUTPUT_ROOT, 'train', class_name, 'images')
    lbl_folder = os.path.join(OUTPUT_ROOT, 'train', class_name, 'labels')
    img_files = glob.glob(os.path.join(img_folder, '*.jpg'))

    for img_path in img_files:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        lbl_path = os.path.join(lbl_folder, f"{fname}.txt")
        bboxes, classes = read_yolo_labels(lbl_path)
        if not bboxes:
            continue

        # Exposure
        aug = augmentations['exposure'](image=img, bboxes=bboxes, class_labels=classes)
        cv2.imwrite(os.path.join(img_folder, f"{fname}_exp.jpg"), aug['image'])
        write_yolo_labels(os.path.join(lbl_folder, f"{fname}_exp.txt"), aug['bboxes'], aug['class_labels'])

        # Rotation variants
        for i in range(1, ROTATION_SAMPLES + 1):
            aug = augmentations['rotation'](image=img, bboxes=bboxes, class_labels=classes)
            cv2.imwrite(os.path.join(img_folder, f"{fname}_rot{i}.jpg"), aug['image'])
            write_yolo_labels(os.path.join(lbl_folder, f"{fname}_rot{i}.txt"), aug['bboxes'], aug['class_labels'])

        # Scaling
        aug = augmentations['scaling'](image=img, bboxes=bboxes, class_labels=classes)
        cv2.imwrite(os.path.join(img_folder, f"{fname}_scale.jpg"), aug['image'])
        write_yolo_labels(os.path.join(lbl_folder, f"{fname}_scale.txt"), aug['bboxes'], aug['class_labels'])

        # Motion Blur
        aug = augmentations['motion_blur'](image=img, bboxes=bboxes, class_labels=classes)
        cv2.imwrite(os.path.join(img_folder, f"{fname}_motion.jpg"), aug['image'])
        write_yolo_labels(os.path.join(lbl_folder, f"{fname}_motion.txt"), aug['bboxes'], aug['class_labels'])

        # Color
        aug = augmentations['color'](image=img, bboxes=bboxes, class_labels=classes)
        cv2.imwrite(os.path.join(img_folder, f"{fname}_color.jpg"), aug['image'])
        write_yolo_labels(os.path.join(lbl_folder, f"{fname}_color.txt"), aug['bboxes'], aug['class_labels'])

        # Sharpen
        aug = augmentations['sharpen'](image=img, bboxes=bboxes, class_labels=classes)
        cv2.imwrite(os.path.join(img_folder, f"{fname}_sharp.jpg"), aug['image'])
        write_yolo_labels(os.path.join(lbl_folder, f"{fname}_sharp.txt"), aug['bboxes'], aug['class_labels'])

    print(f"{class_name} augmentation complete.")
