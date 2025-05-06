import os
import cv2
import glob
import random
import albumentations as A

# --- CONFIGURATION ---
NUM_AUG_PER_IMAGE = 5
TARGET_SIZE = (640, 640)
SPLIT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}

# Project name
OUTPUT_ROOT = './Road_Damage-OBJD'

# Each tuple: (image_dir, label_dir, class_name)
DATASETS = [
    (
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Potholes - Annotated/1 - Potholes-20250429T144329Z-001/1 - Potholes',
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Potholes - Annotated',
        'potholes'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Cracks - Annotated/12 - Cracks',
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Cracks - Annotated',
        'crack_issues'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Alligator cracks - Annotated/2 - Alligator Cracks - IMAGES',
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Alligator cracks - Annotated',
        'alligator_crack_issues'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Ravelling - Annotated/Ravelling - IMAGES',
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Ravelling - Annotated',
        'ravelling'
    )
    ,
    (
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Pumping and Depression - Annotated/05-Pumping and Depression - IMAGES',
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Pumping and Depression - Annotated',
        'pumping_and_depression'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Open Manhole - Annotated/10 - Open Manhole - IMAGES',
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Open Manhole - Annotated',
        'open_manhole'
    )
]

# Create folder structure: split/images and split/labels for combined output
for split in SPLIT_RATIOS:
    os.makedirs(os.path.join(OUTPUT_ROOT, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, split, 'labels'), exist_ok=True)

# Helper functions
def read_yolo_labels(path):
    bboxes, classes = [], []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                classes.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
    return bboxes, classes


def write_yolo_labels(path, bboxes, classes):
    with open(path, 'w') as f:
        for cls, box in zip(classes, bboxes):
            f.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")

# Gather, split, and save
for img_dir, lbl_dir, class_name in DATASETS:
    # Gather all images
    imgs = glob.glob(os.path.join(img_dir, '*.[jJ][pP][gG]')) + glob.glob(os.path.join(img_dir, '*.[pP][nN][gG]'))
    items = [(img, os.path.join(lbl_dir, os.path.splitext(os.path.basename(img))[0] + '.txt')) for img in imgs]

    # Shuffle and split
    random.seed(42)
    random.shuffle(items)
    total = len(items)
    train_end = int(total * SPLIT_RATIOS['train'])
    val_end = train_end + int(total * SPLIT_RATIOS['val'])
    split_items = {
        'train': items[:train_end],
        'val':   items[train_end:val_end],
        'test':  items[val_end:]
    }

    # Process each split
    for split, batch in split_items.items():
        for img_path, lbl_path in batch:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            # Read and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_res = cv2.resize(img, TARGET_SIZE)
            # Save processed image to combined split
            dst_img = os.path.join(OUTPUT_ROOT, split, 'images', f"{fname}.jpg")
            cv2.imwrite(dst_img, img_res)
            # Copy label
            bboxes, classes = read_yolo_labels(lbl_path)
            dst_lbl = os.path.join(OUTPUT_ROOT, split, 'labels', f"{fname}.txt")
            write_yolo_labels(dst_lbl, bboxes, classes)

# Augmentation for training split (with debug logs)
transform = A.Compose([
    A.Resize(*TARGET_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
    A.CoarseDropout(p=0.3),  # using default parameters for coarse dropout
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.HueSaturationValue(hue_shift_limit=16, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    A.ToGray(p=0.15),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.RandomRain(blur_value=3, p=0.2),
    A.RandomFog(fog_coef=(0.1, 0.3), p=0.2),
    A.RandomSunFlare(flare_roi=(0,0.5,1,1), p=0.2),
    A.CoarseDropout(p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

# Process augmentation for 'train' split
train_folder = os.path.join(OUTPUT_ROOT, 'train', 'images')
label_folder = os.path.join(OUTPUT_ROOT, 'train', 'labels')
train_imgs = glob.glob(os.path.join(train_folder, '*.jpg'))
print(f"Found {len(train_imgs)} images for augmenting in {train_folder}")
total_aug = 0
for img_path in train_imgs:
    fname = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Processing image: {fname}")
    img = cv2.imread(img_path)
    lbl_path = os.path.join(label_folder, f"{fname}.txt")
    bboxes, classes = read_yolo_labels(lbl_path)
    print(f"  Loaded {len(bboxes)} bboxes from {lbl_path}")
    if not bboxes:
        print(f"  Skipping {fname}: no bounding boxes")
        continue
    for i in range(1, NUM_AUG_PER_IMAGE+1):
        try:
            aug = transform(image=img, bboxes=bboxes, class_labels=classes)
            out_img = os.path.join(train_folder, f"{fname}_aug{i}.jpg")
            out_lbl = os.path.join(label_folder, f"{fname}_aug{i}.txt")
            cv2.imwrite(out_img, aug['image'])
            write_yolo_labels(out_lbl, aug['bboxes'], aug['class_labels'])
            total_aug += 1
            print(f"  Saved augmentation {i} for {fname}: {out_img} ({len(aug['bboxes'])} bboxes)")
        except Exception as e:
            print(f"  Error augmenting {fname} aug{i}: {e}")
print(f"Completed augmentations for training data: {total_aug} files created")
