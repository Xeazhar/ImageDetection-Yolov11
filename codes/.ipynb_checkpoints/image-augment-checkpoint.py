import os
import cv2
import glob
import random
import albumentations as A

# --- CONFIGURATION ---
NUM_AUG_PER_IMAGE = 5
TARGET_SIZE = (640, 640)
SPLIT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}

data_name = 'combined_road_damages'
# Each tuple: (image_dir, label_dir)
DATASETS = [
    (
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Potholes - Annotated/1 - Potholes-20250429T144329Z-001/1 - Potholes',
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Potholes - Annotated'
    ),
    (
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Joints - Annotated/INADEQUATE SEALANT IN JOINS-20250430T142240Z-001/11 - NO-INADEQUATE SEALANT IN JOINS',
        r'C:/Users/jazzb/ImageDetection-Yolov9/annotated/Joints - Annotated'
    )
]

# Define roots
PROCESSED_ROOT = os.path.join('.', 'processed', data_name)
AUGMENTED_ROOT = os.path.join('.', 'augmented', data_name)

# Create output directories
for root, splits in [(PROCESSED_ROOT, SPLIT_RATIOS.keys()), (AUGMENTED_ROOT, ['train'])]:
    for split in splits:
        os.makedirs(os.path.join(root, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(root, split, 'labels'), exist_ok=True)
        print(f"Directory ready: {root}/{split}")

# Helpers

def read_yolo_labels(path):
    bboxes, classes = [], []
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                classes.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
        print(f"Read {len(bboxes)} boxes from {path}")
    except FileNotFoundError:
        print(f"Label file not found: {path}")
    return bboxes, classes


def write_yolo_labels(path, bboxes, classes):
    with open(path, 'w') as f:
        for cls, box in zip(classes, bboxes):
            f.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")

# Gather images and labels
all_items = []  # list of (img_path, lbl_path)
for img_dir, lbl_dir in DATASETS:
    imgs = glob.glob(os.path.join(img_dir, '*.[jJ][pP][gG]')) + glob.glob(os.path.join(img_dir, '*.[pP][nN][gG]'))
    for img in imgs:
        fname = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(lbl_dir, f"{fname}.txt")
        all_items.append((img, lbl))
print(f"Found {len(all_items)} images across {len(DATASETS)} datasets.")

# Shuffle and split
global random_seed
random.seed(42)
random.shuffle(all_items)
total = len(all_items)
train_end = int(total * SPLIT_RATIOS['train'])
val_end = train_end + int(total * SPLIT_RATIOS['val'])
splits = {
    'train': all_items[:train_end],
    'val':   all_items[train_end:val_end],
    'test':  all_items[val_end:]
}
for split, items in splits.items():
    print(f"{split}: {len(items)} items")

# Preprocess: resize, normalize, save
for split, items in splits.items():
    for img_path, lbl_path in items:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None: continue
        img_res = cv2.resize(img, TARGET_SIZE)
        img_norm = (img_res.astype('float32')/255.0 * 255).astype('uint8')
        out_img = os.path.join(PROCESSED_ROOT, split, 'images', f"{fname}.jpg")
        cv2.imwrite(out_img, img_norm)
        bboxes, classes = read_yolo_labels(lbl_path)
        out_lbl = os.path.join(PROCESSED_ROOT, split, 'labels', f"{fname}.txt")
        write_yolo_labels(out_lbl, bboxes, classes)
print("Preprocessing done.")

# Augmentation pipeline
transform = A.Compose([
    # Spatial
    A.Resize(*TARGET_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),

    # Photometric and Noise
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.HueSaturationValue(hue_shift_limit=16, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    A.ToGray(p=0.15),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.RandomRain(blur_value=3, p=0.2),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
    A.RandomSunFlare(flare_roi=(0, 0.5, 1, 1), angle_lower=0.3, p=0.2),


], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


# Augment training set
train_items = splits['train']
total_aug = 0
for img_path, lbl_path in train_items:
    fname = os.path.splitext(os.path.basename(img_path))[0]
    proc_img_path = os.path.join(PROCESSED_ROOT, 'train', 'images', f"{fname}.jpg")
    img = cv2.imread(proc_img_path)
    bboxes, classes = read_yolo_labels(os.path.join(PROCESSED_ROOT, 'train', 'labels', f"{fname}.txt"))
    for i in range(1, NUM_AUG_PER_IMAGE + 1):
        aug = transform(image=img, bboxes=bboxes, class_labels=classes)
        out_img = os.path.join(AUGMENTED_ROOT, 'train', 'images', f"{fname}_aug{i}.jpg")
        out_lbl = os.path.join(AUGMENTED_ROOT, 'train', 'labels', f"{fname}_aug{i}.txt")
        cv2.imwrite(out_img, aug['image'])
        write_yolo_labels(out_lbl, aug['bboxes'], aug['class_labels'])
        total_aug += 1
        print(f"Aug {i}/{NUM_AUG_PER_IMAGE} for {fname} saved ({len(aug['bboxes'])} bboxes)")
print(f"Total augmented: {total_aug}")
