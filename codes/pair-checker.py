import os

# Allowed image file extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

# Your datasets
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

print("🔍 Checking for image files without matching labels...\n")

for image_dir, label_dir, class_name in DATASETS:
    print(f"📂 Dataset: {class_name}")
    missing_labels = []

    for file in os.listdir(image_dir):
        file_lower = file.lower()
        if any(file_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
            base_name = os.path.splitext(file)[0]
            label_path = os.path.join(label_dir, base_name + '.txt')
            if not os.path.exists(label_path):
                missing_labels.append(file)

    if missing_labels:
        print(f"⚠️  {len(missing_labels)} images in '{class_name}' don't have labels:")
        for img in missing_labels:
            print(f"  • {img}")
    else:
        print("✅ All images have corresponding labels.")

    print()

print("🧾 Label-check complete.")
