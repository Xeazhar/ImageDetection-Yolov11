import os

# List of allowed image extensions
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

print("🔍 Starting cleanup of orphaned labels...\n")

for image_dir, label_dir, class_name in DATASETS:
    print(f"📂 Checking dataset: {class_name}")
    deleted_count = 0

    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            label_name = os.path.splitext(filename)[0]
            label_path = os.path.join(label_dir, filename)

            # Check if any image with this base name exists
            has_image = False
            for ext in IMAGE_EXTENSIONS:
                image_path = os.path.join(image_dir, label_name + ext)
                if os.path.exists(image_path):
                    has_image = True
                    break

            if not has_image:
                os.remove(label_path)
                deleted_count += 1
                print(f"❌ Deleted orphaned label: {label_path}")

    print(f"✅ Finished {class_name}: {deleted_count} orphaned labels removed.\n")

print("🎉 Cleanup complete.")
