import os
import random

# Directories
label_dir = r"C:\Users\jazzb\ImageDetection-Yolov9\annotated\Alligator Cracks - Annotated"
image_dir = r"C:\Users\jazzb\ImageDetection-Yolov9\annotated\Alligator Cracks - Annotated\2 - Alligator Cracks - IMAGES"

# Image extensions to check
image_exts = [".jpg", ".jpeg", ".png"]

# Step 1: Collect all base filenames that have both a label and an image
valid_pairs = []

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        base_name = os.path.splitext(file)[0]
        label_path = os.path.join(label_dir, file)

        # Check for corresponding image in image_dir
        for ext in image_exts:
            image_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(image_path):
                valid_pairs.append((base_name, label_path, image_path))
                break

# Step 2: Randomly select 60 pairs
if len(valid_pairs) < 60:
    print(f"❌ Only found {len(valid_pairs)} valid pairs. Cannot delete 60.")
else:
    selected_pairs = random.sample(valid_pairs, 60)

    print("🟡 Starting deletion of 60 random image-label pairs...\n")

    for idx, (base_name, label_path, image_path) in enumerate(selected_pairs, 1):
        print(f"🔹 [{idx}/60] Deleting pair: {base_name}")

        # Delete label
        try:
            os.remove(label_path)
            print(f"   ✅ Deleted label: {label_path}")
        except Exception as e:
            print(f"   ❌ Error deleting label: {e}")

        # Delete image
        try:
            os.remove(image_path)
            print(f"   ✅ Deleted image: {image_path}")
        except Exception as e:
            print(f"   ❌ Error deleting image: {e}")

    print("\n✅ Deletion of 60 random pairs completed.")
