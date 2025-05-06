import os

# List of base filenames (without extensions)
files_to_remove = [
    "11 - Longitudinal Joint (2)",
    "11 - Longitudinal Joint (3)",
    "12 - Longitudinal Crack (RL)(1)",
    "12 - Reflection Crack (CL)(2)",
    "12 - Reflection Crack (CL)",
    "12 - Reflection Crack (Longitudinal Joint)",
    "12 - Reflection Crack (Weakened Plane Joint)",
    "12 - Reflection Crack",
    "12- Cracks (Longitudinal Crack) (RL)",
    "12 -Longitudinal Crack",
]

# Directories
label_dir = r"C:\Users\jazzb\ImageDetection-Yolov9\annotated\Cracks - Annotated"
image_dir = r"C:\Users\jazzb\ImageDetection-Yolov9\annotated\Cracks - Annotated\12 - Cracks"

# Image extensions to check
image_exts = [".jpg", ".jpeg", ".png"]

print("🟡 Starting file deletion process...\n")

for idx, base_name in enumerate(files_to_remove, 1):
    print(f"🔹 [{idx}/{len(files_to_remove)}] Processing: {base_name}")

    # --- Label ---
    label_path = os.path.join(label_dir, base_name + ".txt")
    if os.path.exists(label_path):
        os.remove(label_path)
        print(f"   ✅ Deleted label: {label_path}")
    else:
        print(f"   ⚠️ Label not found: {label_path}")
    
    # --- Image ---
    image_deleted = False
    for ext in image_exts:
        image_path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"   ✅ Deleted image: {image_path}")
            image_deleted = True
            break
    if not image_deleted:
        print(f"   ⚠️ Image not found: {base_name} (tried {', '.join(image_exts)})")

print("\n✅ File deletion process completed.")
