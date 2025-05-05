import os

# Define the old class IDs and new class IDs for potholes
old_to_new_class_mapping = {
    1.0: 1,  # Convert class ID 1.0 to 1

}

# Define the path to your potholes dataset's labels
root_path = r'C:\Users\jazzb\ImageDetection-Yolov9\annotated\Cracks - Annotated'

# Process all .txt files directly in the potholes folder (without any subfolders)
if os.path.exists(root_path):
    for filename in os.listdir(root_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(root_path, filename)
            new_lines = []

            with open(filepath, 'r') as file:
                lines = file.readlines()

            # Process each line in the label file
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # Skip malformed lines

                # Check if the class ID is in the old_to_new_class_mapping
                old_class_id = float(parts[0])  # Handle class ID as float
                if old_class_id in old_to_new_class_mapping:
                    # Replace with the mapped class ID (as integer)
                    parts[0] = str(int(old_to_new_class_mapping[old_class_id]))  # Convert new class ID to integer

                new_line = ' '.join(parts)
                new_lines.append(new_line)

            # Save the modified label file
            with open(filepath, 'w') as file:
                file.write('\n'.join(new_lines))

            print(f"Updated: {filepath}")
else:
    print(f"Label folder not found: {root_path}")
