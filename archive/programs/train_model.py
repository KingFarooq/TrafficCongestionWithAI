import os

# Define paths
image_dir = "/home/ubuntu/traffic/Real Model/Images/"
annotation_dir = "/home/ubuntu/traffic/Real Model/Annotations/"

# Get list of image and annotation files
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
annotation_files = [f for f in os.listdir(annotation_dir) if f.lower().endswith(".xml")]

# Extract base names without extensions and normalize case
image_basenames = {os.path.splitext(f)[0].strip().lower() for f in image_files}
annotation_basenames = {os.path.splitext(f)[0].strip().lower() for f in annotation_files}

# Find missing annotations
missing_annotations = image_basenames - annotation_basenames
missing_images = annotation_basenames - image_basenames

# Print results
if missing_annotations:
    print("\nImages missing annotations:")
    for missing in sorted(missing_annotations):
        print(f"{missing}.jpg (or corresponding format)")

if missing_images:
    print("\nAnnotations missing images:")
    for missing in sorted(missing_images):
        print(f"{missing}.xml")

if not missing_annotations and not missing_images:
    print("✅ All images have matching annotations!")
