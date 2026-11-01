from PIL import Image, ImageOps
import os

def correct_orientation(image_path, output_path):
    # Open the image
    image = Image.open(image_path)
    
    # Rotate the image to correct orientation
    # Automatically rotates based on EXIF data
    corrected_image = ImageOps.exif_transpose(image)
    
    # Save the corrected image
    corrected_image.save(output_path)
    
    print(f"Corrected image saved to {output_path}")

# Directories
input_directory = '/home/ubuntu/Pictures/Resized'
output_directory = '/home/ubuntu/Pictures/Upright'

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each image in the input directory
for filename in os.listdir(input_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        correct_orientation(image_path, output_path)
