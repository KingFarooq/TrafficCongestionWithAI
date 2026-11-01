from PIL import Image
import os

# Set the path to your image directory and the desired size
input_folder = '/home/ubuntu/Pictures/No Watermark Photos'
output_folder = '/home/ubuntu/Pictures/Resized'
size = (256, 256)  # Set the desired size

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Resize all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = Image.open(os.path.join(input_folder, filename))
        img = img.resize(size, Image.LANCZOS)
        img.save(os.path.join(output_folder, filename))

print('All images resized successfully!')
