import os
import hashlib

def calculate_hash(image_path, chunk_size=4096):
    hash_obj = hashlib.md5()
    with open(image_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def delete_duplicates(directory):
    hashes = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(directory, filename)
            image_hash = calculate_hash(image_path)
            
            if image_hash in hashes:
                print(f"Deleting duplicate image: {image_path}")
                os.remove(image_path)
            else:
                hashes[image_hash] = filename

# Directory containing images
image_directory = '/home/ubuntu/Pictures/Other pictures/'

# Delete duplicates
delete_duplicates(image_directory)
