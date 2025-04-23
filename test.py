import os
import xml.etree.ElementTree as ET

def list_classes(annotation_directory):
    class_names = set()  # Use a set to avoid duplicates

    for filename in os.listdir(annotation_directory):
        if filename.lower().endswith('.xml'):
            annotation_path = os.path.join(annotation_directory, filename)
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_text = obj.find('name').text
                class_names.add(class_text)

    return class_names

# Directory containing annotation files
annotation_directory = '/home/ubuntu/traffic/Real Model/Annotations'
class_names = list_classes(annotation_directory)

# Print class names
for class_name in class_names:
    print(class_name)
