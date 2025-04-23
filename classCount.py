import os
import xml.etree.ElementTree as ET

def count_classes(annotation_directory):
    classes = set()
    for filename in os.listdir(annotation_directory):
        if filename.lower().endswith('.xml'):
            annotation_path = os.path.join(annotation_directory, filename)
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                classes.add(class_name)
    return classes

# Define the annotation directory
annotation_directory = '/home/ubuntu/traffic/Real Model/Annotations/'

# Get the unique class labels
unique_classes = count_classes(annotation_directory)

# Count the number of classes
num_classes = len(unique_classes)
print(f"Unique classes: {unique_classes}")
print(f"Number of classes: {num_classes}")
