import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

# --- CONFIGURATION ---
# Update these if your dataset paths changed
IMAGES_DIR = Path(r"C:\Users\fmanw\Desktop\old laptop\traffic\Real Model\RoadMark-binary\train\ortho") 
MASKS_DIR = Path(r"C:\Users\fmanw\Desktop\old laptop\traffic\Real Model\RoadMark-binary\train\mask")

OUT_IMAGES_DIR = Path(r"C:\Users\fmanw\Desktop\old laptop\traffic\Real Model\data\Images")
OUT_ANN_DIR = Path(r"C:\Users\fmanw\Desktop\old laptop\traffic\Real Model\data\Annotations")

OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUT_ANN_DIR.mkdir(parents=True, exist_ok=True)

# Your original 20 classes
CLASSES = [
    "Straight Arrow", "Left Turn Arrow", "Right Turn Arrow", "U-Turn Arrow",
    "Straight+Left Arrow", "Straight+Right Arrow", "Left+Right Arrow",
    "Crosswalk Markings", "Stop Line", "Solid White Line", "Solid Yellow Line",
    "Dashed White Line", "Dashed Yellow Line", "Double Solid White Line",
    "Double Solid Yellow Line", "Chevron Markings", "Bike Lane Markings",
    "Bus Lane Markings", "Yield Markings", "Speed Limit Markings"
]

def analyze_mask_and_image(mask, image):
    labels = []
    white_pixels = cv2.countNonZero(mask)
    if white_pixels < 50: 
        return labels

    # Convert image to HSV to easily find yellow
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        # Isolate this specific shape in the image to check its color
        shape_mask = np.zeros_like(mask)
        cv2.drawContours(shape_mask, [cnt], -1, 255, -1)
        
        # Calculate mean hue in the shape
        mean_hue = cv2.mean(hsv[:, :, 0], mask=shape_mask)[0]
        is_yellow = (15 < mean_hue < 35) # Yellow hue range in OpenCV
        
        # HEURISTICS based on shape and color
        if w > (h * 3) and h > 3:
            labels.append("Stop Line")
        elif (w > (h * 3) or h > (w * 3)):
            # Long thin lines
            if is_yellow:
                labels.append("Solid Yellow Line")
            else:
                labels.append("Solid White Line")
        elif area > 150 and 10 < w < 50 and 10 < h < 50:
            # Compact complex shapes (Arrows)
            # We cannot tell direction from a mask, so we guess Left
            labels.append("Left Turn Arrow")
            
    return list(set(labels)) # Remove duplicates

def create_xml(filename, labels):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    for label in labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        # Dummy box
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "0"
        ET.SubElement(bndbox, "ymin").text = "0"
        ET.SubElement(bndbox, "xmax").text = "10"
        ET.SubElement(bndbox, "ymax").text = "10"
    return ET.ElementTree(root)

# --- MAIN LOOP ---
print("Processing RoadMark Dataset for 20 classes...")
count = 0
for img_name in os.listdir(IMAGES_DIR):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    mask_name = os.path.splitext(img_name)[0] + ".png" 
    img_path = IMAGES_DIR / img_name
    mask_path = MASKS_DIR / mask_name
    
    if not mask_path.exists():
        continue

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        continue
        
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    labels = analyze_mask_and_image(thresh, img)
    
    if labels:
        shutil.copy(img_path, OUT_IMAGES_DIR / img_name)
        xml_tree = create_xml(img_name, labels)
        xml_tree.write(OUT_ANN_DIR / (os.path.splitext(img_name)[0] + ".xml"))
        count += 1
        if count % 500 == 0:
            print(f"Processed {count} images...")

print(f"Done! Converted {count} images.")