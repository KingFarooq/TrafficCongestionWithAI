import os
import numpy as np
import xml.etree.ElementTree as ET

CLASSES = [
    "Straight Arrow", "Left Turn Arrow", "Right Turn Arrow", "U-Turn Arrow",
    "Straight+Left Arrow", "Straight+Right Arrow", "Left+Right Arrow",
    "Crosswalk Markings", "Stop Line", "Solid White Line", "Solid Yellow Line",
    "Dashed White Line", "Dashed Yellow Line", "Double Solid White Line",
    "Double Solid Yellow Line", "Chevron Markings", "Bike Lane Markings",
    "Bus Lane Markings", "Yield Markings", "Speed Limit Markings"
]

ANNOTATIONS_DIR = "/home/ubuntu/traffic/Real Model/Annotations/"
counts = np.zeros(20)

for fname in os.listdir(ANNOTATIONS_DIR):
    if fname.endswith('.xml'):
        try:
            tree = ET.parse(os.path.join(ANNOTATIONS_DIR, fname))
            for obj in tree.getroot().findall('object'):
                name = obj.find('name').text
                if name in CLASSES:
                    counts[CLASSES.index(name)] += 1
        except:
            pass

print("Class distribution:")
for i, (name, count) in enumerate(zip(CLASSES, counts)):
    bar = "█" * int(count / max(counts) * 30) if max(counts) > 0 else ""
    print(f"  {name:<30} {int(count):>4}  {bar}")
print(f"\nTotal annotations: {int(sum(counts))}")
print(f"Classes with zero examples: {sum(1 for c in counts if c == 0)}/20")