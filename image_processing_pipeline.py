import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Enhanced Configuration with All Restrictions
CLASSES = [
    "Straight Arrow", 
    "Left Turn Arrow", 
    "Right Turn Arrow", 
    "U-Turn Arrow",
    "Straight+Left Arrow",
    "Straight+Right Arrow",
    "Left+Right Arrow",
    "Crosswalk Markings",
    "Stop Line", 
    "Solid White Line", 
    "Solid Yellow Line",
    "Dashed White Line",
    "Dashed Yellow Line",
    "Double Solid White Line",
    "Double Solid Yellow Line",
    "Chevron Markings",
    "Bike Lane Markings",
    "Bus Lane Markings",
    "Yield Markings",
    "Speed Limit Markings"
]

# Directory paths as specified
IMAGE_DIR = "/home/ubuntu/traffic/Real Model/Images/"
ANNOTATIONS_DIR = "/home/ubuntu/traffic/Real Model/Annotations/"
TEST_IMAGES_DIR = "/home/ubuntu/traffic/Real Model/Test Images/"
MODEL_PATH = "/home/ubuntu/traffic/trained_model.keras"
OUTPUT_DIR = "/home/ubuntu/traffic/Real Model/Modified_Images/"

# Comprehensive Restrictions List (40+ options)
RESTRICTIONS = [
    # Physical changes
    "no new lanes", "no widening", "no construction", 
    "no road elevation changes", "no underpasses", "no overpasses",
    "no intersection modifications", "no median changes",
    "no curb modifications", "no sidewalk changes",
    
    # Lane types
    "no bus lanes", "no bike lanes", "no HOV lanes", 
    "no turning lanes", "no reversible lanes", "no shoulder conversions",
    
    # Markings
    "no new road markings", "no lane marking changes",
    "no crosswalk modifications", "no arrow modifications",
    "no stop line changes", "no yield markings",
    
    # Traffic control
    "no new traffic signals", "no signal timing changes",
    "no stop sign additions", "no yield sign additions",
    "no speed limit changes", "no school zone modifications",
    "no pedestrian signals", "no flashing beacons",
    
    # Technology
    "no smart traffic systems", "no surveillance cameras",
    "no vehicle detection systems", "no dynamic signage",
    "no license plate readers", "no traffic sensors",
    
    # Policy
    "no congestion pricing", "no parking changes",
    "no loading zone changes", "no transit priority",
    "no truck restrictions", "no delivery zones",
    
    # Environmental
    "no tree removal", "no landscaping changes",
    "no drainage modifications", "no lighting changes",
    "no sound barrier changes", "no wildlife crossings"
]

# Complete Solution Set with Metrics
SOLUTIONS = {
    # Physical modifications
    "lane_additions": [
        {"name": "Add general purpose lane", "impact": 0.8, "cost": 0.8, "safety": 0.6, "type": "physical"},
        {"name": "Add auxiliary lane", "impact": 0.7, "cost": 0.7, "safety": 0.7, "type": "physical"},
        {"name": "Add climbing lane", "impact": 0.6, "cost": 0.6, "safety": 0.8, "type": "physical"}
    ],
    "lane_conversions": [
        {"name": "Convert to HOV lane", "impact": 0.7, "cost": 0.5, "safety": 0.7, "type": "physical"},
        {"name": "Convert to bus lane", "impact": 0.6, "cost": 0.4, "safety": 0.8, "type": "physical"},
        {"name": "Convert to bike lane", "impact": 0.5, "cost": 0.3, "safety": 0.9, "type": "physical"},
        {"name": "Convert shoulder to travel lane", "impact": 0.6, "cost": 0.4, "safety": 0.5, "type": "physical"}
    ],
    "intersection_improvements": [
        {"name": "Add roundabout", "impact": 0.9, "cost": 0.9, "safety": 0.9, "type": "physical"},
        {"name": "Add dedicated turn lanes", "impact": 0.8, "cost": 0.7, "safety": 0.8, "type": "physical"},
        {"name": "Add channelization islands", "impact": 0.7, "cost": 0.6, "safety": 0.8, "type": "physical"},
        {"name": "Add pedestrian refuge island", "impact": 0.5, "cost": 0.5, "safety": 0.9, "type": "physical"}
    ],
    "grade_separations": [
        {"name": "Add pedestrian overpass", "impact": 0.4, "cost": 0.7, "safety": 0.9, "type": "physical"},
        {"name": "Add pedestrian underpass", "impact": 0.4, "cost": 0.8, "safety": 0.9, "type": "physical"},
        {"name": "Add vehicle overpass", "impact": 0.9, "cost": 0.95, "safety": 0.8, "type": "physical"}
    ],
    
    # Operational improvements
    "signal_optimizations": [
        {"name": "Optimize signal timing", "impact": 0.7, "cost": 0.3, "safety": 0.7, "type": "operational"},
        {"name": "Add adaptive signals", "impact": 0.8, "cost": 0.6, "safety": 0.8, "type": "operational"},
        {"name": "Add pedestrian signals", "impact": 0.5, "cost": 0.4, "safety": 0.9, "type": "operational"},
        {"name": "Coordinate signal systems", "impact": 0.8, "cost": 0.5, "safety": 0.8, "type": "operational"}
    ],
    "marking_improvements": [
        {"name": "Refresh lane markings", "impact": 0.5, "cost": 0.2, "safety": 0.7, "type": "operational"},
        {"name": "Add turn arrows", "impact": 0.6, "cost": 0.3, "safety": 0.7, "type": "operational"},
        {"name": "Add bike lane markings", "impact": 0.5, "cost": 0.3, "safety": 0.8, "type": "operational"},
        {"name": "Add crosswalk markings", "impact": 0.4, "cost": 0.2, "safety": 0.9, "type": "operational"}
    ],
    "smart_systems": [
        {"name": "Add traffic cameras", "impact": 0.6, "cost": 0.5, "safety": 0.7, "type": "operational"},
        {"name": "Implement smart parking", "impact": 0.5, "cost": 0.6, "safety": 0.6, "type": "operational"},
        {"name": "Add variable message signs", "impact": 0.5, "cost": 0.5, "safety": 0.7, "type": "operational"},
        {"name": "Install vehicle detection", "impact": 0.6, "cost": 0.4, "safety": 0.8, "type": "operational"}
    ],
    "policy_changes": [
        {"name": "Implement congestion pricing", "impact": 0.7, "cost": 0.3, "safety": 0.7, "type": "policy"},
        {"name": "Add transit priority", "impact": 0.6, "cost": 0.4, "safety": 0.8, "type": "policy"},
        {"name": "Create delivery zones", "impact": 0.5, "cost": 0.2, "safety": 0.7, "type": "policy"},
        {"name": "Adjust parking regulations", "impact": 0.4, "cost": 0.1, "safety": 0.6, "type": "policy"}
    ]
}

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

def get_input(prompt, input_type=str, options=None, default=None):
    """Helper function to get user input with validation"""
    while True:
        try:
            response = input(prompt).strip()
            if not response and default is not None:
                return default
            if response.upper() == 'N/A':
                return None
            
            if input_type == bool:
                if response.lower() in ['y', 'yes']:
                    return True
                elif response.lower() in ['n', 'no']:
                    return False
                else:
                    print("Please answer 'y', 'n', or 'N/A'")
                    continue
            
            if options:
                if response.lower() in [opt.lower() for opt in options]:
                    return next(opt for opt in options if opt.lower() == response.lower())
                else:
                    print(f"Please choose from: {', '.join(options)} or 'N/A'")
                    continue
            
            return input_type(response)
        except ValueError:
            print(f"Please enter a valid {input_type.__name__} or 'N/A'")

def get_road_details():
    """Collect comprehensive road information from user"""
    print("\n=== Road Information Survey ===")
    print("(For each question, answer or type 'N/A' if not applicable)\n")
    
    print("Available Restrictions (40+ options):")
    # Display restrictions in numbered columns for better readability
    for i in range(0, len(RESTRICTIONS), 4):
        row = RESTRICTIONS[i:i+4]
        print("  ".join(f"{i+j+1}. {item:<30}" for j, item in enumerate(row)))
    print()
    
    details = {
        'name': get_input("1. Road name: ", default="Unnamed Road"),
        'length': get_input("2. Road length (meters): ", float),
        'width': get_input("3. Road width (meters): ", float),
        'lanes': get_input("4. Number of lanes: ", int),
        'current_cars': get_input("5. Current cars on road: ", int),
        'capacity': get_input("6. Maximum car capacity: ", int),
        'avg_speed': get_input("7. Average speed (km/h): ", int),
        'congestion_times': get_input("8. Peak congestion times (e.g., '7-9 AM, 4-6 PM'): "),
        'restrictions': [],
        'special_events': [
            e.strip() 
            for e in get_input("10. Special events (comma separated): ", str).split(',')
            if e.strip()
        ] if get_input("Any special events? (y/n/N/A): ", bool) else [],
        'has_crosswalk': get_input("11. Pedestrian crossing? (y/n/N/A): ", bool),
        'has_bikelane': get_input("12. Bike lane? (y/n/N/A): ", bool),
        'condition': get_input("13. Road condition (Good/Fair/Poor/N/A): ", str, ['Good', 'Fair', 'Poor'])
    }
    
    # Handle restrictions
    if get_input("9. Any restrictions? (y/n/N/A): ", bool):
        print("\nSelect restrictions by number (comma separated):")
        print("Example: '1,3,5' for no new lanes, no construction, no HOV lanes")
        
        while True:
            choices = input("Enter restriction numbers: ").strip()
            if choices.upper() == 'N/A':
                break
                
            try:
                selected = [int(c.strip()) for c in choices.split(',') if c.strip()]
                valid_choices = [i for i in selected if 1 <= i <= len(RESTRICTIONS)]
                details['restrictions'] = [RESTRICTIONS[i-1] for i in valid_choices]
                break
            except ValueError:
                print(f"Please enter numbers between 1-{len(RESTRICTIONS)} separated by commas")
    
    return details

def generate_sample_data(num_samples=50):
    """Generate synthetic training data if no real data exists"""
    print("\nGenerating sample training data...")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    
    for i in range(num_samples):
        img = Image.new('RGB', (224, 224), color=(100, 150, 100))
        draw = ImageDraw.Draw(img)
        elements = random.sample(CLASSES, random.randint(1, 4))
        annotations = []
        
        for element in elements:
            if "Arrow" in element:
                x, y = random.randint(50, 150), random.randint(50, 150)
                arrow = "→" if "Right" in element else ("←" if "Left" in element else "↑")
                if "+" in element:
                    arrow = "↖" if "Left" in element else "↗"
                draw.text((x, y), arrow, fill=(255, 255, 0), font=ImageFont.load_default(size=20))
                annotations.append(f"<object><name>{element}</name><bndbox><xmin>{x}</xmin><ymin>{y}</ymin><xmax>{x+20}</xmax><ymax>{y+20}</ymax></bndbox></object>")
            elif "Line" in element or "Markings" in element:
                y = random.randint(80, 150)
                color = (255, 255, 255) if "White" in element else (255, 255, 0)
                width = 3 if "Solid" in element else [5, 10]  # Dashed pattern
                if isinstance(width, list):
                    for x in range(20, 200, 15):
                        draw.line([(x, y), (x+10, y)], fill=color, width=3)
                else:
                    draw.line([(20, y), (200, y)], fill=color, width=width)
                annotations.append(f"<object><name>{element}</name><bndbox><xmin>20</xmin><ymin>{y-5}</ymin><xmax>200</xmax><ymax>{y+5}</ymax></bndbox></object>")
            elif element == "Crosswalk Markings":
                for y in range(100, 120, 5):
                    draw.line([(50, y), (150, y)], fill=(255, 255, 255), width=2)
                annotations.append(f"<object><name>{element}</name><bndbox><xmin>50</xmin><ymin>100</ymin><xmax>150</xmax><ymax>120</ymax></bndbox></object>")
        
        img_path = os.path.join(IMAGE_DIR, f"sample_{i}.jpg")
        img.save(img_path)
        
        ann_path = os.path.join(ANNOTATIONS_DIR, f"sample_{i}.xml")
        with open(ann_path, 'w') as f:
            f.write(f"""<annotation>
                <filename>sample_{i}.jpg</filename>
                {"".join(annotations)}
            </annotation>""")

def load_images_and_labels():
    """Load images and corresponding labels"""
    images = []
    labels = []
    
    if not os.listdir(IMAGE_DIR) or not os.listdir(ANNOTATIONS_DIR):
        generate_sample_data()
    
    print("\nLoading training data...")
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(IMAGE_DIR, filename)
            base_name = os.path.splitext(filename)[0]
            ann_path = os.path.join(ANNOTATIONS_DIR, f"{base_name}.xml")

            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.resize(image, (224, 224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255.0
                images.append(image)
            except Exception as e:
                print(f"Error loading image {filename}: {str(e)}")
                continue

            label_vector = np.zeros(len(CLASSES))
            if os.path.exists(ann_path):
                try:
                    tree = ET.parse(ann_path)
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text
                        if class_name in CLASSES:
                            label_vector[CLASSES.index(class_name)] = 1
                except Exception as e:
                    print(f"Error parsing XML {ann_path}: {str(e)}")
            labels.append(label_vector)
    
    print(f"Loaded {len(images)} images with {len(labels)} annotations")
    return np.array(images), np.array(labels)

def build_and_train_model():
    """Build and train the CNN model"""
    X, Y = load_images_and_labels()
    split_idx = int(0.8 * len(X))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val, Y_val = X[split_idx:], Y[split_idx:]

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASSES), activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(0.0001),
        loss='binary_crossentropy',
        metrics=['AUC', 'accuracy']
    )

    print("\nTraining model...")
    model.fit(
        datagen.flow(X_train, Y_train, batch_size=16),
        validation_data=(X_val, Y_val),
        epochs=20,
        verbose=1
    )
    
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def predict_traffic_elements(image_path, model):
    """Predict traffic elements in an image with detailed reporting"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)
        
        predictions = model.predict(img_expanded, verbose=0)[0]
        
        # Get all detections with confidence > 30%
        detected = [(CLASSES[i], float(predictions[i])) 
                   for i in range(len(CLASSES)) if predictions[i] > 0.3]
        
        # Sort by confidence (descending)
        detected.sort(key=lambda x: x[1], reverse=True)
        
        return detected, img_rgb
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return [], None

def calculate_solution_score(solution, congestion_level, road_info):
    """Calculate a weighted score for each potential solution"""
    # Base score components
    impact_weight = 0.5
    safety_weight = 0.3
    cost_weight = 0.2
    
    # Adjust weights based on congestion level
    if congestion_level == "Severely Congested":
        impact_weight = 0.6
        safety_weight = 0.2
    elif congestion_level == "Congested":
        impact_weight = 0.55
        safety_weight = 0.25
    
    # Calculate base score
    score = (solution['impact'] * impact_weight + 
             solution['safety'] * safety_weight + 
             (1 - solution['cost']) * cost_weight)
    
    # Apply modifiers based on road characteristics
    # Check if 'has_crosswalk' is not None before using it
    if road_info['has_crosswalk'] and "pedestrian" in solution['name'].lower():
        score *= 1.1
        
    # Check if 'has_bikelane' is not None
    if road_info['has_bikelane'] and "bike" in solution['name'].lower():
        score *= 1.1
        
    # Check if 'lanes' is not None before comparison
    if road_info['lanes'] is not None and road_info['lanes'] <= 2 and "lane" in solution['name'].lower():
        score *= 1.15
        
    # Check if 'congestion_times' is not None and not empty
    if road_info['congestion_times'] and "signal" in solution['name'].lower():
        score *= 1.1
        
    return score

def analyze_traffic(road_info, detected_elements):
    """Comprehensive traffic analysis with all restrictions considered"""
    congestion = "Not Congested"
    reasons = []
    congestion_score = 0
    
    # Calculate congestion score (0-1 scale)
    congestion_score = 0

    # Check if 'current_cars' and 'capacity' are not None
    if road_info['current_cars'] is not None and road_info['capacity'] is not None:
        capacity_ratio = road_info['current_cars'] / road_info['capacity']
        congestion_score += min(capacity_ratio, 1) * 0.5

    # Check if 'avg_speed' is not None
    if road_info['avg_speed'] is not None:
        speed_factor = max(0, (50 - road_info['avg_speed']) / 50)
        congestion_score += speed_factor * 0.3
    
    if road_info['congestion_times']:
        congestion_score += 0.1
    
    if road_info['special_events']:
        congestion_score += 0.1 * min(len(road_info['special_events']), 3)
    
    # Determine congestion level
    if congestion_score >= 0.7:
        congestion = "Severely Congested"
        reasons.append("Severe congestion (score: {:.1f}/1.0)".format(congestion_score))
    elif congestion_score >= 0.4:
        congestion = "Congested"
        reasons.append("Moderate congestion (score: {:.1f}/1.0)".format(congestion_score))
    else:
        reasons.append("No significant congestion (score: {:.1f}/1.0)".format(congestion_score))
    
    # Generate all possible solutions
    all_solutions = []
    for category in SOLUTIONS.values():
        all_solutions.extend(category)
    
    # Filter solutions based on restrictions
    viable_solutions = []
    for solution in all_solutions:
        valid = True
        
        # Check against all restrictions
        restriction_checks = {
            "no new lanes": "lane" in solution['name'].lower() and "add" in solution['name'].lower(),
            "no widening": "widen" in solution['name'].lower(),
            "no construction": solution['type'] == "physical" and solution['cost'] > 0.4,
            "no overpasses": "overpass" in solution['name'].lower(),
            "no underpasses": "underpass" in solution['name'].lower(),
            "no bike lanes": "bike lane" in solution['name'].lower(),
            "no bus lanes": "bus lane" in solution['name'].lower(),
            "no HOV lanes": "HOV" in solution['name'].lower(),
            "no signal timing changes": "signal timing" in solution['name'].lower(),
            "no pedestrian signals": "pedestrian signal" in solution['name'].lower()
        }
        
        for restriction, condition in restriction_checks.items():
            if restriction in road_info['restrictions'] and condition:
                valid = False
                break
                
        if valid:
            # Calculate solution score
            score = calculate_solution_score(solution, congestion, road_info)
            viable_solutions.append({
                "name": solution['name'],
                "score": score,
                "type": solution['type'],
                "impact": solution['impact'],
                "safety": solution['safety'],
                "cost": solution['cost']
            })
    
    # Sort solutions by score (descending)
    viable_solutions.sort(key=lambda x: x['score'], reverse=True)
    
    # Select top solutions based on congestion level
    if congestion == "Severely Congested":
        top_solutions = [s['name'] for s in viable_solutions[:5]]
    elif congestion == "Congested":
        top_solutions = [s['name'] for s in viable_solutions[:4]]
    else:
        top_solutions = [s['name'] for s in viable_solutions[:3]]
    
    return congestion, reasons, top_solutions

def visualize_solution(original_img, road_info, solutions, detected_elements):
    """Create visualization with all possible modifications"""
    # Create a copy of the original image
    img = original_img.copy()
    height, width = img.shape[:2]
    
    # Convert to PIL Image for easier text handling
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        # Use a basic font (size may need adjustment)
        font = ImageFont.load_default()
        large_font = ImageFont.load_default(size=20)
        
        # Draw basic info
        draw.text((20, 20), f"{road_info['name']}", fill=(0, 0, 0), font=large_font)
        draw.text((20, 50), f"Condition: {road_info['condition']}", fill=(0, 0, 0), font=font)
        
        # Draw detected elements
        if detected_elements:
            draw.text((20, 80), "Detected Elements:", fill=(0, 0, 0), font=font)
            for i, element in enumerate(detected_elements[:4]):
                text = f"- {element[0]} ({element[1]*100:.1f}%)"
                draw.text((30, 110 + i*25), text, fill=(0, 0, 0), font=font)
        
        # Convert back to numpy array for OpenCV-style drawing
        img = np.array(pil_img)
        
        # Visual mapping for all solution types
        visual_elements = {
            # Lane additions
            "Add general purpose lane": lambda: (
                cv2.rectangle(img, (0, height-100), (width, height), (200, 200, 200), -1),
                cv2.putText(img, "NEW LANE", (width//2-40, height-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            ),
            "Add auxiliary lane": lambda: (
                cv2.rectangle(img, (0, height-120), (width, height-80), (180, 180, 180), -1),
                cv2.putText(img, "AUX LANE", (width//2-40, height-100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 1)
            ),
            
            # Lane conversions
            "Convert to HOV lane": lambda: (
                cv2.rectangle(img, (0, height-60), (width, height-30), (0, 100, 200), -1),
                cv2.putText(img, "HOV LANE", (width//2-40, height-45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            ),
            "Convert to bus lane": lambda: (
                cv2.rectangle(img, (0, height-80), (width, height-40), (255, 165, 0), -1),
                cv2.putText(img, "BUS LANE", (width//2-40, height-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
            ),
            
            # Intersection improvements
            "Add roundabout": lambda: (
                cv2.circle(img, (width//2, height//2), 50, (0, 255, 255), -1),
                cv2.putText(img, "ROUNDABOUT", (width//2-60, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            ),
            "Add dedicated turn lanes": lambda: (
                cv2.rectangle(img, (width-150, height-150), (width-50, height-50), (255, 255, 0), -1),
                cv2.putText(img, "TURN LANE", (width-140, height-100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            ),
            
            # Pedestrian structures
            "Add pedestrian overpass": lambda: (
                cv2.rectangle(img, (width//3, height//3), (2*width//3, height//3+20), (255,0,0), -1),
                cv2.line(img, (width//3, height//3+20), (width//3, height), (255,0,0), 3),
                cv2.line(img, (2*width//3, height//3+20), (2*width//3, height), (255,0,0), 3),
                cv2.putText(img, "OVERPASS", (width//3+10, height//3+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            ),
            
            # Signal improvements
            "Optimize signal timing": lambda: (
                cv2.circle(img, (width-50, 50), 20, (0, 255, 0), -1),
                cv2.putText(img, "OPTIMIZED", (width-120, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            ),
            
            # Markings
            "Add bike lane markings": lambda: (
                cv2.rectangle(img, (0, height-60), (width, height-30), (0, 255, 0), -1),
                cv2.putText(img, "BIKE LANE", (width//2-40, height-45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            )
        }
        
        # Apply visual modifications for top solutions (max 3)
        applied = 0
        for solution in solutions:
            if solution in visual_elements and applied < 3:
                visual_elements[solution]()
                applied += 1
                
        return img
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        return original_img

def main():
    """Main program execution with comprehensive reporting"""
    print("=== AI Traffic Optimization System ===")
    
    # Train model
    try:
        model = build_and_train_model()
    except Exception as e:
        print(f"Failed to train model: {str(e)}")
        return
    
    # Get road info
    road_info = get_road_details()
    
    # Process test images
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    
    if not test_images:
        print("No test images found!")
        return
        
    # Automatically select the first image
    image_path = os.path.join(TEST_IMAGES_DIR, test_images[0])
    print(f"\nProcessing first available image: {test_images[0]}")
    
    try:
        # Process image
        detected, original_img = predict_traffic_elements(image_path, model)
        if original_img is None:
            print("Error: Failed to process image")
            return
        
        # Print detailed road marking report
        print("\n=== Road Marking Detection Report ===")
        if detected:
            print("Detected Road Markings (with confidence %):")
            for i, (marking, confidence) in enumerate(detected, 1):
                print(f"{i}. {marking}: {confidence*100:.1f}%")
        else:
            print("No road markings detected with sufficient confidence")
            
        # Analyze traffic
        congestion, reasons, solutions = analyze_traffic(road_info, detected)
        
        # Print analysis results
        print("\n=== Traffic Analysis Results ===")
        print(f"Congestion Level: {congestion}")
        print("Analysis:")
        for reason in reasons:
            print(f"- {reason}")
        
        # Print solutions
        print("\nRecommended Solutions:")
        for i, solution in enumerate(solutions, 1):
            print(f"{i}. {solution}")
        
        # Visualize and save
        modified_img = visualize_solution(original_img, road_info, solutions, detected)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"{road_info['name'].replace(' ','_')}_{timestamp}.jpg")
        
        # Debugging output
        print(f"\nAttempting to save to: {output_path}")
        print(f"Image shape: {modified_img.shape}, dtype: {modified_img.dtype}")
        
        # Save with error checking
        if modified_img is None:
            print("Error: No image data to save!")
        elif not os.path.exists(os.path.dirname(output_path)):
            print("Error: Output directory doesn't exist!")
        else:
            cv2.imwrite(output_path, cv2.cvtColor(modified_img, cv2.COLOR_RGB2BGR))
            print(f"Image saved successfully to {output_path}")
        
        # Show comparison if not on headless server
        try:
            plt.figure(figsize=(15,7))
            plt.subplot(1,2,1)
            plt.imshow(original_img)
            plt.title("Current Road")
            plt.axis('off')
            
            plt.subplot(1,2,2)
            plt.imshow(modified_img)
            plt.title("Proposed Modifications")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not display images: {str(e)}")
            print("This is normal if running on a server without display")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()