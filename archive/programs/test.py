import cv2
import numpy as np

def load_and_preprocess_image(image_path):
    """Load an image and convert it to grayscale."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image

def detect_edges(gray_image):
    """Detect edges in the grayscale image using Canny edge detection."""
    edges = cv2.Canny(gray_image, 100, 200)
    return edges

def find_contours(edges):
    """Find contours in the edge-detected image."""
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def classify_elements(contours, original_image):
    """Classify and draw bounding boxes around detected elements."""
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Example threshold for area
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"Detected element with area: {area}")

def main(image_path):
    """Main function to execute the image processing pipeline."""
    original_image, gray_image = load_and_preprocess_image(image_path)
    edges = detect_edges(gray_image)
    contours = find_contours(edges)
    classify_elements(contours, original_image)

    # Display the result
    cv2.imshow("Detected Elements", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'path_to_your_image.jpg' with the actual path to your image file
    main("traffic/road.jpg")