import cv2
import numpy as np
import json
from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile

# Load the YOLO model
model = YOLO("best.pt")

app = Flask(__name__)

# Function to perform object detection and return results in JSON format
def detect_objects(image_path):
    """
    This function takes the path to an image, detects objects using YOLO,
    and returns the detected boxes, class names, and confidence scores in a dictionary.
    """
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Perform object detection using YOLO
    results = model.predict(image_path)

    # Extract bounding boxes, class indices, and confidence scores
    boxes = results[0].boxes.xyxy  # Bounding boxes (x_min, y_min, x_max, y_max)
    classes = results[0].boxes.cls  # Class indices
    scores = results[0].boxes.conf  # Confidence scores

    # Get the class names from the model
    class_names = model.names

    number_plate = []
    # Prepare a list of detection results
    detections = []
    for box, cls, score in zip(boxes, classes, scores):
        # Convert the bounding box coordinates to integers
        x_min, y_min, x_max, y_max = map(int, box)

        # Get the class name for the detected object
        class_name = class_names[int(cls)]

        number_plate.append(class_name)

        # detections.append({
        #     "class_name": class_name,
        #     "confidence": float(score),
        #     "bounding_box": {
        #         "x_min": x_min,
        #         "y_min": y_min,
        #         "x_max": x_max,
        #         "y_max": y_max
        #     }
        # })

    # Return the detections as JSON
    return number_plate

@app.route('/detect', methods=['POST'])
def detect_objects_in_request():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    
    # Save the uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        file.save(temp_file_path)

    # Detect objects in the image
    detections = detect_objects(temp_file_path)

    # Return the detection results as JSON
    return jsonify(detections)


if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)