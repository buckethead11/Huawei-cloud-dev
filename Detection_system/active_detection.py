import cv2
import numpy as np
import time
import pygame
import pyttsx3  # For text-to-speech

# Initialize Pygame
pygame.init()

# Initialize Pygame text-to-speech
engine = pyttsx3.init()

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Open the webcam
ip_camera_url = "http://192.168.1.233/video.mjpg"
cap = cv2.VideoCapture(ip_camera_url)

# Define a dictionary of color ranges for detection
color_ranges = {
    "Red": ((0, 50, 50), (10, 255, 255)),  # Red color range
    "Green": ((35, 50, 50), (85, 255, 255)),  # Green color range
    "Blue": ((90, 50, 50), (130, 255, 255)),  # Blue color range (adjusted)
    "Yellow": ((20, 50, 50), (35, 255, 255)),  # Yellow color range
    "Orange": ((10, 50, 50), (20, 255, 255)),  # Orange color range
    "Purple": ((140, 50, 50), (165, 255, 255)),  # Purple color range
    "Cyan": ((85, 50, 50), (100, 255, 255)),  # Cyan color range
    "White": ((0, 0, 200), (180, 40, 255)),  # White color range
    "Black": ((0, 0, 0), (180, 255, 30))  # Black color range (updated)
    # Add more color ranges as needed
}

def identify_color(roi):
    if roi.size == 0:  # Check if the ROI is empty
        return "Other"

    # Convert the ROI to the HSV color space for better color analysis
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    max_match = 0
    detected_color = "Other"

    # Iterate through color ranges and find the best match
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_roi, lower, upper)
        match_percentage = (cv2.countNonZero(mask) / mask.size) * 100

        if match_percentage > max_match:
            max_match = match_percentage
            detected_color = color_name

    return detected_color

def announce_detected_objects(detected_objects):
    object_counts = {}  # Dictionary to store object counts

    for obj in detected_objects:
        label = obj["label"]
        if label in object_counts:
            object_counts[label] += 1
        else:
            object_counts[label] = 1

    if object_counts:
        #object_text_list = [f"{count} {label}" if count > 1 else f"{count} {label}" for label, count in object_counts.items()]

        object_text = ", ".join(object_text_list)
        text_to_speech(f"{object_text} in front of you")
    else:
        text_to_speech("No objects detected")

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

# Initialize timer variables
start_time = time.time()
detection_interval = 5  # Run detection every 5 seconds
detection_duration = 3  # Run detection for 0.3 seconds

# Initialize a flag variable to track the cycle
detection_cycle_started = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw lines representing the user's walking path forward
    path_color = (0, 255, 0)  # Green color in BGR
    path_thickness = 2
    path_y_start = frame.shape[0] - 30  # Starting point near the bottom
    path_y_end = frame.shape[0] // 4  # Ending point at the center

    # Left path
    path_x_left_start = 0
    path_x_left_end = frame.shape[1] // 2 - 50
    cv2.line(frame, (path_x_left_start, path_y_start), (path_x_left_end, path_y_end), path_color, path_thickness)

    # Right path
    path_x_right_start = frame.shape[1] - 0
    path_x_right_end = frame.shape[1] // 2 + 50
    cv2.line(frame, (path_x_right_start, path_y_start), (path_x_right_end, path_y_end), path_color, path_thickness)

    # Label the area in between as "Detection Zone"
    cv2.putText(frame, "Detection Zone", (frame.shape[1] // 2 - 50, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, path_color, 2)

    # Check if it's time to run object detection
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time >= detection_interval:
        if not detection_cycle_started:
            print("Object Detection Cycle Started")
            detection_cycle_started = True

        # Perform object detection within the Detection Zone
        detection_zone = frame[path_y_end:path_y_start, path_x_left_start:path_x_right_start]
        blob = cv2.dnn.blobFromImage(detection_zone, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        detected_objects = []  # List to store detected objects

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.7:
                    label = classes[class_id]

                    # Get the coordinates of the bounding box
                    center_x = int(detection[0] * detection_zone.shape[1])
                    center_y = int(detection[1] * detection_zone.shape[0])
                    width = int(detection[2] * detection_zone.shape[1])
                    height = int(detection[3] * detection_zone.shape[0])

                    # Calculate the top-left corner of the bounding box
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Extract the region of interest (ROI) within the Detection Zone
                    roi = detection_zone[y:y + height, x:x + width]

                    # Identify the color within the ROI
                    detected_color = identify_color(roi)

                    # Store detected object information in a dictionary
                    detected_object = {
                        "label": label,
                        "confidence": confidence,
                        "color": detected_color
                    }
                    detected_objects.append(detected_object)

                    # Draw a green bounding box and label the object within the Detection Zone
                    color = (0, 255, 0)  # Green color in BGR
                    cv2.rectangle(detection_zone, (x, y), (x + width, y + height), color, 2)
                    text = f"{label}: {confidence:.2f}, Color: {detected_color}"
                    cv2.putText(detection_zone, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Announce detected objects
        announce_detected_objects(detected_objects)

        # Reset the timer
        start_time = time.time()

    else:
        if detection_cycle_started:
            print("Rest Cycle Started")
            detection_cycle_started = False

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
