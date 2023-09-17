import cv2
import numpy as np
import keyboard
from gtts import gTTS
import pygame
from pygame import mixer
import io

# Initialize Pygame mixer for audio output
mixer.init()

# Open the webcam
ip_camera_url = "http://10.143.170.211/video.mjpg"
cap = cv2.VideoCapture(ip_camera_url)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# Load COCO class labels
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")


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
    "Black": ((0, 0, 0), (180, 255, 30)),  # Black color range (updated)
    "Pink": ((150, 50, 50), (170, 255, 255)),  # Pink color range
    "Brown": ((10, 50, 50), (20, 180, 255)),  # Brown color range
    "Gray": ((0, 0, 80), (180, 40, 200)),  # Gray color range
    # Add more color ranges as needed
}


def color_detector():
    # Initialize Pygame mixer for audio output
    mixer.init()

    # Open the webcam
    cap = cv2.VideoCapture(0)
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # Load COCO class labels
    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")

    # Define a dictionary of color ranges for detection
    color_ranges = {
        "Red": ((0, 50, 50), (10, 255, 255)),
        "Green": ((35, 50, 50), (85, 255, 255)),
        "Blue": ((90, 50, 50), (130, 255, 255)),
        "Yellow": ((20, 50, 50), (35, 255, 255)),
        "Orange": ((10, 50, 50), (20, 255, 255)),
        "Purple": ((140, 50, 50), (165, 255, 255)),
        "Cyan": ((85, 50, 50), (100, 255, 255)),
        "White": ((0, 0, 200), (180, 40, 255)),
        "Black": ((0, 0, 0), (180, 255, 30)),
        "Pink": ((150, 50, 50), (170, 255, 255)),
        "Brown": ((10, 50, 50), (20, 180, 255)),
        "Gray": ((0, 0, 80), (180, 40, 200)),
        # Add more color ranges as needed
    }

    def identify_color(roi):
        if roi.size == 0:
            return "Other"

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        max_match = 0
        detected_color = "Other"

        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_roi, lower, upper)
            match_percentage = (cv2.countNonZero(mask) / mask.size) * 100

            if match_percentage > max_match:
                max_match = match_percentage
                detected_color = color_name

        return detected_color

    def speak_color_detected(detected_color):
        sentence = f"The detected color is {detected_color}"
        tts = gTTS(text=sentence, lang='en')

        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)

        pygame.mixer.init()
        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()

    color_detection_enabled = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if color_detection_enabled:
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(net.getUnconnectedOutLayersNames())

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        label = classes[class_id]

                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        width = int(detection[2] * frame.shape[1])
                        height = int(detection[3] * frame.shape[0])

                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)

                        roi = frame[y:y + height, x:x + width]

            detected_color = identify_color(roi)

            print(f"Detected: {label}, Confidence: {confidence:.2f}, Color: {detected_color}")

            speak_color_detected(detected_color)

            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            text = f"{label}: {confidence:.2f}, Color: {detected_color}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        height, width, _ = frame.shape

        crosshair_x, crosshair_y = width // 2, height // 2

        crosshair_size = 20
        crosshair_color = (0, 0, 255)
        cv2.line(frame, (crosshair_x - crosshair_size, crosshair_y), (crosshair_x + crosshair_size, crosshair_y),
                 crosshair_color, 2)
        cv2.line(frame, (crosshair_x, crosshair_y - crosshair_size), (crosshair_x, crosshair_y + crosshair_size),
                 crosshair_color, 2)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if keyboard.is_pressed('c'):
            crosshair_roi = frame[crosshair_y - crosshair_size:crosshair_y + crosshair_size,
                            crosshair_x - crosshair_size:crosshair_x + crosshair_size]
            detected_color = identify_color(crosshair_roi)

            speak_color_detected(detected_color)
            print(f"Color around crosshair: {detected_color}")

    cap.release()
    cv2.destroyAllWindows()


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

def speak_color_detected(detected_color):
    # Convert the detected_color to a sentence
    sentence = f"The detected color is {detected_color}"

    # Use gTTS to convert the sentence to speech
    tts = gTTS(text=sentence, lang='en')

    # Get the audio as bytes
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)

    # Play the audio using Pygame
    pygame.mixer.init()
    pygame.mixer.music.load(audio_data)
    pygame.mixer.music.play()

color_detection_enabled = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if color_detection_enabled:
        # Perform object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    label = classes[class_id]

                    # Get the coordinates of the bounding box
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    # Calculate the top-left corner of the bounding box
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Extract the region of interest (ROI)
                    roi = frame[y:y+height, x:x+width]

                    # Identify the color within the ROI
                    detected_color = identify_color(roi)

                    print(f"Detected: {label}, Confidence: {confidence:.2f}, Color: {detected_color}")

                    # Speak out the detected color using Pygame
                    speak_color_detected(detected_color)

                    # Draw a green bounding box and label the object
                    color = (0, 255, 0)  # Green color in BGR
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                    text = f"{label}: {confidence:.2f}, Color: {detected_color}"
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Get the frame dimensions
    height, width, _ = frame.shape

    # Calculate the position of the crosshair in the middle of the frame
    crosshair_x, crosshair_y = width // 2, height // 2

    # Draw a crosshair in the middle of the frame
    crosshair_size = 20
    crosshair_color = (0, 0, 255)  # Red color in BGR
    cv2.line(frame, (crosshair_x - crosshair_size, crosshair_y), (crosshair_x + crosshair_size, crosshair_y),
             crosshair_color, 2)
    cv2.line(frame, (crosshair_x, crosshair_y - crosshair_size), (crosshair_x, crosshair_y + crosshair_size),
             crosshair_color, 2)

    cv2.imshow("Object Detection", frame)

    # Check for the 'q' key to exit the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Check for the 'c' key to toggle color detection
    if keyboard.is_pressed('c'):
        # Get the color around the crosshair
        crosshair_roi = frame[crosshair_y - crosshair_size:crosshair_y + crosshair_size,
                             crosshair_x - crosshair_size:crosshair_x + crosshair_size]
        detected_color = identify_color(crosshair_roi)

        # Speak out the detected color using Pygame
        speak_color_detected(detected_color)
        print(f"Color around crosshair: {detected_color}")

cap.release()
cv2.destroyAllWindows()
