import cv2
import numpy as np
import math
import os
import time
import keyboard
import speech_recognition as sr
import pygame
from gtts import gTTS

# Load YOLOv3 model for object detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels for object detection
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

print('hi')

# Set IP camera URL
# ip_camera_url = "http://192.168.1.183/video.mjpg"
# ip_camera_url = "http://192.168.1.56:8080/video"
ip_camera_url = "http://192.168.1.86/video.mjpg"

# Open a connection to the IP camera
cap = cv2.VideoCapture(0)

# Initialize the recognizer
recognizer = sr.Recognizer()

# Initialize the pygame mixer
pygame.mixer.init()

# User input for the target object and color
user_desired_object = None  # Initialize to None
user_desired_color = None  # Initialize to None

# Directory for saving audio output files
audio_output_directory = os.path.expanduser('~/audio_output')
os.makedirs(audio_output_directory, exist_ok=True)

# Dictionary mapping clock positions to angles (in degrees)
clock_positions = {
    "12 o'clock": 90,
    "11 o'clock": 120,
    "10 o'clock": 150,
    "9 o'clock": 180,
    "8 o'clock": 210,
    "7 o'clock": 240,
    "6 o'clock": 270,
    "5 o'clock": 300,
    "4 o'clock": 330,
    "3 o'clock": 360,
    "2 o'clock": 390,
    "1 o'clock": 420
}

# Initialize the timer variables
start_time = time.time()
cycle_duration = 10  # Total duration of one cycle (10 seconds rest + 0.3 seconds detection)
cycle_start_time = start_time  # Start time of the current cycle
is_detection_active = False

# Real height of the mini alarm clock in meters (replace with your value)
target_object_real_height = 0.15

# Initialize pygame for audio
pygame.mixer.init()

# Function to play audio file
def play_audio(filename):
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

# Function to generate an audio file with clock position and distance
def generate_audio_clock_and_distance(clock_position, distance):
    text_to_speech = f"{clock_position}, {distance:.2f} meters"
    filename = os.path.join(audio_output_directory, f"{clock_position}_distance.mp3")
    tts = gTTS(text=text_to_speech, lang='en')
    tts.save(filename)
    return filename

# Audio prompt filename
audio_prompt_filename = "what_object.mp3"

# Load YOLOv3 model for color detection
color_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels for color detection
color_classes = []
with open("coco.names", "r") as f:
    color_classes = f.read().strip().split("\n")

# Function to perform color detection
def identify_color(roi):
    if roi.size == 0:  # Check if the ROI is empty
        return "Other"

    # Convert the ROI to the HSV color space for better color analysis
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    max_match = 0
    detected_color = "Other"

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

    # Iterate through color ranges and find the best match
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_roi, lower, upper)
        match_percentage = (cv2.countNonZero(mask) / mask.size) * 100

        if match_percentage > max_match:
            max_match = match_percentage
            detected_color = color_name

    return detected_color

# Lists to store detected objects and their colors
detected_objects = []
detected_colors = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate the time elapsed since the start of the current cycle
    current_time = time.time()
    time_elapsed = current_time - cycle_start_time

    if not user_desired_object:
        # Prompt the user with audio to specify the desired object
        if not is_detection_active:
            play_audio(audio_prompt_filename)
            time.sleep(1)  # Wait for 10 seconds
            print("Press the spacebar and speak the target object...")
            keyboard.wait("space")  # Wait for spacebar press
            print("Listening for the desired object...")
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
            try:
                user_desired_object = recognizer.recognize_google(audio).lower()
                print(f"You are looking for: {user_desired_object}")

                # Prompt the user with audio to specify the desired color
                play_audio("what_color.mp3")
                time.sleep(1)  # Wait for 10 seconds
                print("Press the spacebar and speak the target color...")
                keyboard.wait("space")  # Wait for spacebar press
                print("Listening for the desired color...")
                with sr.Microphone() as source:
                    audio = recognizer.listen(source)
                try:
                    user_desired_color = recognizer.recognize_google(audio).lower()
                    print(f"You are looking for the color: {user_desired_color}")
                except sr.UnknownValueError:
                    print("Could not understand the audio for color")
                except sr.RequestError as e:
                    print(f"Error: {e}")

            except sr.UnknownValueError:
                print("Could not understand the audio for object")
            except sr.RequestError as e:
                print(f"Error: {e}")

    if time_elapsed < 10:  # Rest for the first 10 seconds of the cycle
        is_detection_active = False
    elif 10 <= time_elapsed < 10.3:  # Run object detection for 0.3 seconds
        is_detection_active = True
        start_time = current_time

    if is_detection_active:
        # Perform object detection and color detection only during the active detection period
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        object_detected = False  # Flag to indicate object detection

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    label = classes[class_id]
                    print(label)
                    if label == user_desired_object and not object_detected:
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        width = int(detection[2] * frame.shape[1])
                        height = int(detection[3] * frame.shape[0])
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)

                        # Extract the region of interest (ROI)
                        roi = frame[y:y + height, x:x + width]

                        # Calculate the position relative to the center point crosshair
                        delta_x = center_x - frame.shape[1] // 2
                        delta_y = frame.shape[0] // 2 - center_y

                        # Calculate the estimated distance
                        estimated_distance = (target_object_real_height * frame.shape[0]) / (2 * height * math.tan(math.radians(30)))

                        # Identify the color within the ROI
                        detected_color = identify_color(roi)
                        print(detected_color)

                        # Check if the detected color matches the desired color
                        if detected_color.lower() == user_desired_color.lower():
                            # Check if the object is near the center
                            if abs(delta_x) < 20 and abs(delta_y) < 20:  # Adjust the threshold as needed
                                print(f"*** {user_desired_object.upper()} DETECTED IN FRONT OF YOU! ***")
                                print(f"Estimated Distance: {estimated_distance:.2f} meters")
                                detected_objects.append(user_desired_object)
                                detected_colors.append(detected_color)

                                # Generate and play audio for the special message
                                audio_file = os.path.join(audio_output_directory, "object_in_front.mp3")
                                tts = gTTS(text="Object is in front of you", lang='en')
                                tts.save(audio_file)
                                play_audio(audio_file)
                            else:
                                # Object is at a clock position
                                # Calculate corrected angle based on clock positions
                                delta_angle = math.degrees(math.atan2(delta_y, delta_x))
                                if delta_angle < 0:
                                    delta_angle += 360

                                # Find the closest clock position
                                closest_position = min(clock_positions.keys(), key=lambda position: abs(clock_positions[position] - delta_angle))

                                # Generate and play audio with clock position and distance
                                audio_file = generate_audio_clock_and_distance(closest_position, estimated_distance)
                                play_audio(audio_file)

                            object_detected = True  # Set the object detection flag to True

        # Draw crosshair at the center of the frame
        crosshair_size = 20
        crosshair_color = (0, 0, 255)
        cv2.line(frame, (frame.shape[1] // 2, frame.shape[0] // 2 - crosshair_size),
                 (frame.shape[1] // 2, frame.shape[0] // 2 + crosshair_size), crosshair_color, 2)
        cv2.line(frame, (frame.shape[1] // 2 - crosshair_size, frame.shape[0] // 2),
                 (frame.shape[1] // 2 + crosshair_size, frame.shape[0] // 2), crosshair_color, 2)

    # Draw clock labels around the screen
    for position, angle in clock_positions.items():
        x_label = int(frame.shape[1] // 2 + 0.9 * frame.shape[1] // 2 * math.cos(math.radians(angle)))
        y_label = int(frame.shape[0] // 2 - 0.9 * frame.shape[0] // 2 * math.sin(math.radians(angle)))
        cv2.putText(frame, position, (x_label, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Check if the current cycle has completed and start the next cycle
    if time_elapsed >= cycle_duration:
        cycle_start_time = current_time

# Print all detected objects and their colors
print("Detected Objects:")
for i in range(len(detected_objects)):
    print(f"Object {i + 1}: {detected_objects[i]} (Color: {detected_colors[i]})")

cap.release()
cv2.destroyAllWindows()
