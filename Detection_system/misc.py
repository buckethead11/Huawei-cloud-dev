import cv2
import numpy as np
import winsound
import math
import time
import speech_recognition as sr
import pygame.mixer
import keyboard
import os
from gtts import gTTS

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
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

# User input for the object to be notified about
user_desired_object = None  # Initialize to None

# Directory for saving audio output files
audio_output_directory = os.path.expanduser('~/audio_output')
os.makedirs(audio_output_directory, exist_ok=True)

# Dictionary mapping clock positions to angles
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
            time.sleep(10)  # Wait for 10 seconds
            print("Press the spacebar and speak the target object...")
            keyboard.wait("space")  # Wait for spacebar press
            print("Listening for the desired object...")
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
            try:
                user_desired_object = recognizer.recognize_google(audio).lower()
                print(f"You are looking for: {user_desired_object}")
            except sr.UnknownValueError:
                print("Could not understand the audio")
            except sr.RequestError as e:
                print(f"Error: {e}")

    if time_elapsed < 10:  # Rest for the first 10 seconds of the cycle
        is_detection_active = False
    elif 10 <= time_elapsed < 10.3:  # Run object detection for 0.3 seconds
        is_detection_active = True
        start_time = current_time

    if is_detection_active:
        # Perform object detection only during the active detection period
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

                    if label == user_desired_object and not object_detected:
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        width = int(detection[2] * frame.shape[1])
                        height = int(detection[3] * frame.shape[0])
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)

                        # Calculate the position relative to the center point crosshair
                        delta_x = center_x - frame.shape[1] // 2
                        delta_y = frame.shape[0] // 2 - center_y
                        angle = math.degrees(math.atan2(delta_y, delta_x)) + 180  # Add 180-degree offset
                        if angle >= 360:
                            angle -= 360

                        # Calculate corrected angle based on clock positions
                        corrected_angle = (angle + 270) % 360

                        # Map corrected angle to clock positions
                        position = "Unknown"
                        if 0 <= corrected_angle < 30:
                            position = "6 o'clock"
                        elif 30 <= corrected_angle < 60:
                            position = "5 o'clock"
                        elif 60 <= corrected_angle < 90:
                            position = "4 o'clock"
                        elif 90 <= corrected_angle < 120:
                            position = "3 o'clock"
                        elif 120 <= corrected_angle < 150:
                            position = "2 o'clock"
                        elif 150 <= corrected_angle < 180:
                            position = "1 o'clock"
                        elif 180 <= corrected_angle < 210:
                            position = "12 o'clock"
                        elif 210 <= corrected_angle < 240:
                            position = "11 o'clock"
                        elif 240 <= corrected_angle < 270:
                            position = "10 o'clock"
                        elif 270 <= corrected_angle < 300:
                            position = "9 o'clock"
                        elif 300 <= corrected_angle < 330:
                            position = "8 o'clock"
                        elif 330 <= corrected_angle < 360:
                            position = "7 o'clock"

                        # Calculate the estimated distance
                        estimated_distance = (target_object_real_height * frame.shape[0]) / (2 * height * math.tan(math.radians(30)))

                        print(f"*** {user_desired_object.upper()} DETECTED! ***")
                        print(f"Location: X={x}, Y={y}, Width={width}, Height={height}")
                        print(f"Position: {position}")
                        print(f"Estimated Distance: {estimated_distance:.2f} meters")
                        winsound.Beep(500, 1000)  # Play beep sound for 1 second

                        # Generate and play audio with clock position and distance
                        audio_file = generate_audio_clock_and_distance(position, estimated_distance)
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

cap.release()
cv2.destroyAllWindows()
