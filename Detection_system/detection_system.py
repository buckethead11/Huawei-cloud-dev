import keyboard
import pygame
import os
import cv2
import numpy as np
import math
import time
import speech_recognition as sr
import pygame.mixer
import keyboard
import os
from gtts import gTTS
import base64
import io
import requests
import openai
import cv2
import numpy as np
import urllib.request
import pygame
from pygame import mixer
import tempfile

#IMPORTS FOR DOCUMENT SYSTEM
import cv2
import speech_recognition as sr
import threading
import base64
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkocr.v1 import *
import pygame
import openai
import pygame
from gtts import gTTS
import os

from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkimage.v2 import ImageClient, RunImageTaggingRequest, ImageTaggingReq
from huaweicloudsdkimage.v2.region.image_region import ImageRegion

ip_camera_url = "http://192.168.1.233/video.mjpg"

def run_document_system():
    # Set your OpenAI API key
    api_key = "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"

    # Initialize the OpenAI API client
    openai.api_key = api_key

    #FUNCTION FOR GPT DOCUMENT INTERPRETATION

    def gpt_document_interpreter(query):
        def text_to_speech(text):
            # Convert the text response to audio using gTTS
            tts = gTTS(text=text, lang='en')

            # Save the audio to a temporary file
            audio_path = "temp_audio.mp3"
            tts.save(audio_path)
            return audio_path

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a camera wearable that helps the blind user to interpret documents."},
                {"role": "user", "content": f"Here is the summarised version of the a document. Can you interpret it to me in one paragraph?:\n\n{query}\n\nAnswer me with the paragraph directly"},
            ],
        )

        content = response.choices[0].message.content
        print(content)

        # Convert text to speech
        audio_path = text_to_speech(content)

        # Initialize Pygame Mixer
        pygame.mixer.init()

        # Load and play the audio
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Delete the temporary audio file
        os.remove(audio_path)

    '''
    # Example usage
    query = "Sprint PCS Phone Charges Sprint PCS www.sprintpcs.con customer account number billing period ending invoice date oled May6,2000 May8,2000 3 of PCS Phone Charges for $29.99 Service Plan -May 7to Jun.6 •$29 99 Monthly ServiceCharge • Includes 180 Minutes To Use Anytime Anywhere On The Sprint PCS Network •Caller ID,Call Waiting •Three-Way Calling • Sprint PCS Voiceinall •Detailed Billing PCS Phone Monthly Service Charges Description Charges $29.99 Service Plan-May 7to Jun.6 $29.99 Reminder:Service charges are billed one month in advance. Other PCS Options and Charges ltem Charges Long Distance May 7 to Jun.6 $0.00 Added Off-Peak Minutes: Evening/Weekend May 7 to Jun.6 $10.00 Actvation Fee May 5 $29.99 Sprint PCS Advantage Agreement Credit -$10.00 $25 Sprint PCS Service Credit -$25.00 Phone $199.99 Leather Case $14.99 Current Activity Charges for $249.96"
    gpt_document_interpreter(query)
    '''



    # Initialize the Huawei OCR client
    class HuaweiOCR:
        def __init__(self, ak, sk, region):
            self.credentials = BasicCredentials(ak, sk)
            self.client = OcrClient.new_builder() \
                .with_credentials(self.credentials) \
                .with_region(OcrRegion.value_of(region)) \
                .build()

        def recognize_local_image(self, file_path):
            with open(file_path, "rb") as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            request = RecognizeWebImageRequest()
            request.body = WebImageRequestBody(image=image_base64)
            try:
                response = self.client.recognize_web_image(request)
                return response
            except exceptions.ClientRequestException as e:
                return {"status_code": e.status_code,
                        "request_id": e.request_id,
                        "error_code": e.error_code,
                        "error_msg": e.error_msg}

        @staticmethod
        def extract_words(response):
            words_list = []
            result = response.result
            words_block_list = result.words_block_list if result and hasattr(result, 'words_block_list') else []
            for block in words_block_list:
                word = block.words if hasattr(block, 'words') else ""
                words_list.append(word)
            sentence = " ".join(words_list)
            return sentence

    # IP camera URL
    ip_camera_url = "http://192.168.1.233/video.mjpg"

    # Initialize the IP camera capture
    cap = cv2.VideoCapture(ip_camera_url)

    # Define a variable to hold the captured frame
    frame = None

    # Function to capture frames in a separate thread
    def capture_frames():
        global frame
        while True:
            ret, current_frame = cap.read()
            frame = current_frame
            cv2.imshow("IP Camera", frame)
            cv2.waitKey(1)  # Add a small delay to reduce resource usage

    # Start the frame capture thread
    frame_capture_thread = threading.Thread(target=capture_frames)
    frame_capture_thread.daemon = True
    frame_capture_thread.start()

    # Initialize the speech recognizer
    recognizer = sr.Recognizer()

    # Function to recognize the "scan document" command
    def recognize_audio_command():
        with sr.Microphone() as source:
            print("Listening for the 'scan document' command...")
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio)
                if "scan document" in command.lower():
                    return True
            except sr.WaitTimeoutError:
                pass
            return False

    # Function to play an audio file
    def play_audio(audio_file):
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    # Wait for the "scan document" audio command
    while not recognize_audio_command():
        pass

    # Play the scanning audio
    play_audio("scanning_document.mp3")

    # Save the frame as "document.jpeg"
    if frame is not None:
        cv2.imwrite("document.jpeg", frame)

        # Release the camera and close OpenCV window
        cap.release()
        cv2.destroyAllWindows()

        # Run Huawei OCR on the captured image
        ocr_instance = HuaweiOCR("FIQHFW83ELH7JB5YRMFE", "9QoHcbIpa3I6WADQalBWUznLTqUDNWJZazpUHoL6", "ap-southeast-2")
        ocr_response = ocr_instance.recognize_local_image("phone_bill.png")

        if isinstance(ocr_response, RecognizeWebImageResponse):
            sentence = HuaweiOCR.extract_words(ocr_response)
            print(sentence)
            if(sentence):
                gpt_document_interpreter(sentence)
        else:
            print("An error occurred:", ocr_response)

    # Release the camera and close OpenCV window if the loop exits
    cap.release()
    cv2.destroyAllWindows()

def color_detector():
    # Initialize Pygame mixer for audio output
    mixer.init()

    # Open the webcam
    cap = cv2.VideoCapture(ip_camera_url)
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
        "Orange": ((20, 50, 50), (35, 255, 255)),
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


def object_finder():
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
    ip_camera_url = "http://192.168.1.233/video.mjpg"

    # Open a connection to the IP camera
    cap = cv2.VideoCapture(ip_camera_url)

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
        # Generate a temporary directory to save the audio file
        temp_dir = tempfile.mkdtemp()
        filename = os.path.join(temp_dir, f"{clock_position}_distance.mp3")
        tts = gTTS(text=text_to_speech, lang='en')
        tts.save(filename)
        return filename

        # Check if audio has been played already, and if not, play it
        if not audio_played:
            play_audio(filename)
            audio_played = True  # Set the flag to True to indicate audio has been played

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
            "Orange": ((20, 50, 50), (35, 255, 255)),  # Yellow color range
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
        elif 10   <= time_elapsed < 10.3:  # Run object detection for 0.3 seconds
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
                            estimated_distance = (target_object_real_height * frame.shape[0]) / (
                                        2 * height * math.tan(math.radians(30)))

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

                                    # Generate a temporary directory to save the audio file
                                    temp_dir = tempfile.mkdtemp()
                                    audio_file = os.path.join(temp_dir, "object_in_front.mp3")
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
                                    closest_position = min(clock_positions.keys(), key=lambda position: abs(
                                        clock_positions[position] - delta_angle))

                                    # Generate and play audio with clock position and distance
                                    if object_detected != True:
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


# Initialize the speech recognition
recognizer = sr.Recognizer()

# Initialize the pygame mixer for audio playback
pygame.mixer.init()

# Define the paths to audio files
activate_audio_path = "activate_object_finder.mp3"
deactivate_audio_path = "deactive_object_finder.mp3"

# Flag to control microphone and object finder
microphone_active = False
object_finder_active = False
run_object_finder = False  # Flag to run the object finder once

while True:
    # Check if the "m" key is pressed
    if keyboard.is_pressed('m'):
        if not microphone_active:
            print("Press and hold the 'm' key to give a voice command.")
            microphone_active = True

            # Start listening for voice commands
            with sr.Microphone() as source:
                print("Say a command:")
                audio = recognizer.listen(source)

            try:
                # Recognize the audio command
                command = recognizer.recognize_google(audio)
                print("You said:", command)

                if command.lower() == "object finder":
                    # Activate the object finder functionality
                    object_finder_active = True
                    pygame.mixer.music.load(activate_audio_path)
                    pygame.mixer.music.play()
                    print("Object finder activated. Press 'o' to find objects.")
                    run_object_finder = True

                if command.lower() == "quit object finder":
                    # Deactivate the object finder functionality
                    object_finder_active = False
                    pygame.mixer.music.load(deactivate_audio_path)
                    pygame.mixer.music.play()
                    print("Object finder deactivated.")

                # Add more conditionals for other recognized commands if needed.
                if command.lower() == "environment":
                    environment_audio_path = "start_environment_detection.mp3"
                    pygame.mixer.music.load(environment_audio_path)
                    pygame.mixer.music.play()
                    print("Environment detection start")

                    # Initialize the webcam (you may need to change the device index)
                    cap = cv2.VideoCapture(ip_camera_url)

                    if not cap.isOpened():
                        print("Error: Could not open webcam.")
                        exit()

                    # Capture a frame from the webcam
                    ret, frame = cap.read()

                    if not ret:
                        print("Error: Could not capture a frame.")
                        exit()

                    # Release the webcam
                    cap.release()

                    # Specify the file path and name for saving the snapshot
                    file_path = "snapshot.jpeg"

                    # Save the captured frame as an image
                    cv2.imwrite(file_path, frame)

                    # Initialize Pygame
                    pygame.init()
                    mixer.init()


                    def yolo_img_detection(image_file_path):
                        # Load YOLOv3 model and its configuration file
                        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

                        # Load COCO labels (80 classes)
                        with open("coco.names", "r") as f:
                            classes = f.read().strip().split("\n")

                        # Load image from URL
                        '''
                        url = img_url
                        req = urllib.request.urlopen(url)
                        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                        image = cv2.imdecode(arr, -1)
                        '''

                        #Load image
                        image = cv2.imread(image_file_path)

                        # Get image dimensions
                        height, width, _ = image.shape

                        # Prepare the image for YOLO model
                        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

                        # Set input to the network
                        net.setInput(blob)

                        # Get detections
                        outs = net.forward(net.getUnconnectedOutLayersNames())

                        # Initialize lists to store detected objects' information
                        class_ids = []
                        confidences = []
                        boxes = []

                        # Process each detection
                        for out in outs:
                            for detection in out:
                                scores = detection[5:]
                                class_id = np.argmax(scores)
                                confidence = scores[class_id]

                                if confidence > 0.5:  # You can adjust this confidence threshold
                                    # Object detected
                                    center_x = int(detection[0] * width)
                                    center_y = int(detection[1] * height)
                                    w = int(detection[2] * width)
                                    h = int(detection[3] * height)

                                    # Rectangle coordinates
                                    x = int(center_x - w / 2)
                                    y = int(center_y - h / 2)

                                    # Store information
                                    class_ids.append(class_id)
                                    confidences.append(float(confidence))
                                    boxes.append([x, y, w, h])

                        # Apply non-maximum suppression to remove overlapping boxes
                        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                        # Initialize a list to store the detection results
                        detection_results = []

                        # Draw bounding boxes and labels on the image
                        for i in range(len(boxes)):
                            if i in indexes:
                                x, y, w, h = boxes[i]
                                label = str(classes[class_ids[i]])
                                confidence = confidences[i]

                                # Add the detection result to the list
                                detection_results.append({
                                    "label": label,
                                    "confidence": confidence,
                                    "box": (x, y, x + w, y + h)
                                })

                                color = (0, 255, 0)  # Green color for the bounding box
                                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                                cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, color, 2)

                        # Display the result
                        # cv2.imshow("Object Detection", image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                        final_result = ""

                        # Print the detection results
                        for result in detection_results:
                            result_text = f"Label: {result['label']}, Confidence: {result['confidence']}, Bounding Box: {result['box']} \n"
                            final_result += result_text

                        return final_result


                    def huawei_image_detection(image_path):
                        time.sleep(2)
                        if __name__ == "__main__":
                            ak = "AB39NJWJBQGMLXKSX7JX"
                            sk = "CxGQm3DtEas0sE0C9XvlLTHeY8Q9AVFsPdOn7Nq9"

                            credentials = BasicCredentials(ak, sk)

                            client = ImageClient.new_builder() \
                                .with_credentials(credentials) \
                                .with_region(ImageRegion.value_of("ap-southeast-1")) \
                                .build()

                            try:
                                request = RunImageTaggingRequest()



                                # Pass the base64-encoded image data within the URL
                                request.body = ImageTaggingReq(
                                    limit=50,
                                    threshold=75,
                                    language="en",
                                    url= image_path
                                    # Pass the base64-encoded image within the URL
                                )
                                response = client.run_image_tagging(request)

                                if response.status_code == 200:
                                    # print("Response Content:")
                                    # print(response)
                                    return response

                                else:
                                    print("Image tagging request failed with status code:", response.status_code)

                            except exceptions.ClientRequestException as e:
                                print(e.status_code)
                                print(e.request_id)
                                print(e.error_code)
                                print(e.error_msg)


                    def gpt_environment(combined_response):
                        # Set your OpenAI API key
                        api_key = "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"

                        # Initialize the OpenAI API client
                        openai.api_key = api_key

                        detection_results = combined_response

                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-16k",
                            messages=[
                                {"role": "system", "content": """You are a wearable for the blind that helps to describe the surrounding environment. Given below are the results from 2 detection models. Can you analyse and hence infer them, describing the environment to the blind user simply in one paragraph. Exclude specifics and technical terms/jargons, and also reply like you are talkinmg directly to the blind user. \n\n
                                Sample: You seem to be in a train cabin, with many seats. There are many humans detected, indicating that this cabin is quite crowded. There are also handrails around.
                        """},
                                {"role": "user", "content": detection_results},
                            ],
                        )

                        content = response.choices[0].message.content
                        return content


                    # Image
                    API_KEY = "8af6298d8f104173e1320d50ea241a12"
                    IMAGE_PATH = f"snapshot.jpeg"
                    ALBUM_ID = ""

                    with open(IMAGE_PATH, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read())

                    url = "https://api.imgbb.com/1/upload"
                    payload = {
                        "key": API_KEY,
                        "image": encoded_image,
                    }
                    response = requests.post(url, payload)
                    response = response.json()
                    img_url = response["data"]["display_url"]


                    yolo_response = yolo_img_detection("snapshot.jpeg")
                    huawei_response = huawei_image_detection(img_url)
                    combined_response = f"{yolo_response} \n \n {huawei_response}"
                    print(combined_response)

                    # Generate environment description using GPT-3
                    environment_description = gpt_environment(combined_response)
                    print(environment_description)

                    # Initialize the text-to-speech engine (you might need to install a TTS library)
                    # Replace 'your_tts_library' with the actual library you want to use
                    # Example: pyttsx3, gTTS, etc.
                    # Use a library that supports audio output to Pygame.
                    tts_engine = 'your_tts_library'

                    # Convert the GPT-3 response to audio
                    # Convert the GPT-3 response to audio
                    if tts_engine == 'your_tts_library':
                        # Use your preferred text-to-speech library to generate audio
                        # Example code for gTTS (Google Text-to-Speech):
                        from gtts import gTTS

                        tts = gTTS(text=environment_description, lang='en')
                        tts.save('output.mp3')

                        # Load and play the audio using Pygame mixer
                        mixer.music.load('output.mp3')

                        # Calculate the duration of the audio in milliseconds
                        audio_duration_ms = int(mixer.Sound('output.mp3').get_length() * 1000)

                        # Play the audio
                        mixer.music.play()

                        # Add a delay based on the duration of the audio
                        pygame.time.wait(audio_duration_ms)

                if command.lower() == "color":
                    environment_audio_path = "detecting_color.mp3"
                    pygame.mixer.music.load(environment_audio_path)
                    pygame.mixer.music.play()
                    color_detector()

                if command.lower() == "stop environment":
                    environment_audio_path = "stop_environment_detection.mp3"
                    pygame.mixer.music.load(environment_audio_path)
                    pygame.mixer.music.play()
                    print("Environment detection stopped")

                if command.lower() == "scan document":
                    run_document_system()

            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
    else:
        microphone_active = False

    # Check if the "o" key is pressed and object finder is active
    if run_object_finder and object_finder_active and keyboard.is_pressed('o'):
        # Add your object finder code here.
        print("Object finder is active. Searching for objects...")
        object_finder()

        run_object_finder = False  # Set the flag to False to run the object finder only once

    # Exit the loop when the 'q' key is pressed
    if keyboard.is_pressed('q'):
        pygame.mixer.quit()  # Quit pygame mixer before exiting
        break
