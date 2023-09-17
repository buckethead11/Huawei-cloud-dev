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
                {"role": "user", "content": f"Can you tell me the main point of this document strictly in no more than 20 words? I just need to know briefly what it is about:  \n\n{query}\n\nAnswer me with the paragraph directly"},
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
    frame = True

    '''
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
    '''


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
        ocr_response = ocr_instance.recognize_local_image("document_test.jpg")

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

run_document_system()