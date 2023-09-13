import speech_recognition
import pyttsx3
from config import api_key
from geopy.geocoders import Nominatim
import requests
import smtplib
from config import api_key
import time
import re

# Initialize
recognizer = speech_recognition.Recognizer()
engine = pyttsx3.init()
url = "https://maps.googleapis.com/maps/api/directions/json?"
start = 'NTU hall 12'

# Speech to text
def speech_to_text(recognizer):
    try:
        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)
            
            text = recognizer.recognize_google(audio)
            text = text.lower()
            
            print(f"Recognized {text}")
        
    except speech_recognition.UnknownValueError():
        recognizer = speech_recognition.Recognizer()

    return text

def get_directions(start, destination):
    params = {
    "origin": start,
    "destination": destination,
    "mode": "walking",  # Use "walking" for walking directions
    "key": api_key,
    }

    r = requests.get(url, params=params)

    data = r.json()
    return data

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

if __name__ == '__main__':
    engine.say('What is your destination')
    engine.runAndWait()
    destination = speech_to_text(recognizer)
    directions = get_directions(start, destination)
    if directions['status'] == 'OK':
        # Extract and print the walking directions
        for step in directions['routes'][0]['legs'][0]['steps']:
            instruction = step['html_instructions']
            clean_instruction = remove_html_tags(instruction)  # Remove HTML tags
            engine.say(clean_instruction)  # Speak the cleaned direction instruction
            print(instruction)  # Print the instruction to the console
            engine.runAndWait()  # Wait for the speech to finish
    else:
        print("Error:", directions['status'])
