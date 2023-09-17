import paramiko
import time
import threading
import pygame
import speech_recognition as sr
import timer
import tempfile
import os
from gtts import gTTS
import re
import shutil

# Replace these with your ECS IP address and SSH credentials
ecs_ip = "190.92.207.98"
ecs_username = "root"
ecs_password = "Aintnowei69"  # Replace with your SSH password

# Command to start the chatbot using "bash run_chat" (modify as needed)
start_command = "bash run_chat"

# Variables to store chatbot responses and user inputs
chatbot_responses = ""
user_inputs = ["""Summarise to me quickly this summarised document in 1 short sentence, fewer than 15 words: NANYANG TECHNOLOGICAL UNIVERSITY SINGAPORE NTUStaffGames 2023 –SOCCER (7-A-SIDE)GAME Date：10 Nov 2023 Time:6.00pm-10.00pm Venue:SRC Main Ficld Staff eligibilin:Full time NTU& NIE staff only Games Format & Rules 1."""]
def text_to_speech(text):
    try:
        # Create a temporary directory for audio files
        temp_dir = tempfile.mkdtemp()

        # Generate a temporary audio file path
        temp_audio_file = os.path.join(temp_dir, "speech.mp3")

        # Convert text to speech using gTTS
        tts = gTTS(text=text, lang='en')

        # Save the speech to the temporary audio file
        tts.save(temp_audio_file)

        # Initialize Pygame mixer (if not already initialized)
        if not pygame.mixer.get_init():
            pygame.init()
            pygame.mixer.init()

        # Check if there's an active music stream (audio being played)
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()  # Stop the current playback
            pygame.mixer.music.unload()  # Unload the previous audio

        # Load and play the new audio file
        pygame.mixer.music.load(temp_audio_file)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)  # Sleep briefly to allow the audio to finish

        # Clean up by removing the temporary directory
        shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Function to read and print output from a channel
def read_channel(channel):
    global chatbot_responses
    while True:
        output = channel.recv(4096).decode()
        if not output:
            break
        chatbot_responses += output
        print(output, end='')

# Initialize Pygame for TTS and configure it
pygame.init()
pygame.mixer.init()
voice = pygame.mixer.music

# Create a function to speak the chatbot responses
def speak(text):
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.set_volume(1.0)
    pygame.mixer.music.play()
    pygame.mixer.music.set_volume(0.5)

# Establish an SSH connection
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Function to send user input to the chatbot
def send_user_input(input_text):
    channel.send(input_text + "\n")

# Function to capture user input
def get_user_input():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Press the spacebar and speak...")
        while True:
            try:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

                # Recognize the audio and send it to the chatbot
                user_input = recognizer.recognize_google(audio)
                print("You said:", user_input)
                send_user_input(user_input)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        ssh_client.connect(ecs_ip, username=ecs_username, password=ecs_password)
        print("Connection successful")

        # Start the chatbot
        channel = ssh_client.invoke_shell()
        channel.send(start_command + "\n")

        # Create separate threads for reading stdout and stderr
        stdout_thread = threading.Thread(target=read_channel, args=(channel,))
        stderr_thread = threading.Thread(target=read_channel, args=(channel.makefile_stderr(),))

        stdout_thread.start()
        stderr_thread.start()

        # Wait for a moment to ensure the chatbot starts (you may need to adjust the timing)
        time.sleep(5)  # Increase the sleep time if necessary

        # Check if the chatbot is ready
        while True:
            if "User:" in chatbot_responses:
                print("Chatbot is ready to receive input")
                chatbot_responses = ""

                # Send initial user messages programmatically
                for user_input in user_inputs:
                    send_user_input(user_input)

                # Start the user input thread
                user_input_thread = threading.Thread(target=get_user_input)
                user_input_thread.start()
                break

            # Capture the chatbot's responses to the user's messages and print them
        while True:
            if "Cloud:" and "User:" in chatbot_responses:
                chatbot_responses = chatbot_responses.strip()
                pattern = r'Cloud:(.*?)User:'
                matches = re.findall(pattern, chatbot_responses, re.DOTALL)
                extracted_text = [match.strip() for match in matches]
                paragraph = ' '.join(extracted_text)

                '''
                # Play audio asynchronously
                audio_path = "output.mp3"
                audio_thread = threading.Thread(target=text_to_speech, args=(paragraph,))
                audio_thread.start()
                '''

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        try:
            # Close the SSH session properly
            ssh_client.close()
        except Exception as close_exception:
            print(f"Error while closing SSH session: {str(close_exception)}")
