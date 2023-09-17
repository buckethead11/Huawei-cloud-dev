import openai
import pygame
from gtts import gTTS

# Set your OpenAI API key
api_key = "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"

# Initialize the OpenAI API client
openai.api_key = api_key

query = "Sprint PCS Phone Charges Sprint PCS www.sprintpcs.con customer account number billing period ending invoice date oled May6,2000 May8,2000 3 of PCS Phone Charges for $29.99 Service Plan -May 7to Jun.6 •$29 99 Monthly ServiceCharge • Includes 180 Minutes To Use Anytime Anywhere On The Sprint PCS Network •Caller ID,Call Waiting •Three-Way Calling • Sprint PCS Voiceinall •Detailed Billing PCS Phone Monthly Service Charges Description Charges S29.99 Service Plan-May 7to Jun.6 $29.99 Reminder:Service charges are billed one month in advance. Other PCS Options and Charges ltem Charges Long Distance May 7 to Jun.6 $0.00 Added Off-Peak Minutes: Evening/Weekend May 7 to Jun.6 $10.00 Actvation Fee May 5 $29.99 Sprint PCS Advantage Agreement Credit -$10.00 $25 Sprint PCS Service Credit -$25.00 Phone $199.99 Leather Case $14.99 Current Activity Charges for $249.96"

def gpt_document_interpreter(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a camera wearable that helps the blind user to interpret documents."},
            {"role": "user", "content": f"Here is the summarised version of the a document. Can you interpret it to me in one paragraph?:\n\n{query}\n\nAnswer me with the paragraph directly"},
        ],
    )

    content = response.choices[0].message.content
    return content

response = gpt_document_interpreter(query)
print(response)

# Convert the text response to audio using gTTS
tts = gTTS(text=response, lang='en')

# Save the audio to a temporary file
audio_path = "temp_audio.mp3"
tts.save(audio_path)

# Initialize Pygame Mixer
pygame.mixer.init()

# Load and play the audio
pygame.mixer.music.load(audio_path)
pygame.mixer.music.play()

# Wait for the audio to finish playing
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

# Delete the temporary audio file
import os
os.remove(audio_path)
