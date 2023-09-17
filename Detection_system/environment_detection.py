import openai
import cv2
import numpy as np
import urllib.request
import pygame
from pygame import mixer

from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkimage.v2 import ImageClient, RunImageTaggingRequest, ImageTaggingReq
from huaweicloudsdkimage.v2.region.image_region import ImageRegion

# Initialize Pygame
pygame.init()
mixer.init()

def yolo_img_detection(img_url):
    # Load YOLOv3 model and its configuration file
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # Load COCO labels (80 classes)
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")

    # Load image from URL
    url = img_url
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)

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
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the result
    #cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    final_result = ""

    # Print the detection results
    for result in detection_results:
        result_text = f"Label: {result['label']}, Confidence: {result['confidence']}, Bounding Box: {result['box']} \n"
        final_result += result_text

    return final_result

def huawei_image_detection(img_url):
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
            request.body = ImageTaggingReq(
                limit=50,
                threshold=75,
                language="en",
                url=img_url
            )
            response = client.run_image_tagging(request)

            if response.status_code == 200:
                #print("Response Content:")
                #print(response)
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
            {"role": "system", "content": """You are a wearable for the blind that helps them to describe/navigate their surroundings. Given below are 2 object/image recognition results given by 2 tools. Can you interpret and infer them to describe the environment/surrounding to the blind user in one paragraph. Explain it very simply\n\n
    """},
            {"role": "user", "content": detection_results},
        ],
    )

    content = response.choices[0].message.content
    return content


# Image URL
img_url = "https://www.asiaone.com/sites/default/files/styles/a1_600x316/public/original_images/Mar2016/20160319_pedesterians_st.jpg?itok=Da8US4DU"
yolo_response = yolo_img_detection(img_url)
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










