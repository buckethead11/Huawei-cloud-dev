import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Set classes to detect (in this case, humans)
target_class = "person"

# Load image
image_path = "image.jpg"
image = cv2.imread(image_path)

# Check if image was loaded successfully
if image is None:
    print("Image not loaded or invalid path.")
    exit()

height, width, _ = image.shape

# Prepare image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set input for YOLO
net.setInput(blob)

# Get output layer names
output_layers = net.getUnconnectedOutLayersNames()

# Run YOLO forward pass
outs = net.forward(output_layers)

# Process detection results
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Check if the detected class is "person" and confidence > 0.5
        if classes[class_id] == target_class and confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate coordinates for bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Draw bounding box around the person
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detected humans
cv2.imshow("Human Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
