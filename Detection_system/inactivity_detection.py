import cv2
import numpy as np
import winsound

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Get index of "person" class
person_class_id = classes.index("person")

# Get the names of all layers in the network
layer_names = net.getLayerNames()

# Get the names of the output layers
output_layer_names = net.getUnconnectedOutLayersNames()

# Open a connection to the webcam (0 represents the default webcam)
cap = cv2.VideoCapture(0)

# Initialize fall detection variables
prev_h = None
fall_detected = False

# Initialize variables for inactivity detection
prev_frame = None
inactive_time = 0
inactivity_duration = 1 * 30  # 4 seconds at 30 frames per second
motion_threshold = 1000  # Adjust this threshold for sensitivity
beep_triggered = False  # Flag to track if beep has been triggered

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Detect "person" objects in the frame using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    # Lists to store detected person objects and their positions
    person_boxes = []

    # Process detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == person_class_id and confidence > 0.5:
                # Person detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                person_boxes.append([x, y, w, h])

                # Fall detection
                if prev_h is not None:
                    if h < prev_h * 0.65:  # Lower sensitivity threshold
                        fall_detected = True
                        winsound.Beep(1000, 500)  # Play beep sound (1000 Hz for 500 ms)
                        print("Fall detected")

                prev_h = h

    # Draw green rectangles around detected person objects
    for box in person_boxes:
        x, y, w, h = box
        color = (0, 255, 0)  # Green color for person
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Check for human inactivity using frame difference (MSE)
    if prev_frame is not None:
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        mse = np.mean(np.square(gray_diff))
        if mse < motion_threshold:
            if not beep_triggered and inactive_time >= inactivity_duration:
                print("Inactivity detected")
                winsound.Beep(1000, 500)  # Beep sound
                beep_triggered = True
            inactive_time = 0  # Reset inactivity timer
        else:
            inactive_time += 1
            beep_triggered = False

    prev_frame = frame

    # Display the frame
    cv2.imshow("Human Detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
