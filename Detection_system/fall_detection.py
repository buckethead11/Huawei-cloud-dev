import cv2

# Load YOLOv3 pre-trained model
model_weights = 'yolov3.weights'
model_config = 'yolov3.cfg'
net = cv2.dnn.readNet(model_weights, model_config)

# Get the names of output layers
layer_names = net.getUnconnectedOutLayersNames()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Parameters for motion analysis
prev_frame = None
motion_threshold = 10000  # Adjust this threshold based on your environment

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform human detection using the YOLOv3 model
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)  # Use all output layers

    # Process the detections
    for detection in detections:
        for obj in detection:
            class_id = int(obj[1])
            confidence = obj[2]

            if class_id == 0 and confidence > 0.2:  # 0 corresponds to person class
                box = obj[3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                x, y, w, h = box.astype(int)
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)  # Draw green rectangle around human

                # Rest of the fall detection logic...
                # Motion analysis, fall detection, etc.

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
