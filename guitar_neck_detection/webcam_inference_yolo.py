import cv2
from ultralytics import YOLO

# Load the YOLO model (change to the correct path to your model weights)
model = YOLO('/Users/jtappen/Projects/cv_project/guitar_neck_detection/runs/detect/train/weights/best.pt')

# Open webcam (0 is the default camera, change to 1 if you have an external webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Resize the frame to 640x640
    resized_frame = cv2.resize(frame, (640, 640))  # Resize the frame

    # Perform inference on the resized frame
    results = model.predict(resized_frame, imgsz=640, conf=0.13)

    # cv2.imshow('Live Webcam Inference')
    if isinstance(results, list):
        results = results[0]  # Get the first result if it's a list

    # Plot the bounding boxes on the frame
    frame_with_boxes = results.plot()  # Automatically adds boxes on the frame

    # Show the resulting frame
    cv2.imshow('Live Webcam Inference', frame_with_boxes)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()