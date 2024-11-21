import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # Load the YOLO model (change to the correct path to your model weights)
    model = YOLO('/Users/jtappen/Projects/cv_project/guitar_neck_detection/runs/detect/train7/weights/last.pt')

    # Open webcam (0 is the default camera, change to 1 if you have an external webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        h, w = frame.shape[:2]

        if not ret:
            print("Error: Failed to capture image")
            break

        # Resize the frame to 640x640
        resized_frame = cv2.resize(frame, (640, 640))  # Resize the frame

        # Perform inference on the resized frame
        results = model.predict(resized_frame, imgsz=640, conf=0.7)

        # cv2.imshow('Live Webcam Inference')
        if isinstance(results, list):
            results = results[0]  # Get the first result if it's a list

        # # Plot the bounding boxes on the frame
        frame_with_boxes = results.plot()  # Automatically adds boxes on the frame

        max_x_box = None
        max_x_value = -float('inf')
        if hasattr(results, 'boxes'):
            for box in results.boxes:
                # Extract box coordinates and class index
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                cls_idx = int(box.cls.cpu().numpy())
                label = model.names[cls_idx]  # Get class name

                # Check if the label matches "odd_fret"
                if label == "odd_fret" and x_max > max_x_value:
                    max_x_value = x_max
                    max_x_box = (x_min, y_min, x_max, y_max)

        # Outline the "odd_fret" box with the largest x_max in red
        if max_x_box:
            resized_frame = draw_first_fret_boxes(resized_frame, max_x_box)

        resized_frame = cv2.resize(resized_frame, (1920, 1080))
        cv2.imshow('Live Webcam Inference', resized_frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def draw_first_fret_boxes(resized_frame, max_x_box):
    x_min, y_min, x_max, y_max = map(int, max_x_box)  # Convert to integers for drawing
    cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)  # Red outline
    box_width = x_max - x_min
    three_fret_box_width = x_max + 2*(box_width) - 15

    if(three_fret_box_width) < 640:
        padded_y_min = y_min + 10
        padded_y_max = y_max - 10
        padded_height = padded_y_max - padded_y_min  # Height after padding
        section_height = padded_height // 5  # Divide the remaining height into 5 equal sections

        # Draw 5 horizontal sections with padding
        for i in range(5):
            section_y_min = padded_y_min + i * section_height
            section_y_max = section_y_min + section_height

            cv2.rectangle(resized_frame, 
                        (x_min, section_y_min), 
                        (three_fret_box_width, section_y_max), 
                        (255, 0, 0), 2)  # Blue outline for sections
    
    return resized_frame

if __name__ == "__main__":
    main()