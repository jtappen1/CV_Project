import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

guitar_notes = defaultdict(lambda: defaultdict(int))

def main():
    # Load the YOLO model (change to the correct path to your model weights)
    # model = YOLO('/Users/jtappen/Projects/cv_project/guitar_neck_detection/runs/detect/train7/weights/last.pt')
    model = YOLO('/Users/jtappen/Projects/cv_project/guitar_neck_detection/runs/detect/train10/weights/last.pt')
    # Open webcam (0 is the default camera, change to 1 if you have an external webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    generate_pentatonic_notes()

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
        results = model.predict(resized_frame, imgsz=640, conf=0.75)

        # cv2.imshow('Live Webcam Inference')
        if isinstance(results, list):
            results = results[0]  # Get the first result if it's a list

        # # Plot the bounding boxes on the frame
        frame_with_boxes = results.plot()  # Automatically adds boxes on the frame

        odd_fret_boxes = []  # List to store all "odd_fret" boxes
        if hasattr(results, 'boxes'):
            for box in results.boxes:
                # Extract box coordinates and class index
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                cls_idx = int(box.cls.cpu().numpy())
                label = model.names[cls_idx]  # Get class name

                # Check if the label matches "odd_fret"
                if label == "odd_fret":
                    odd_fret_boxes.append((x_min, y_min, x_max, y_max))

        # Sort the boxes by x_max in descending order
        odd_fret_boxes = sorted(odd_fret_boxes, key=lambda box: box[2], reverse=True)
        for idx in range(len(odd_fret_boxes)):
            x_min, y_min, x_max, y_max = map(int, odd_fret_boxes[idx]) 
            cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Annotate the box with its index (number) beside it
            text_position = (x_max + 5, y_min + 20)  # Position text slightly to the right of the box
            cv2.putText(resized_frame, f"{idx + 1}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            curr_box = odd_fret_boxes[idx]
            if idx == 0 or idx== 4 or idx == 5:
                resized_frame = draw_first_fret_boxes(resized_frame, curr_box, 3, idx + 1)
            else:
                resized_frame = draw_first_fret_boxes(resized_frame, curr_box, 2, idx + 1)
            

        resized_frame = cv2.resize(resized_frame, (1920, 1080))
        cv2.imshow('Live Webcam Inference', resized_frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def draw_first_fret_boxes(resized_frame, max_x_box, ratio, fret):
    x_min, y_min, x_max, y_max = map(int, max_x_box)  # Convert to integers for drawing
    cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)  # Red outline
    box_width = x_max - x_min
    three_fret_box_width = x_min + ratio*(box_width) - 15

    if(three_fret_box_width) < 640:
        return draw_string_lines(resized_frame, x_min, three_fret_box_width, y_min, y_max, fret, ratio)

    
    return resized_frame


def draw_string_lines(resized_frame, x_min, x_max, y_min, y_max, fret, ratio):
    padded_y_min = y_min + 10
    padded_y_max = y_max - 10
    padded_height = padded_y_max - padded_y_min  # Height after padding
    section_height = padded_height // 5  # Divide the remaining height into 5 equal sections

    for i in range(6):
        line_y = padded_y_min + i * section_height
        resized_frame = draw_notes_on_neck(resized_frame, fret, ratio, line_y, x_min, x_max, i)
        cv2.line(resized_frame, 
                 (x_min, line_y), 
                 (x_max, line_y), 
                 (255, 0, 0), 2)  # Blue line with thickness 2
       
        
    return resized_frame

def generate_pentatonic_notes():
    # Define the F minor pentatonic scale
    pentatonic_notes = ['F', 'G#', 'A#', 'C', 'D#']
    
    # Define the chromatic scale (12 notes)
    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Define the open string notes for standard tuning (EADGBE)
    open_strings = ['E', 'A', 'D', 'G', 'B', 'E']
    
    for string_num, open_note in enumerate(open_strings, start=0):
        # Find the starting index of the open note in the chromatic scale
        start_index = chromatic_scale.index(open_note)
        
        # Generate notes for the frets (1-24) and filter for pentatonic notes or set to 0
        for fret in range(1, 25):  # Start from fret 1
            note = chromatic_scale[(start_index + fret) % len(chromatic_scale)]
            
            # If the note is in the pentatonic scale, store the note, else store 0
            if note in pentatonic_notes:
                guitar_notes[string_num][fret] = note
            else:
                guitar_notes[string_num][fret] = 0
    

def draw_notes_on_neck(resized_frame, fret, ratio, line_y, x_min, x_max, string_num):
    if fret >= 2:
        fret *= 2
    fret_width = (x_max - x_min) // ratio
    
    for i in range(0, ratio):
        note = guitar_notes[string_num ][(fret + i)]
        if note != 0:
            cv2.circle(resized_frame,((x_max - (i * fret_width)- fret_width)+ (fret_width//2), line_y), 5 , (0, 255, 255), -1)

    return resized_frame
        
        

if __name__ == "__main__":
    main()