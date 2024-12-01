import os
from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
from ultralytics import YOLO
from collections import defaultdict
from flask_cors import CORS
from fret_detection import draw_first_fret_boxes, generate_pentatonic_notes


app = Flask(__name__)
CORS(app)
guitar_notes = defaultdict(lambda: defaultdict(int))
current_scale = "F Minor Pentatonic"
saved_annotations = False  # Global variable to hold the saved annotated frame
current_annotations = []


# Available scales
scales = {
    "F Minor Pentatonic": ['F', 'G#', 'A#', 'C', 'D#'],
    "G Major Pentatonic": ['G', 'A', 'B', 'D', 'E'],
    "C Major Pentatonic": ['C', 'D', 'E', 'G', 'A']
}

# Webcam settings
cap = cv2.VideoCapture(0)
model = YOLO('/Users/jtappen/Projects/cv_project/guitar_neck_detection/runs/detect/train10/weights/last.pt')
guitar_notes = generate_pentatonic_notes(scales[current_scale])

def video_feed():
    global current_annotations, saved_annotations

    while True:
            
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Resize the frame to 640x640
        resized_frame = cv2.resize(frame, (640, 640))  # Resize the frame

        if saved_annotations:
            for annotation in current_annotations:
                if annotation["label"] == "string":
                    cv2.line(resized_frame, 
                    (annotation["x_min"], annotation["line_y"]), 
                    (annotation["x_max"], annotation["line_y"]), 
                    (0, 0, 0), 2)
                elif annotation["label"] == "note":
                    cv2.circle(resized_frame, (annotation["x_min"], annotation["line_y"]), 5 , (0, 255, 255), -1)
                elif annotation["label"] == "fret_box":
                    cv2.rectangle(resized_frame, (annotation["x_min"], annotation["y_min"]), (annotation["x_max"], annotation["y_max"]), (0, 0, 255), 4)  # Red outline
        else:
            current_annotations.clear()
            # Perform inference on the resized frame
            results = model.predict(resized_frame, imgsz=640, conf=0.75)

            # cv2.imshow('Live Webcam Inference')
            if isinstance(results, list):
                results = results[0]  # Get the first result if it's a list

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
                # cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Annotate the box with its index (number) beside it
                text_position = (x_max + 5, y_min + 20)  # Position text slightly to the right of the box
                cv2.putText(resized_frame, f"{idx + 1}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                curr_box = odd_fret_boxes[idx]
                if idx == 0 or idx== 4 or idx == 5:
                    resized_frame = draw_first_fret_boxes(current_annotations, guitar_notes, resized_frame, curr_box, 3, idx + 1)
                else:
                    resized_frame = draw_first_fret_boxes(current_annotations, guitar_notes, resized_frame, curr_box, 2, idx + 1)
                
            
        resized_frame = cv2.resize(resized_frame, (1920, 1080))
        if saved_annotations is not None:
                resized_frame = cv2.addWeighted(resized_frame, 1, saved_annotations, 1, 0)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from Flask!"})

@app.route('/video_feed')
def video_feed_route():
    return Response(video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_scale', methods=['POST'])
def update_scale():
    global current_scale, guitar_notes
    scale_name = request.json.get("scale")
    if scale_name in scales:
        current_scale = scale_name
        guitar_notes = generate_pentatonic_notes(scales[current_scale])
        return jsonify({"message": f"Scale updated to {scale_name}"})
    else:
        return jsonify({"error": "Invalid scale"}), 400

@app.route('/scales', methods=['GET'])
def get_scales():
    """
    Endpoint to fetch available scales and the currently selected scale.
    """
    try:
        return jsonify({
            "availableScales": list(scales.keys()),
            "currentScale": current_scale
        }), 200
    except Exception as e:
        return jsonify({"error": "Failed to fetch scales", "details": str(e)}), 500
    
@app.route('/')
def index():
    return render_template('index.html', scales=list(scales.keys()), current_scale=current_scale)

@app.route('/save_annotations', methods=['POST'])
def save_annotations():
    """
    Save the current frame with annotations.
    """
    global saved_annotations

    try:
        saved_annotations = not saved_annotations
        return jsonify({"message": "Annotations saved successfully!"}), 200
    except Exception as e:
        return jsonify({"error": "Failed to save annotations", "details": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000, debug=True)