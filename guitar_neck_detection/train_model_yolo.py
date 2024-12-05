
import torch
from ultralytics import YOLO
import yaml

def main():

    model = YOLO('yolov8n.pt')
    yaml_path = "/Users/jtappen/Projects/cv_project/guitar_neck_detection/data/batch_final/data.yaml"
    
    print("Starting Training...")

    model.train(
        data=yaml_path,  # Path to your data.yaml
        epochs=15,  # Number of epochs to train
        imgsz=640,  # Image size (640 is common, can be adjusted)
        batch=3,  # Batch size (adjust based on your GPU memory)
        device='cpu',  # Use the GPU (0 is typically the first GPU)
        workers=4,  # Number of CPU workers for data loading
    )
    print("Training finished!")

    metrics = model.val()
    with open("evaluation_results.txt", "w") as f:
        f.write(str(metrics))

def run_inference():
    input_image_path = "/Users/jtappen/Projects/cv_project/guitar_neck_detection/data/batch_frets_1/train/images/Photo-on-11-17-24-at-6-25-PM-7_jpeg.rf.41650279ea4d3ffe263948bfdaf0182d.jpg"


    model = YOLO('/Users/jtappen/Projects/cv_project/guitar_neck_detection/runs/detect/train7/weights/last.pt')
    # Perform inference on the new image
    results = model.predict(input_image_path, imgsz=640, conf=0.3)

    # Print results
    print(f"Results for {input_image_path}:")
    for result in results:
        print(result.boxes.data)
        result.show()
    

if __name__ == "__main__":
    main()
