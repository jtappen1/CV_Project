import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from ultralytics import YOLO
from guitar_neck_detection.misc.via_image_dataset import VIAImageDataset
from guitar_neck_detection.misc.parse_annotations import parse_via_annotations
import yaml
from models import create_model
from torchvision import transforms
from PIL import Image
def my_collate_fn(batch):
    return tuple(zip(*batch))

def main():

    model = YOLO('yolov8n.pt')
    yaml_path = "/Users/jtappen/Projects/cv_project/guitar_neck_detection/data/image_dataset_1/data.yaml"
    with open('/Users/jtappen/Projects/cv_project/guitar_neck_detection/data/image_dataset_1/data.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    print("Starting Training...")

    model.train(
        data=yaml_path,  # Path to your data.yaml
        epochs=10,  # Number of epochs to train
        imgsz=640,  # Image size (640 is common, can be adjusted)
        batch=3,  # Batch size (adjust based on your GPU memory)
        device='cpu',  # Use the GPU (0 is typically the first GPU)
        workers=4,  # Number of CPU workers for data loading
    )
    print("Training finished!")

    metrics = model.val()
    with open("evaluation_results.txt", "w") as f:
        f.write(str(metrics))

    source_path = "/Users/jtappen/Projects/cv_project/guitar_neck_detection/data/image_dataset_1/test/images"  # Path to your test images
    results = model.predict(source=source_path, imgsz=640)
    print("Inference completed, results saved!")

    torch.save(model.state_dict(), config["weights_path"])

def run_inference():
    input_image_path = "/Users/jtappen/Projects/cv_project/guitar_neck_detection/data/image_dataset_1/test/images/Photo-on-11-17-24-at-4-20-PM-3_jpeg.rf.947dc58620b3e8e582d69da9e49391a9.jpg"
    # image = Image.open(input_image_path)
    # image_resized = image.resize((640, 640))
    # image_np = np.array(image_resized)
    
    # # YOLO expects BGR format for OpenCV, so we convert RGB -> BGR
    # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


    model = YOLO('/Users/jtappen/Projects/cv_project/guitar_neck_detection/runs/detect/train/weights/best.pt')
    # Perform inference on the new image
    results = model.predict(input_image_path, imgsz=640, conf=0.06)

    # Print results
    print(f"Results for {input_image_path}:")
    for result in results:
        print(result.boxes.data)
        result.show()
        # Draw bounding boxes on the image
    #     for box in result.boxes.xywh:  # xywh is in (x_center, y_center, width, height)
    #         x_center, y_center, width, height = box

    #         # Convert to pixel coordinates
    #         x1 = int((x_center - width / 2) * image_bgr.shape[1])
    #         y1 = int((y_center - height / 2) * image_bgr.shape[0])
    #         x2 = int((x_center + width / 2) * image_bgr.shape[1])
    #         y2 = int((y_center + height / 2) * image_bgr.shape[0])

    #         # Draw bounding box on the image
    #         cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color for the box

    #         # Optionally, add label and confidence (if available)
    #         label = result.names[int(result.boxes.cls[0])]  # Class name
    #         confidence = result.boxes.conf[0]  # Confidence score
    #         cv2.putText(image_bgr, f"{label} {confidence:.2f}", (x1, y1 - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # # Save or display the output image

    # # Optionally, display the image
    # cv2.imshow("Prediction", image_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # main()
    run_inference()
