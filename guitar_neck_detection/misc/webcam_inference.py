import os
import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml
from models import create_model


def run_webcam_inference(config):

    model = create_model(num_classes=5, size=512)
    model_path = os.path.join(config["weights_path"], config["model_name"])
    # model_path = config["weights_path"]+config["model_name"]
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)  # Use this line to capture from webcam

    while True:
        # Step 2: Capture each frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Calculate the center of the image
        center_x, center_y = width // 2, height // 2

        # Define the size of the region to extract
        region_size = 640
        half_region = region_size // 2

        # Calculate the coordinates of the region
        x_min = max(center_x - half_region, 0)
        x_max = min(center_x + half_region, width)
        y_min = max(center_y - half_region, 0)
        y_max = min(center_y + half_region, height)

        # Extract the region
        extracted_region = frame[y_min:y_max, x_min:x_max]

        # If the extracted region is smaller than 512x512, pad it (optional)
        if extracted_region.shape[0] < region_size or extracted_region.shape[1] < region_size:
            # Create a new blank image with the desired size
            padded_region = 255 * np.ones((region_size, region_size, 3), dtype=np.uint8)
            # Calculate padding
            pad_y = (region_size - extracted_region.shape[0]) // 2
            pad_x = (region_size - extracted_region.shape[1]) // 2
            # Pad the extracted region
            padded_region[pad_y:pad_y+extracted_region.shape[0], pad_x:pad_x+extracted_region.shape[1]] = extracted_region
            extracted_region = padded_region

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
        image = extracted_region

        # Convert the image to a tensor
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to PyTorch Tensor
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Move the image to the device (CPU or GPU)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            predictions = model(image_tensor)

        # Extract the bounding boxes, labels, and scores from predictions
        boxes = predictions[0]['boxes'].cpu().numpy()   # Bounding boxes
        labels = predictions[0]['labels'].cpu().numpy()  # Class labels
        scores = predictions[0]['scores'].cpu().numpy()  # Confidence scores

        # Convert the image back to OpenCV format for display
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Draw bounding boxes
        for i, box in enumerate(boxes):
            if scores[i] > 0.5:  # Only draw boxes with confidence > 0.5
                image_width = 512
                image_height = 512  # Use image.shape for OpenCV

                x1, y1, x2, y2 = map(float, box)  # Convert coordinates to integers
                x1 = int(x1*image_width)
                x2 = int(x2 *image_width)
                y1 = int(y1 *image_height)
                y2 =  int(y2 * image_height)
                print("x1", x1)
                print("x2", x2)
                print( "y1", y1)
                print("y2", y2)

                
                # Draw the bounding box on the image
                cv2.rectangle(frame, (x1+x_min, y1+y_min), (x2+ x_min, y2), (255, 0, 0), 2)
                
                # Add class label and score
                label = f'{labels[i]}: {scores[i]:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image with bounding boxes
    # cv2.imwrite('output_image_with_boxes.jpg', image_np)
        cv2.imshow('Detected Guitar', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    with open('guitar_neck_detection/configs/model_configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
        run_webcam_inference(config=config)