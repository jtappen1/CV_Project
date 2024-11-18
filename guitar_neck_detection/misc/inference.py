import os
import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml
from models import create_model

# Load the trained model
# Replace this with the path to your trained model's weights
def run_inference(config):
    model = create_model(num_classes=5, size=512)
    model_path = os.path.join(config["weights_path"], config["model_name"])
    # model_path = config["weights_path"]+config["model_name"]
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    

    # Load the image (assuming it's 512x512) using PIL
    image = Image.open('guitar_neck_detection/test/RStelecaster5.jpeg')

    # Convert the image to a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to PyTorch Tensor
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image to the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
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
            image_width, image_height = image.size  # Use image.shape for OpenCV

            x1, y1, x2, y2 = map(float, box)  # Convert coordinates to integers
            print("Normalized")
            print("x1", x1)
            print("x2", x2)
            print( "y1", y1)
            print("y2", y2)
            x1 = int(x1*image_width)
            x2 = int(x2 *image_width)
            y1 = int(y1 *image_height)
            y2 =  int(y2 * image_height)
            print("Full Values")
            print("x1", x1)
            print("x2", x2)
            print( "y1", y1)
            print("y2", y2)

            
            # Draw the bounding box on the image
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add class label and score
            label = f'{labels[i]}: {scores[i]:.2f}'
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image with bounding boxes
    # cv2.imwrite('output_image_with_boxes_7.jpg', image_np)
    cv2.imshow('Detected Guitar', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    with open('guitar_neck_detection/configs/model_configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
        run_inference(config=config)