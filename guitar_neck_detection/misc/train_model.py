import os
import torch
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from guitar_neck_detection.misc.via_image_dataset import VIAImageDataset
from guitar_neck_detection.misc.parse_annotations import parse_via_annotations
import yaml
from models import create_model
from torchvision import transforms

def my_collate_fn(batch):
    return tuple(zip(*batch))

def main(config):
    model = create_model(num_classes=5 , size=512)

    via_json_path = config["via_json_path"]
    images_dir = config["images_dir"]

    dataset = parse_via_annotations(via_json_path=via_json_path, images_dir=images_dir)

    transform = transforms.Compose([
    # transforms.Resize((512, 512)),   # Resize to 512x512
    # transforms.RandomRotation(45),   # Rotate by 45 degrees
    # transforms.CenterCrop(512),      # Center crop the image to 512x512
    transforms.ToTensor(),           # Convert to tensor for further processing
])
    via_dataset = VIAImageDataset(dataset, transform=transform)

    # Create a DataLoader
    data_loader = DataLoader(via_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=my_collate_fn)

    # Move the model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Define optimizer and learning rate
    optimizer = optim.SGD(model.parameters(),config["learning_rate"], config["momentum"], config["weight_decay"])

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}")
    
    weights_path = os.path.join(config["weights_path"], config["model_name"])
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to weights directory")


if __name__ == "__main__":
    with open('guitar_neck_detection/configs/model_configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
        main(config=config)
