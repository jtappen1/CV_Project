import torch
import torch.optim as optim
import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torch.utils.data import DataLoader
import torchvision.transforms as T
from guitar_neck_detection.via_image_dataset import VIAImageDataset


# Load the pre-trained SSD model
model = ssd300_vgg16(pretrained=True)

# Modify the model to classify only one class (guitar neck) + background
num_classes = 4  # 1 class (guitar neck) + background
in_channels = model.head.classification_head[0].in_channels
num_anchors = model.head.classification_head[0].num_anchors

# Replace the classification head
model.head.classification_head[0] = torchvision.models.detection.ssd.SSDClassificationHead(in_channels, num_anchors, num_classes)

transform = T.Compose([
    T.Resize((300, 300)),  # SSD requires a fixed input size
    T.ToTensor(),
])

via_dataset = VIAImageDataset(dataset, transform=transform)

# Create a DataLoader
data_loader = DataLoader(via_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Move the model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define optimizer and learning rate
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

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

