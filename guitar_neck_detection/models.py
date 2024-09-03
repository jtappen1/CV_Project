import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16

# Load the pre-trained SSD model
model = ssd300_vgg16(pretrained=True)

# Modify the model to classify only one class (guitar neck) + background
num_classes = 4  # 1 class (guitar neck) + background
in_channels = model.head.classification_head[0].in_channels
num_anchors = model.head.classification_head[0].num_anchors

# Replace the classification head
model.head.classification_head[0] = torchvision.models.detection.ssd.SSDClassificationHead(in_channels, num_anchors, num_classes)
