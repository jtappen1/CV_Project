import json
import torch
from PIL import Image

# Define the path to your VIA JSON file and images directory
via_json_path = '/guitar_annotations.json'
images_dir = 'path_to_images/'

# Load VIA annotations
with open(via_json_path) as f:
    via_data = json.load(f)

def parse_via_annotations(via_data, images_dir):
    dataset = []
    for image_id, image_data in via_data.items():
        image_path = images_dir + image_data['filename']
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        boxes = []
        labels = []

        for region in image_data['regions']:
            shape_attrs = region['shape_attributes']
            if shape_attrs['name'] == 'polygon':
                # Convert polygon to bounding box
                all_points_x = shape_attrs['all_points_x']
                all_points_y = shape_attrs['all_points_y']
                xmin = min(all_points_x)
                xmax = max(all_points_x)
                ymin = min(all_points_y)
                ymax = max(all_points_y)

                boxes.append([xmin / w, ymin / h, xmax / w, ymax / h])  # Normalized bounding box

                # Assuming a single class, e.g., "guitar" -> label = 1
                labels.append(1)

        # Convert everything to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
        }

        dataset.append((image, target))

    return dataset
