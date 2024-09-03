Steps: Detection Model Bounding box:
    1. Collect images that I want to train off of
    2.  Resize them to be the correct size
    3. Hand Label these images to detect guitar necks 

    Initally just do my guitar, but then later extend to more types and different things 

    start by 512 x 512
    gimp to crop and resize them down


plan: 
Mask R-CNN: Extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI).

Pros: High accuracy and widely used.
Cons: Computationally intensive.

Data Collection: Collect images containing the objects you want to segment. Ensure that the dataset has multiple instances of the same class if you’re focusing on multi-instance segmentation.

Annotations: Use annotation tools like COCO Annotator, LabelMe, or VIA to create instance masks. Each object instance should have a unique mask.

Dataset Format: Common formats include COCO and Pascal VOC. Ensure your dataset annotations are in the correct format for your chosen model.