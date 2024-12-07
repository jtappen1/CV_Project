# Guitar Scale and Neck Detection Project

## Overview
This project is a computer vision program designed to detect the neck of a guitar in real-time from video webcam input. Using techniques such as object detection, the program is capable of identifying and drawing bounding boxes around the guitar neck and overlaying notes in real time onto the neck, making it useful for applications such as learning guitar scales, guitar playing analysis, or general image recognition tasks.

##  Features
Real-time detection of the guitar neck from a webcam feed.
Bounding box drawing on detected areas.
Guitar notes and strings for specific guitar scales overlaid on guitar neck.

## Requirements
### Backend
To run this project, you need to create an enviorment and install the following dependencies:

You can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

To run the server: 
```
python server.py
```

### Frontend

Prepare the frontend by running in the frontend directory:
```
npm install
```

## How It Works
1. The program processes the input (image or webcam feed) frame by frame.
2. It applies the YOLO-v8 (one-stage object detection) model to predict the presence of the guitar neck within the frame, as well as several key points around the neck to base detection off of.
3. Bounding boxes are drawn around the detected guitar neck.
4. Notes are calculated for the selected scale and displayed real time.
5. Results are displayed in real-time.

## Model Architecture
This program utilizes a pre-trained YOLO-v8 (one-stage object detection) model, fine-tuned for detecting guitar necks. The model takes an input image and generates bounding boxes along with classification scores for the regions that contain the guitar neck and specific key frets.



