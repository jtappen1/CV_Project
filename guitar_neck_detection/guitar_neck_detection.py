import cv2
import numpy as np

# Step 1: Load the image or capture from webcam
cap = cv2.VideoCapture(0)  # Use this line to capture from webcam

while True:
    # Step 2: Capture each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = cv2.createLineSegmentDetector()

# Step 4: Detect lines in the grayscale image
    lines = detector.detect(grayscale_img)[0]  # detect() returns a tuple, the first element is the lines

    detector.drawSegments(frame, lines)

# Step 5: Draw the detected line segments on the original image

    # Step 5: Detect contours
    # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # Optional: Filter contours to focus on the guitar neck
    # contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # # Step 6: Draw the contours on the original frame
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Step 7: Display the processed frame

    cv2.imshow("Guitar Neck", frame)

    # Step 8: Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 9: Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
# image = cv2.imread('guitar.jpg')
# while cap.isOpened():
#     success, image = cap.read()
# # Step 2: Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Step 3: Apply edge detection
#     edges = cv2.Canny(gray, 100, 200)

#     # Step 4: Detect contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # Optional: Filter contours based on area, aspect ratio, or other criteria
#     # You might want to filter contours based on size to isolate the guitar neck
#     contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

#     # Step 5: Draw the contours on the original image
#     cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
#     cv2.flip(image, 1)
#     cv2.imshow('Guitar Neck Outline', image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
