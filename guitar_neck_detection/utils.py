
import cv2


def draw_horizontal_lines(results, frame):
    for box in results.boxes:
        # Extract bounding box coordinates from the YOLO model output
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get the coordinates of the bounding box

        # Scale the bounding box coordinates back to the original frame size
        x1 = int(x1 * w / 640)
        y1 = int(y1 * h / 640)
        x2 = int(x2 * w / 640)
        y2 = int(y2 * h / 640)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        neck_width = (x2 - x1) // 3

        # Split the box into 3 horizontal sections
        for i in range(1, 3):
            line_x = x1 + i * neck_width
            cv2.line(frame, (line_x, y1), (line_x, y2), (255, 0, 0), 2)

            if i == 2:
                section_width = neck_width // 4
                for j in range (1,4):
                    fret_lines = x2 - j * section_width
                    cv2.line(frame, (fret_lines, y1), (fret_lines, y2), (0, 0, 255), 2)