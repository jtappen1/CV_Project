import cv2
import mediapipe as mp

# def is_fist(hand_landmarks):
#     # Check if the fingertips are close to the palm
#     return (
#         all(distance(hand_landmarks[i], hand_landmarks[0]) < threshold for i in range(4, 21))  # Check fingertips
#     )

def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


def is_peace_sign(mp_hands, hand_landmarks):
    # Get landmarks
    landmarks = hand_landmarks.landmark

    # Define thumb, index, and middle landmarks
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    # Check if index and middle fingers are extended
    index_extended = distance(index_tip, index_pip) > distance(thumb_tip, index_pip) * 1.5
    middle_extended = distance(middle_tip, middle_pip) > distance(thumb_tip, middle_pip) * 1.5

    # Check if other fingers are curled
    other_fingers_curl = (
        distance(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP], landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]) < distance(landmarks[mp_hands.HandLandmark.RING_FINGER_DIP], landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]) * 1.5 and
        distance(landmarks[mp_hands.HandLandmark.PINKY_TIP], landmarks[mp_hands.HandLandmark.PINKY_DIP]) < distance(landmarks[mp_hands.HandLandmark.PINKY_DIP], landmarks[mp_hands.HandLandmark.PINKY_PIP]) * 1.5
    )

    # Define thresholds for finger extension
    return index_extended and middle_extended and other_fingers_curl