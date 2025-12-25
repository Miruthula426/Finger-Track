import cv2
import mediapipe as mp
import numpy as np
import os
import math
from datetime import datetime

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
draw_points = []
recording = False
timestamp = ""

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def trace_to_image(points, size=256):
    canvas = np.ones((500, 500), dtype=np.uint8) * 255
    for i in range(1, len(points)):
        cv2.line(canvas, points[i - 1], points[i], (0), 6)
    coords = np.column_stack(np.where(canvas < 255))
    if coords.size == 0:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    letter = canvas[y:y+h, x:x+w]
    resized = cv2.resize(letter, (size, size), interpolation=cv2.INTER_AREA)
    return resized

def save_labeled_image(image):
    label = input("Enter the letter you wrote: ").strip().upper()
    if not label:
        print("Skipped: No label given.")
        return
    folder = os.path.join("set", label)
    os.makedirs(folder, exist_ok=True)
    count = len([f for f in os.listdir(folder) if f.endswith(".png")])
    filename = os.path.join(folder, f"{label.lower()}_{count+1}.png")
    cv2.imwrite(filename, image)
    print(f" Saved to: {filename}")

print("Touch thumb and index to start recording. Press 'q' to stop.")
last_touch = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_tip = thumb_tip = None
    touched = False

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            finger_tip = (int(hand.landmark[8].x * w), int(hand.landmark[8].y * h))
            thumb_tip = (int(hand.landmark[4].x * w), int(hand.landmark[4].y * h))
            if distance(finger_tip, thumb_tip) < 35:
                touched = True

    if touched and not last_touch:
        recording = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        draw_points.clear()
        print("Recording started...")

    if recording and finger_tip:
        if not draw_points or draw_points[-1] != finger_tip:
            draw_points.append(finger_tip)
            cv2.circle(frame, finger_tip, 4, (255, 0, 0), -1)

    if touched:
        cv2.putText(frame, " Recording...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Touch fingers to start", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    cv2.imshow("Finger Trace Recorder", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    last_touch = touched

cap.release()
cv2.destroyAllWindows()

if recording and draw_points:
    image = trace_to_image(draw_points)
    if image is not None:
        save_labeled_image(image)
    else:
        print("No drawing detected.")
