import cv2
import mediapipe as mp
import numpy as np
import os
import math
from tensorflow.keras.models import load_model

# Model path
MODEL_PATH = "C:/Fingertrack/models/fingertrace_model.keras"
if not os.path.exists(MODEL_PATH):
    print("❌ Model not found! Run train_model.py first.")
    exit()
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Labels from set folder
DATA_DIR = "set"
labels = sorted([name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))])
print("Labels:", labels)

# Utilities
def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def trace_to_image(points, size=256):
    canvas = np.ones((500,500), dtype=np.uint8)*255
    for i in range(1, len(points)):
        cv2.line(canvas, points[i-1], points[i], (0), 6)
    coords = np.column_stack(np.where(canvas < 255))
    if coords.size == 0:
        return None, None
    x, y, w, h = cv2.boundingRect(coords)
    letter = canvas[y:y+h, x:x+w]
    resized = cv2.resize(letter, (size, size))
    reshaped = resized.reshape(1, size, size, 1).astype("float32")/255.0
    return reshaped, resized

def predict_letter(image):
    predictions = model.predict(image)
    index = int(np.argmax(predictions))
    confidence = float(predictions[0][index])
    return labels[index], confidence

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
draw_points = []
recording = False
last_touch = False

print("🖐 Touch thumb + index to start tracing. Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_tip = thumb_tip = None
    touched = False

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            finger_tip = (int(hand.landmark[8].x*w), int(hand.landmark[8].y*h))
            thumb_tip = (int(hand.landmark[4].x*w), int(hand.landmark[4].y*h))
            if distance(finger_tip, thumb_tip) < 35:
                touched = True

    if touched and not last_touch:
        recording = True
        draw_points.clear()
        print("🎬 Recording started...")

    if recording and finger_tip:
        if not draw_points or draw_points[-1] != finger_tip:
            draw_points.append(finger_tip)
            cv2.circle(frame, finger_tip, 4, (255,0,0), -1)

    status = "✍️ Recording..." if recording else "Touch fingers to start"
    cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,0,255) if recording else (100,100,100), 2)

    cv2.imshow("Finger Trace Predictor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    last_touch = touched

cap.release()
cv2.destroyAllWindows()

# Predict
if draw_points:
    image, preview = trace_to_image(draw_points)
    if image is not None:
        predicted, confidence = predict_letter(image)
        print(f"✅ Predicted: {predicted} ({confidence*100:.2f}% confidence)")
        cv2.imshow(f"Prediction: {predicted}", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("⚠️ No valid trace for prediction.")
else:
    print("⚠️ No finger movement detected.")
