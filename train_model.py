import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Safe model folder
MODEL_DIR = "C:/Fingertrack/models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "fingertrace_model.keras")

# Load dataset
DATA_DIR = "set"
if not os.path.exists(DATA_DIR):
    print("❌ No 'set' folder found. Run data.py first.")
    exit()

class_names = sorted([name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))])
if len(class_names) < 2:
    print("❌ Need at least 2 classes (letters) to train.")
    exit()

data = []
labels = []

for idx, name in enumerate(class_names):
    folder = os.path.join(DATA_DIR, name)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        data.append(img)
        labels.append(idx)

data = np.array(data).reshape(-1, 256, 256, 1) / 255.0
labels = to_categorical(labels, num_classes=len(class_names))

print(f"✅ Loaded {len(data)} images from {len(class_names)} classes.")

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(data, labels, epochs=15, batch_size=8, verbose=1)

# Save model safely
model.save(MODEL_PATH)
print(f"✅ Model trained and saved at: {MODEL_PATH}")
