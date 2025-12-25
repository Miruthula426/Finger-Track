# Finger-Track
FingerTrack is an AI-powered air-writing system that detects finger movements using a webcam, converts them into images, and recognizes handwritten letters in real time using deep learning.
# ✋ FingerTrack – Finger Traced Letter Recognition

FingerTrack is a computer vision + deep learning project that recognizes
letters written in the air using finger movement captured via webcam.

## 🚀 Features
- Real-time finger tracking using MediaPipe
- Air-writing using index finger
- Automatic image generation from finger paths
- CNN-based letter recognition
- Live prediction with confidence score

## 🧠 Technologies Used
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy

## 📁 Project Structure
Fingertrack/
├── data.py
├── train_model.py
├── m.py
├── set/ # generated (ignored)
├── models/ # generated (ignored)

shell
Copy code

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Collect data
bash
Copy code
python data.py
3. Train model
bash
Copy code
python train_model.py
4. Predict letters
bash
Copy code
python m.py
⚠️ Notes
The set/ and models/ folders are generated automatically.

At least 2 letters are required for training.

👩‍💻 Author
Miruthula Sakthivel

yaml
Copy code

---

## 4️⃣ Create `requirements.txt`

Create a file named **`requirements.txt`** and paste:

```txt
opencv-python
mediapipe
numpy
tensorflow
This is 🔑 for recruiters.
