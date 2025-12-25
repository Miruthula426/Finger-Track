✋ FingerTrack – Air Writing Letter Recognition using AI

FingerTrack is an AI-powered computer vision project that recognizes letters written in the air using finger movements captured through a webcam. The system tracks finger motion in real time, converts it into an image, and predicts the written letter using a deep learning model.

This project demonstrates practical skills in Computer Vision, Machine Learning, and Human–Computer Interaction.

🚀 Key Features
✍️ Air-writing using index finger (no touch required)
🖐 Real-time hand & finger tracking using MediaPipe
🧠 Automatic image generation from finger paths
🔤 Letter recognition using CNN (TensorFlow / Keras)
📊 Confidence-based prediction output
💻 Runs in real time using a standard webcam

🧠 Technologies Used
Python
OpenCV
MediaPipe
TensorFlow / Keras
NumPy

📁 Project Structure
FingerTrack/
├── data.py              # Collects finger-traced letter images
├── train_model.py       # Trains CNN model on collected data
├── m.py                 # Real-time letter prediction
├── set/                 # Auto-generated training images (ignored in GitHub)
├── models/              # Auto-generated trained model (ignored in GitHub)
├── requirements.txt
└── README.md

⚙️ How It Works

Data Collection
User writes letters in the air using finger movement.
Motion path is captured and converted into grayscale images.
Model Training
A CNN model is trained on the generated images.
The trained model is saved for later use.
Prediction
The system predicts the written letter in real time.
Displays predicted character with confidence score.

▶️ How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Collect training data
python data.py
Write at least 2 different letters for training.
3️⃣ Train the model
python train_model.py
4️⃣ Run real-time prediction
python m.py

📌 Notes
The set/ and models/ folders are generated automatically.
These folders are excluded from GitHub using .gitignore.
A webcam is required for real-time input.

🎯 Applications

Touchless handwriting recognition
Assistive technology
Gesture-based input systems
AI-powered educational tools

👩‍💻 Author

Miruthula Sakthivel
Aspiring Data Scientist | AI & Computer Vision Enthusiast

📜 License
This project is licensed under the MIT License.
