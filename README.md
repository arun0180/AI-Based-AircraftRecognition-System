✈️ AI-Based Aircraft Recognition System
Aircraft Detection using YOLOv8 and Classification using Keras

>>>Google Drive Link Of Full Project:
   https://drive.google.com/drive/folders/149i_iHdHYb1TVz3OH6jZMLhLpTInKpq3

📌 Project Overview
Outputs:
1) Dashboard:
![Homepage](https://raw.githubusercontent.com/arun0180/AI-Based-AircraftRecognition-System/main/dashboard.jpg
)
2) Detection and Classification Output Images
![Homepage](https://raw.githubusercontent.com/arun0180/AI-Based-AircraftRecognition-System/main/ATR_72_29.jpg
)
![Homepage](https://raw.githubusercontent.com/arun0180/AI-Based-AircraftRecognition-System/main/MiG-29_80.jpg
)
![Homepage](https://raw.githubusercontent.com/arun0180/AI-Based-AircraftRecognition-System/main/Su-30_24.jpg
)
![Homepage](https://raw.githubusercontent.com/arun0180/AI-Based-AircraftRecognition-System/main/MQ-9_Reaper_Drone_7.jpg
)
![Homepage](https://raw.githubusercontent.com/arun0180/AI-Based-AircraftRecognition-System/main/Su-30_25.jpg
)


This project presents an AI-based Aircraft Recognition System capable of detecting aircraft in images, videos, and real-time camera feeds and identifying their type automatically.

The system uses:

YOLOv8n for aircraft detection

Keras for aircraft classification

The detection model locates aircraft using bounding boxes, and the classification model identifies the aircraft type with confidence scores.

This system can be applied in:

Airport surveillance

Airspace monitoring

Defense and security systems

UAV and drone tracking

Aviation research and analytics

🚀 Features

✔ Aircraft detection using YOLOv8
✔ Aircraft classification using ResNet-50
✔ Image input support
✔ Video input support
✔ Real-time webcam detection
✔ Web-based interface using Flask
✔ Modular architecture (detection + classification pipeline)

🧠 Models Used
1️⃣ YOLOv8n (Detection Model)

Detects aircraft in an image

Outputs bounding box coordinates

Provides confidence score

Fast and suitable for real-time applications

2️⃣ Keras(Classification Model)

Deep Convolutional Neural Network

Uses transfer learning

Classifies cropped aircraft images

Handles fine-grained aircraft differences

📂 Project Structure
AI-Aircraft-Recognition-System/
│
├── app.py
├── scripts/
│   ├── train_detector.py
│   ├── train_classifier.py
│   ├── recognize_image.py
│   └── detect_video.py
│
├── models/               # Model architecture files
├── test_images/          # Sample test images
├── data.yaml             # YOLO dataset configuration
├── requirements.txt
├── README.md
└── .gitignore
📊 Dataset Information

Aircraft images were collected from:

FGVC-Aircraft Dataset

FAIR1M Dataset

UCAS-AOD Dataset

Public aviation image sources

Dataset includes:

Commercial aircraft

Military aircraft

UAV types

Different angles and lighting conditions

The dataset was split into:

Training set

Validation set

Testing set

Aircraft were annotated using LabelImg in YOLO format.

⚙️ Installation Guide
🔹 Step 1: Clone Repository
git clone https://github.com/yourusername/AI-Aircraft-Recognition-System.git
cd AI-Aircraft-Recognition-System
🔹 Step 2: Create Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate   # Windows
🔹 Step 3: Install Requirements
pip install -r requirements.txt
▶️ How to Run the Project
🔹 Run Web Application
python app.py

Open browser and go to:

http://127.0.0.1:5000
🔹 Train YOLOv8n Detector
python scripts/train_detector.py
🔹 Train Classifier
python scripts/train_classifier.py
🔹 Run Image Detection
python scripts/recognize_image.py
🔹 Run Video Detection
python scripts/detect_video.py
📈 Performance Metrics

The system was evaluated using:

Accuracy

Precision

Recall

F1-score

mAP (mean Average Precision)

Confusion Matrix

Results show:

Accurate detection in complex backgrounds

Reliable classification across multiple aircraft types

Real-time performance with GPU support

💻 Technologies Used

Python

PyTorch

Ultralytics YOLOv8

OpenCV

NumPy

Matplotlib

Flask

🔥 Key Advantages

End-to-end aircraft recognition

Real-time capability

Scalable architecture

Modular detection + classification design

Easy deployment

⚠️ Limitations

Performance may reduce for very small aircraft

Visually similar aircraft may cause minor confusion

Real-time performance depends on hardware

🔮 Future Enhancements

Add aircraft tracking (DeepSORT)

Expand dataset with more aircraft types

Deploy on edge devices (Jetson Nano)

Integrate satellite image support

Add Explainable AI (Grad-CAM visualization)

📌 Applications

Airport security monitoring

Airspace surveillance

Defense monitoring systems

UAV monitoring

Aviation analytics

📜 License

This project is developed for academic and research purposes.

👨‍💻 Author

Developed as part of a Mini Project
Department of Computer Science and Engineering
BMS Institute of Technology & Management

⭐ If You Found This Useful

Give this repository a ⭐ on GitHub.





