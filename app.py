import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Aircraft Recognition Dashboard",
    page_icon="🛫",
    layout="wide"
)

# =========================================================
# CONSTANTS
# =========================================================
DETECTOR_PATH = "runs/detect/AircraftDetector3/weights/best.pt"
CLASSIFIER_PATH = "models/aircraft_classifier.h5"
CLASS_NAMES = sorted(os.listdir("dataset/classifier_dataset/train"))
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    detector = YOLO(DETECTOR_PATH)
    classifier = load_model(CLASSIFIER_PATH)
    return detector, classifier

detector, classifier = load_models()


# =========================================================
# PROFESSIONAL UI THEME
# =========================================================
st.markdown("""
<style>
/* MAIN BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #000428, #004e92);
    font-family: 'Segoe UI', sans-serif;
    color: white;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.35);
    backdrop-filter: blur(8px);
}
[data-testid="stSidebar"] * {
    color: #E6E6E6 !important;
}

/* HEADER */
.dashboard-title {
    font-size: 48px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #00eaff, #7df9ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* CARD STYLING */
.card {
    padding: 25px;
    border-radius: 18px;
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 25px;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(90deg, #00eaff, #0099ff);
    color: black;
    font-weight: bold;
    padding: 10px 30px;
    border-radius: 10px;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #7df9ff, #00eaff);
    cursor: pointer;
}

/* FILE UPLOAD BOX */
.upload-box {
    border: 2px dashed #00eaff;
    padding: 25px;
    border-radius: 14px;
    background: rgba(255,255,255,0.08);
    text-align: center;
}

/* FOOTER */
.footer {
    text-align: center;
    font-size: 0.8rem;
    margin-top: 35px;
    color: #DDD;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
st.markdown("<div class='dashboard-title'>AI Aircraft Recognition Dashboard</div>", unsafe_allow_html=True)
st.write("### 🛫 Real-time Detection • 📷 Image Recognition • 🎥 Video Processing")


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("🧭 Navigation")
mode = st.sidebar.radio(
    "Select Mode",
    ["🖼️ Image", "🎥 Video", "📡 Real-Time Camera"]
)

st.sidebar.write("---")
st.sidebar.subheader("📘 Model Details")
st.sidebar.write("• Detector: YOLOv8n")  
st.sidebar.write("• Classifier: CNN (Keras)")  
st.sidebar.write(f"• Classes Loaded: **{len(CLASS_NAMES)}**")  


# =========================================================
# CLASSIFICATION FUNCTION
# =========================================================
def classify_crop(crop):
    resized = cv2.resize(crop, (224, 224))
    arr = image.img_to_array(resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = np.argmax(classifier.predict(arr, verbose=False), axis=1)[0]
    return CLASS_NAMES[pred]


# =========================================================
# IMAGE MODE
# =========================================================
if mode == "🖼️ Image":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📸 Upload Image")

    uploaded = st.file_uploader("Upload JPG / PNG", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.info("Running Detection...")
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
        results = detector(img, conf=0.25, verbose=False)[0]

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue

            label = classify_crop(crop)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 200), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

        output_path = os.path.join(OUTPUT_DIR, "image_result.jpg")
        cv2.imwrite(output_path, img)

        st.subheader("🔍 Result Preview")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.download_button("⬇ Download Processed Image", open(output_path, "rb"), file_name="result.jpg")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# VIDEO MODE
# =========================================================
elif mode == "🎥 Video":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎞 Upload a Video")

    uploaded = st.file_uploader("Upload MP4/AVI", type=["mp4", "avi", "mov"])

    if uploaded:
        st.info("Processing video...")

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = os.path.join(OUTPUT_DIR, "video_result.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        progress = st.progress(0)

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_id += 1

            results = detector(frame, conf=0.25, verbose=False)[0]
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue

                label = classify_crop(crop)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

            out.write(frame)
            progress.progress(frame_id / total_frames)

        cap.release()
        out.release()

        st.success(" Video Processing Completed!")
        st.video(output_path)
        st.download_button("⬇ Download Video", open(output_path, "rb"), file_name="result_video.mp4")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# REAL-TIME CAMERA
# =========================================================
elif mode == "📡 Real-Time Camera":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📡 Live Camera Detection")
    start = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while start:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected!")
            break

        results = detector(frame, conf=0.25, verbose=False)[0]
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            label = classify_crop(crop)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("<div class='footer'>⚡ Powered by YOLOv8 + TensorFlow CNN</div>", unsafe_allow_html=True)
