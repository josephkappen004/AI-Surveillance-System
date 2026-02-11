# 🔍 AI Surveillance System

An end-to-end AI-powered surveillance platform featuring real-time object detection, face recognition, and high-performance video analysis with dynamic Fast/Accurate inference modes.

---

## 🚀 Features

### 🎥 Live Object Detection
- Open-vocabulary detection using **YOLO-World**
- Detect custom objects dynamically
- Real-time camera stream processing
- GPU (CUDA) acceleration support

### ⚡ Fast vs 🎯 Accurate Video Analysis
- **Fast Mode** → YOLOv8 Standard (COCO-based, high speed)
- **Accurate Mode** → YOLO-World (Open vocabulary)
- Frame sampling & resizing optimizations
- Timestamped detection results
- Screenshot capture with bounding boxes
- Progress tracking & background processing

### 👤 Face Recognition
- Powered by **InsightFace**
- Embedding extraction + cosine similarity matching
- Persistent face storage using SQLite
- Mask-resistant recognition capability

### 🎨 Color-Based Object Filtering
- HSV-based color analysis
- Detect specific colored objects (e.g., red car, blue bag)

### 🛡 Camera Tampering Detection
- Dark frame detection
- Blur detection
- Freeze detection
- Movement anomaly detection

---

## 🏗 System Architecture

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Flask (Python)
- **Computer Vision**: OpenCV
- **Object Detection**: YOLOv8 (Standard & World)
- **Face Recognition**: InsightFace
- **Database**: SQLite
- **GPU Support**: CUDA (if available)
- **Threading**: Background video processing

---

## ⚙️ Performance Optimization

- Dynamic backend switching (Fast / Accurate)
- Frame skipping strategy
- Resolution scaling
- Deduplication of repeated detections
- Multithreaded background video analysis

📈 Optimized processing time from:
**~150 seconds → ~10 seconds** for a 17-second video.



## 🛠 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Surveillance-System.git
cd AI-Surveillance-System 

### 2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Application
python app.py
4️⃣ Run Application
python app.py


Open in browser:

http://127.0.0.1:5000

🧠 Detection Modes
Mode	Model	Speed	Object Coverage
⚡ Fast	YOLOv8 Standard	High	COCO Classes
🎯 Accurate	YOLO-World	Medium	Open Vocabulary
💡 Use Cases

Smart CCTV Monitoring

Missing Person Alerts

Security Surveillance

Object Tracking

Retail Analytics

College Project / Research Prototype
