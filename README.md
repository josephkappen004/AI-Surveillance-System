# 🔍 AI Surveillance System

An end-to-end AI-powered surveillance platform supporting real-time object detection, face recognition, and optimized video analysis with dynamic Fast/Accurate inference modes.

---

## 🚀 Features

### 🎥 Live Object Detection
- Open-vocabulary detection using YOLO-World  
- Real-time camera stream processing  
- GPU (CUDA) acceleration support  
- Dynamic object search capability  

### ⚡ Fast vs 🎯 Accurate Video Analysis
- **Fast Mode** → YOLOv8 Standard (COCO-based, high speed)  
- **Accurate Mode** → YOLO-World (Open vocabulary)  
- Frame skipping and resolution scaling  
- Timestamped detections  
- Screenshot capture with bounding boxes  
- Background processing with progress tracking  

### 👤 Face Recognition
- Powered by InsightFace  
- Embedding extraction and cosine similarity matching  
- SQLite-based face storage  

### 🎨 Color-Based Filtering
- HSV-based color detection  
- Detect specific colored objects  

### 🛡 Tampering Detection
- Dark frame detection  
- Blur detection  
- Freeze detection  
- Movement anomaly detection  

---

## 🏗 Tech Stack

### Backend
- Python  
- Flask  
- OpenCV  
- SQLite  
- Multithreading  

### AI / Machine Learning
- YOLOv8 (YOLO-Standard & YOLO-World)  
- InsightFace  
- Cosine Similarity Matching  
- CUDA (GPU Acceleration)  

### Frontend
- HTML5  
- CSS3  
- JavaScript (Fetch API)  

---

## ⚙️ Performance Optimization

- Dynamic backend switching (Fast / Accurate)  
- Frame sampling strategy  
- Resolution scaling  
- Detection deduplication  
- Multithreaded video processing  

📈 Reduced video processing time from ~150 seconds to ~10 seconds for a 17-second video.

---

# 🛠 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Surveillance-System.git
cd AI-Surveillance-System
```
## 2️⃣ Create And Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate 
```
## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt

```
## 4️⃣ Run Application

```bash
4️⃣ Run Application
```
