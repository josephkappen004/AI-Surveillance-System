# face_detection.py
# Handles face registration, recognition, and face-mode processing

import cv2
import os
import numpy as np
from flask import request, jsonify
from werkzeug.utils import secure_filename
from insightface.app import FaceAnalysis

from face_db import (
    init_db,
    add_face,
    delete_face,
    get_faces,
    load_embeddings,
    cosine_similarity
)

from tampering_detection import is_dark, CFG

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading InsightFace model...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace model loaded successfully!")

init_db()
known_faces = load_embeddings()


def reload_known_faces():
    global known_faces
    known_faces = load_embeddings()
    return known_faces


def register_face_routes(app):

    @app.route("/upload", methods=["POST"])
    def upload():
        global known_faces
        name = request.form.get("name")
        file = request.files.get("image")
        if not name or not file:
            return jsonify({"success": False, "message": "Missing name or image"})
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"success": False, "message": "Invalid filename"})
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        try:
            img = cv2.imread(path)
            if img is None:
                os.remove(path)
                return jsonify({"success": False, "message": "Could not read image"})
            faces = face_app.get(img)
            if not faces:
                os.remove(path)
                return jsonify({"success": False, "message": "No face detected"})
            add_face(name, faces[0].embedding)
            known_faces = load_embeddings()
            return jsonify({"success": True})
        except Exception as e:
            print(f"Upload error: {e}")
            if os.path.exists(path):
                os.remove(path)
            return jsonify({"success": False, "message": f"Error: {str(e)}"})

    @app.route("/faces")
    def list_faces():
        return jsonify(get_faces())

    @app.route("/delete_face", methods=["POST"])
    def remove_face():
        global known_faces
        data = request.json
        if not data or "id" not in data:
            return jsonify({"success": False, "message": "Missing face ID"})
        delete_face(data.get("id"))
        known_faces = load_embeddings()
        return jsonify({"success": True})


def process_faces(frame, gray):
    try:
        _, brightness_check = is_dark(gray)
        if brightness_check <= CFG['darkness_threshold']:
            return []
        faces = face_app.get(frame)
        faces_data = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            label = "Unknown"
            color = (0, 0, 255)
            best_score = 0
            for known in known_faces:
                score = cosine_similarity(face.embedding, known["embedding"])
                if score > 0.45 and score > best_score:
                    label = known["name"]
                    color = (0, 255, 0)
                    best_score = score
            faces_data.append({'bbox': (x1, y1, x2, y2), 'label': label, 'color': color})
        return faces_data
    except Exception as e:
        print(f"Face detection error: {e}")
        return []


def draw_faces(frame, faces_data):
    for face_data in faces_data:
        x1, y1, x2, y2 = face_data['bbox']
        label = face_data['label']
        color = face_data['color']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        ls = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (x1, y1 - ls[1] - 10), (x1 + ls[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)