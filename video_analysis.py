# video_analysis.py
# Handles uploaded video analysis - scans for target objects frame by frame

import cv2
import os
import time
import uuid
import threading
import numpy as np
from flask import request, jsonify, send_from_directory

from object_detection import (
    detect_objects_multi,
    matches_color,
    get_dominant_color_hsv,
    get_targets_snapshot,
    COLOR_HSV_RANGES,
    YOLO_COCO_CLASSES,
    DETECTION_BACKEND,
    ALL_OBJECTS,
)

from tampering_detection import is_dark, CFG

# -------------------- Config --------------------
VIDEO_UPLOAD_FOLDER = "video_uploads"
VIDEO_RESULTS_FOLDER = "video_results"
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_RESULTS_FOLDER, exist_ok=True)

MAX_VIDEO_SIZE_MB = 500
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
FRAME_SKIP = 5  # Analyze every Nth frame for speed
CONFIDENCE_THRESHOLD = 0.3
# Minimum seconds between detections to avoid duplicate timestamps
DEDUP_SECONDS = 2.0

# -------------------- Active Jobs --------------------
# Each job: {id, status, progress, total_frames, processed_frames, results, error, targets, filename}
active_jobs = {}
active_jobs_lock = threading.Lock()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def generate_job_id():
    return str(uuid.uuid4())[:12]


# -------------------- Video Processing --------------------

def analyze_video_thread(job_id, video_path, targets):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with active_jobs_lock:
                active_jobs[job_id]["status"] = "error"
                active_jobs[job_id]["error"] = "Could not open video file"
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        with active_jobs_lock:
            active_jobs[job_id].update({
                "total_frames": total_frames,
                "fps": fps,
                "duration": duration,
                "status": "processing"
            })

        # ---------------- OPTIMIZATION CONFIG ----------------
        DETECT_SIZE = 640     # resize shortest side to this
        FRAME_SKIP_LOCAL = FRAME_SKIP
        # -----------------------------------------------------

        class_names = list({t["object"] for t in targets})

        target_lookup = {}
        for t in targets:
            target_lookup.setdefault(t["object"], []).append(t)

        results = []
        last_detection_times = {}
        frame_number = 0

        while True:
            with active_jobs_lock:
                if active_jobs[job_id]["status"] == "cancelled":
                    break

            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number % FRAME_SKIP_LOCAL != 0:
                continue

            with active_jobs_lock:
                active_jobs[job_id]["processed_frames"] = frame_number
                active_jobs[job_id]["progress"] = min(
                    round((frame_number / max(total_frames, 1)) * 100, 1), 100
                )

            timestamp_sec = frame_number / fps

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dark, _ = is_dark(gray)
            if dark:
                continue

            # ---------- Resize for faster detection ----------
            h, w = frame.shape[:2]
            scale = DETECT_SIZE / min(h, w)
            if scale < 1.0:
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                small = frame
                scale = 1.0

            try:
                detections = detect_objects_multi(
                    small,
                    class_names,
                    CONFIDENCE_THRESHOLD
                )
            except Exception as e:
                print(f"Video detection error at frame {frame_number}: {e}")
                continue

            if not detections:
                continue

            inv_scale = 1.0 / scale

            for det in detections:
                # Scale bbox back to original frame
                x1, y1, x2, y2 = [int(v * inv_scale) for v in det["bbox"]]
                det_label = det["label"].lower()
                det_conf = det["confidence"]

                matching_targets = target_lookup.get(det_label, [])
                if not matching_targets:
                    continue

                for target in matching_targets:
                    t_color = target["color"]

                    # ---------- Color check only if needed ----------
                    color_info = ""
                    if t_color:
                        ok, pct, _ = matches_color(frame, (x1, y1, x2, y2), t_color)
                        if not ok:
                            continue
                        color_info = f"{t_color} ({pct:.0%})"

                    dedup_key = f"{target['object']}_{t_color}"
                    last_ts = last_detection_times.get(dedup_key, -999)
                    if timestamp_sec - last_ts < DEDUP_SECONDS:
                        continue
                    last_detection_times[dedup_key] = timestamp_sec

                    screenshot = frame.copy()
                    cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    label = f"{det_label} ({det_conf:.0%})"
                    if color_info:
                        label += f" [{color_info}]"

                    cv2.putText(
                        screenshot, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
                    )

                    cv2.putText(
                        screenshot,
                        f"Time: {format_timestamp(timestamp_sec)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                    fname = f"{job_id}_{len(results)}.jpg"
                    cv2.imwrite(
                        os.path.join(VIDEO_RESULTS_FOLDER, fname),
                        screenshot,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    )

                    results.append({
                        "object": target["object"],
                        "color_filter": t_color if t_color else "any",
                        "confidence": round(det_conf, 3),
                        "timestamp_sec": round(timestamp_sec, 2),
                        "timestamp": format_timestamp(timestamp_sec),
                        "frame_number": frame_number,
                        "bbox": [x1, y1, x2, y2],
                        "screenshot": fname
                    })

                    with active_jobs_lock:
                        active_jobs[job_id]["results"] = results.copy()

        cap.release()

        with active_jobs_lock:
            if active_jobs[job_id]["status"] != "cancelled":
                active_jobs[job_id].update({
                    "status": "completed",
                    "progress": 100,
                    "processed_frames": total_frames,
                    "results": results,
                    "found": len(results) > 0,
                    "total_detections": len(results)
                })

        print(f"Video analysis complete: job={job_id}, detections={len(results)}")

    except Exception as e:
        print(f"Video analysis error: {e}")
        with active_jobs_lock:
            active_jobs[job_id]["status"] = "error"
            active_jobs[job_id]["error"] = str(e)

    finally:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass



# -------------------- Flask Routes --------------------

def register_video_routes(app):
    """Register all video analysis routes"""

    @app.route("/upload_video", methods=["POST"])
    def upload_video():
        """Upload a video and start analysis"""
        # Get targets from form data
        targets_raw = request.form.get("targets", "")
        mode = request.form.get("mode", "fast")
        if not targets_raw:
            return jsonify({"success": False, "message": "No target objects specified"})

        # Parse targets: format "object1:color1,object2:color2,object3"
        targets = []
        for item in targets_raw.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                obj, color = item.split(":", 1)
                targets.append({"object": obj.strip().lower(), "color": color.strip().lower()})
            else:
                targets.append({"object": item.lower(), "color": ""})

        if not targets:
            return jsonify({"success": False, "message": "No valid targets parsed"})        # Switch backend based on UI toggle (fast/accurate)
        backend = None
        try:
            import object_detection as od
            if hasattr(od, "set_detection_backend"):
                od.set_detection_backend(mode)
            backend = getattr(od, "DETECTION_BACKEND", None)
        except Exception as e:
            print(f"Backend switch skipped: {e}")
            backend = None



        # Validate targets for standard YOLO
        if backend == "yolo_standard":
            coco_lower = [c.lower() for c in YOLO_COCO_CLASSES]
            for t in targets:
                if t["object"] not in coco_lower:
                    return jsonify({
                        "success": False,
                        "message": f"Object '{t['object']}' not available with current model."
                    })

        # Validate colors
        for t in targets:
            if t["color"] and t["color"] not in COLOR_HSV_RANGES:
                return jsonify({
                    "success": False,
                    "message": f"Color '{t['color']}' not supported."
                })

        # Get video file
        file = request.files.get("video")
        if not file:
            return jsonify({"success": False, "message": "No video file uploaded"})

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            })

        # Save video
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"success": False, "message": "Invalid filename"})

        job_id = generate_job_id()
        video_filename = f"{job_id}_{filename}"
        video_path = os.path.join(VIDEO_UPLOAD_FOLDER, video_filename)
        file.save(video_path)

        # Check file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > MAX_VIDEO_SIZE_MB:
            os.remove(video_path)
            return jsonify({
                "success": False,
                "message": f"Video too large ({file_size_mb:.0f}MB). Max: {MAX_VIDEO_SIZE_MB}MB"
            })

        # Create job
        with active_jobs_lock:
            active_jobs[job_id] = {
                "id": job_id,
                "status": "starting",
                "progress": 0,
                "total_frames": 0,
                "processed_frames": 0,
                "fps": 0,
                "duration": 0,
                "results": [],
                "found": False,
                "total_detections": 0,
                "error": None,
                "targets": targets,
                "filename": filename,
                "created_at": time.time()
            }

        # Start processing thread
        thread = threading.Thread(
            target=analyze_video_thread,
            args=(job_id, video_path, targets),
            daemon=True
        )
        thread.start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Video analysis started",
            "targets": targets
        })

    @app.route("/video_job_status/<job_id>")
    def video_job_status(job_id):
        """Get status and results of a video analysis job"""
        with active_jobs_lock:
            job = active_jobs.get(job_id)

        if not job:
            return jsonify({"error": "Job not found"}), 404

        return jsonify({
            "id": job["id"],
            "status": job["status"],
            "progress": job["progress"],
            "total_frames": job["total_frames"],
            "processed_frames": job["processed_frames"],
            "fps": job.get("fps", 0),
            "duration": job.get("duration", 0),
            "duration_formatted": format_timestamp(job.get("duration", 0)),
            "found": job.get("found", False),
            "total_detections": job.get("total_detections", 0),
            "results": job.get("results", []),
            "error": job.get("error"),
            "targets": job.get("targets", []),
            "filename": job.get("filename", "")
        })

    @app.route("/cancel_video_job/<job_id>", methods=["POST"])
    def cancel_video_job(job_id):
        """Cancel a running video analysis job"""
        with active_jobs_lock:
            job = active_jobs.get(job_id)
            if not job:
                return jsonify({"success": False, "message": "Job not found"})
            if job["status"] in ("completed", "error", "cancelled"):
                return jsonify({"success": False, "message": f"Job already {job['status']}"})
            job["status"] = "cancelled"

        return jsonify({"success": True, "message": "Job cancelled"})

    @app.route("/video_screenshot/<filename>")
    def video_screenshot(filename):
        """Serve a screenshot image from video analysis results"""
        from werkzeug.utils import secure_filename
        safe_name = secure_filename(filename)
        if not safe_name:
            return jsonify({"error": "Invalid filename"}), 400

        filepath = os.path.join(VIDEO_RESULTS_FOLDER, safe_name)
        if not os.path.exists(filepath):
            return jsonify({"error": "Screenshot not found"}), 404

        return send_from_directory(
            VIDEO_RESULTS_FOLDER,
            safe_name,
            mimetype='image/jpeg'
        )

    @app.route("/delete_video_job/<job_id>", methods=["POST"])
    def delete_video_job(job_id):
        """Delete a job and its screenshots"""
        with active_jobs_lock:
            job = active_jobs.pop(job_id, None)

        if not job:
            return jsonify({"success": False, "message": "Job not found"})

        # Clean up screenshots
        for result in job.get("results", []):
            screenshot = result.get("screenshot", "")
            if screenshot:
                path = os.path.join(VIDEO_RESULTS_FOLDER, screenshot)
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

        return jsonify({"success": True})

    @app.route("/list_video_jobs")
    def list_video_jobs():
        """List all video analysis jobs"""
        with active_jobs_lock:
            jobs = []
            for job_id, job in active_jobs.items():
                jobs.append({
                    "id": job["id"],
                    "status": job["status"],
                    "progress": job["progress"],
                    "filename": job.get("filename", ""),
                    "found": job.get("found", False),
                    "total_detections": job.get("total_detections", 0),
                    "targets": job.get("targets", []),
                    "created_at": job.get("created_at", 0)
                })

        # Sort by creation time, newest first
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        return jsonify({"jobs": jobs})