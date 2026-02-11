# app.py
# Main application - camera control, streaming, and orchestration

import cv2
import threading
import time
from flask import Flask, render_template, request, jsonify, Response

from tampering_detection import (
    is_dark,
    is_blurry,
    FreezeDetector,
    MovementDetector,
    log_event,
    CFG
)

# Import modules
from face_detection import (
    register_face_routes,
    process_faces,
    draw_faces,
)

from object_detection import (
    init_object_model,
    register_object_routes,
    process_objects,
    draw_objects,
    get_detect_interval,
    get_targets_count,
    clear_all_targets,
    DETECTION_BACKEND,
)

from video_analysis import register_video_routes

# -------------------- Flask App --------------------
app = Flask(__name__)

# -------------------- Register Module Routes --------------------
register_face_routes(app)

# Initialize object detection model before registering routes
backend = init_object_model()
register_object_routes(app)
register_video_routes(app)

# -------------------- Tampering Config --------------------
CFG['freeze_frame_count'] = 10
CFG['freeze_frame_diff_thresh'] = 100
CFG['darkness_threshold'] = 40.0
CFG['alert_cooldown_seconds'] = 3

# -------------------- Shared State --------------------
camera = None
camera_lock = threading.Lock()

detection_active = False
stream_active = False
is_ready = False
detection_mode = "face"

freeze_detector = None
movement_detector = None
last_alert_time = {}

# Frame sharing between threads
latest_frame = None
latest_frame_lock = threading.Lock()

# Detection results
detected_faces = []
detected_faces_lock = threading.Lock()
detected_objects = []
detected_objects_lock = threading.Lock()

# Alerts
active_alerts = []
active_alerts_lock = threading.Lock()


# -------------------- Public accessor for other modules --------------------

def get_latest_frame():
    """Thread-safe accessor for latest frame (used by object_detection.analyze_color)"""
    with latest_frame_lock:
        return latest_frame.copy() if latest_frame is not None else None


# -------------------- Core Routes --------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_alerts")
def get_alerts():
    with active_alerts_lock:
        return jsonify({"alerts": list(active_alerts)})


@app.route("/check_ready")
def check_ready():
    return jsonify({"ready": is_ready})


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/start")
def start_detection():
    global detection_active, camera, stream_active, is_ready
    global freeze_detector, movement_detector, last_alert_time
    global detected_faces, detected_objects, latest_frame, active_alerts
    global detection_mode

    _stop_existing()

    mode = request.args.get('mode', 'face')
    if mode not in ('face', 'object'):
        return jsonify({"status": "error", "message": "Invalid mode"})

    detection_mode = mode
    is_ready = False
    last_alert_time = {}

    with detected_faces_lock:
        detected_faces = []
    with detected_objects_lock:
        detected_objects = []
    with active_alerts_lock:
        active_alerts = []

    latest_frame = None

    freeze_detector = FreezeDetector(
        diff_thresh=CFG['freeze_frame_diff_thresh'],
        count_thresh=CFG['freeze_frame_count']
    )
    movement_detector = MovementDetector(shift_thresh=CFG['movement_shift_thresh'])

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = None
            return jsonify({"status": "error", "message": "Could not open camera"})

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    stream_active = True
    detection_active = True

    proc_thread = threading.Thread(target=_processing_thread, daemon=True)
    proc_thread.start()

    from object_detection import DETECTION_BACKEND as current_backend
    print(f"Detection started: mode='{detection_mode}', backend='{current_backend}'")

    return jsonify({
        "status": "started",
        "mode": detection_mode,
        "backend": current_backend
    })


@app.route("/stop")
def stop_detection():
    _stop_existing()
    print("Detection stopped")
    return jsonify({"status": "stopped"})


# -------------------- Internal Functions --------------------

def _stop_existing():
    global detection_active, stream_active, is_ready, camera
    global freeze_detector, movement_detector, latest_frame

    detection_active = False
    stream_active = False
    is_ready = False

    time.sleep(0.3)

    latest_frame = None
    freeze_detector = None
    movement_detector = None

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

    with detected_faces_lock:
        detected_faces.clear()
    with detected_objects_lock:
        detected_objects.clear()
    with active_alerts_lock:
        active_alerts.clear()


def _processing_thread():
    global detected_faces, detected_objects, active_alerts
    global freeze_detector, movement_detector

    frame_count = 0

    while stream_active:
        with latest_frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1

        if not detection_active:
            time.sleep(0.01)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % 2 == 0:
            alerts = _check_tampering(gray)
            with active_alerts_lock:
                active_alerts = alerts

        current_mode = detection_mode

        if current_mode == "face":
            if frame_count % 5 == 0:
                faces_data = process_faces(frame, gray)
                with detected_faces_lock:
                    detected_faces = faces_data

        elif current_mode == "object":
            interval = get_detect_interval()
            if frame_count % interval == 0:
                objects_data = process_objects(frame, gray)
                with detected_objects_lock:
                    detected_objects = objects_data

        time.sleep(0.005)


def _check_tampering(gray):
    alerts = []
    try:
        dark, brightness = is_dark(gray)

        if dark:
            alerts.append("COVERED")
            if movement_detector:
                movement_detector.prev = None
            if freeze_detector:
                freeze_detector.prev_frame = None
                freeze_detector.freeze_count = 0
        else:
            blurry, _ = is_blurry(gray)
            if blurry:
                alerts.append("BLUR")

            if freeze_detector:
                frozen, _ = freeze_detector.check(gray)
                if frozen:
                    alerts.append("FREEZE")

            if movement_detector:
                moved, _ = movement_detector.check(gray)
                if moved:
                    alerts.append("MOVED")

        now = time.time()
        for alert in alerts:
            last = last_alert_time.get(alert, 0)
            if now - last >= CFG["alert_cooldown_seconds"]:
                last_alert_time[alert] = now
                log_event(alert, "Detected during live monitoring")

    except Exception as e:
        print(f"Tampering detection error: {e}")

    return alerts


def generate_frames():
    global camera, latest_frame, is_ready

    while stream_active:
        with camera_lock:
            if camera is None:
                time.sleep(0.01)
                continue
            ret, frame = camera.read()

        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)

        with latest_frame_lock:
            latest_frame = frame.copy()

        if not is_ready:
            is_ready = True

        display_frame = frame.copy()
        current_mode = detection_mode

        if current_mode == "face":
            with detected_faces_lock:
                faces_copy = list(detected_faces)
            draw_faces(display_frame, faces_copy)

        elif current_mode == "object":
            with detected_objects_lock:
                objects_copy = list(detected_objects)
            draw_objects(display_frame, objects_copy)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        ret, buffer = cv2.imencode('.jpg', display_frame, encode_param)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# -------------------- Main --------------------
if __name__ == "__main__":
    from object_detection import DETECTION_BACKEND as _backend, ALL_OBJECTS, MAX_TARGETS, COLOR_HSV_RANGES

    print(f"\n{'=' * 60}")
    print(f"  AI Recognition System Ready")
    print(f"  Detection Backend : {_backend}")
    print(f"  Open Vocabulary   : {_backend in ('yolo_world', 'grounding_dino', 'owlv2')}")
    print(f"  Max Targets       : {MAX_TARGETS}")
    print(f"  Suggested Objects : {len(ALL_OBJECTS)}")
    print(f"  Color Detection   : HSV ({len(COLOR_HSV_RANGES)} colors)")
    print(f"  Video Analysis    : Enabled")
    print(f"{'=' * 60}\n")

    app.run(
        debug=False,
        threaded=True,
        host='127.0.0.1',
        port=5000,
        use_reloader=False
    )