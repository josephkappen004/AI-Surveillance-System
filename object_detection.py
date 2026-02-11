# object_detection.py
# Handles object detection, color analysis, multi-target management

import cv2
import numpy as np
import threading
import os
from flask import request, jsonify

from tampering_detection import is_dark, CFG

# ========================================================================
#  MODEL LOADING
# ========================================================================

DETECTION_BACKEND = None
obj_processor = None
obj_model = None
obj_device = None


def load_yolo_world():
    global DETECTION_BACKEND
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8x-worldv2.pt")
        DETECTION_BACKEND = "yolo_world"
        print("YOLO-World loaded")
        return None, model, None
    except Exception as e:
        print(f"Failed to load YOLO-World: {e}")
        return None, None, None


def load_grounding_dino():
    global DETECTION_BACKEND
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
        model_id = "IDEA-Research/grounding-dino-tiny"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        DETECTION_BACKEND = "grounding_dino"
        print(f"Grounding DINO loaded on {device}")
        return processor, model, device
    except Exception as e:
        print(f"Failed to load Grounding DINO: {e}")
        return None, None, None


def load_owlv2():
    global DETECTION_BACKEND
    try:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
        DETECTION_BACKEND = "owlv2"
        print(f"OWLv2 loaded on {device}")
        return processor, model, device
    except Exception as e:
        print(f"Failed to load OWLv2: {e}")
        return None, None, None


def load_yolo_fallback():
    global DETECTION_BACKEND
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        DETECTION_BACKEND = "yolo_standard"
        print("Standard YOLOv8 loaded (fallback)")
        return None, model, None
    except Exception as e:
        print(f"Failed to load any model: {e}")
        return None, None, None


def init_object_model():
    """Initialize object detection model with fallback chain

    You can force a backend via env var:
      FORCE_DETECTION_BACKEND=yolo_world | yolo_standard | grounding_dino | owlv2
    """
    global obj_processor, obj_model, obj_device

    print("Loading object detection model...")

    force = os.environ.get("FORCE_DETECTION_BACKEND", "").strip().lower()

    def _finish():
        if obj_model is None:
            print("WARNING: No object detection model could be loaded!")
        print(f"Detection backend: {DETECTION_BACKEND}")
        return DETECTION_BACKEND

    # ---------------- Forced backend (if requested) ----------------
    if force in ("yolo", "yolo_world"):
        obj_processor, obj_model, obj_device = load_yolo_world()
        if obj_model is None:
            obj_processor, obj_model, obj_device = load_yolo_fallback()
        return _finish()

    if force in ("yolo_standard", "yolov8", "yolov8n"):
        obj_processor, obj_model, obj_device = load_yolo_fallback()
        return _finish()

    if force in ("grounding_dino", "dino"):
        obj_processor, obj_model, obj_device = load_grounding_dino()
        return _finish()

    if force in ("owlv2", "owl"):
        obj_processor, obj_model, obj_device = load_owlv2()
        return _finish()

    # ---------------- Default (FAST-first) chain ----------------
    # YOLO (GPU-friendly) first, then heavier transformer backends.
    obj_processor, obj_model, obj_device = load_yolo_world()
    if obj_model is None:
        obj_processor, obj_model, obj_device = load_yolo_fallback()
    if obj_model is None:
        obj_processor, obj_model, obj_device = load_grounding_dino()
    if obj_model is None:
        obj_processor, obj_model, obj_device = load_owlv2()

    return _finish()

# ========================================================================
#  BACKEND SWITCHING (Fast / Accurate toggle)
# ========================================================================

def set_detection_backend(mode: str):
    """Switch detection backend at runtime.

    mode:
      - 'fast' -> yolo_standard (yolov8n.pt)
      - 'accurate' -> yolo_world (yolov8x-worldv2.pt)
    """
    global obj_processor, obj_model, obj_device, DETECTION_BACKEND

    mode = (mode or "").strip().lower()

    if mode in ("fast", "yolo_standard", "yolov8n", "standard"):
        print("🔁 Switching detection backend → YOLO STANDARD (FAST)")
        obj_processor, obj_model, obj_device = load_yolo_fallback()

    elif mode in ("accurate", "yolo_world", "world"):
        print("🔁 Switching detection backend → YOLO WORLD (ACCURATE)")
        obj_processor, obj_model, obj_device = load_yolo_world()

    else:
        print(f"⚠️ Unknown detection mode '{mode}', keeping current backend: {DETECTION_BACKEND}")
        return DETECTION_BACKEND

    print(f"Detection backend now: {DETECTION_BACKEND}")
    return DETECTION_BACKEND

# ========================================================================
#  HSV COLOR DETECTION
# ========================================================================

COLOR_HSV_RANGES = {
    'red': {
        'ranges': [
            {'h_min': 0, 'h_max': 10, 's_min': 70, 's_max': 255, 'v_min': 50, 'v_max': 255},
            {'h_min': 160, 'h_max': 179, 's_min': 70, 's_max': 255, 'v_min': 50, 'v_max': 255},
        ]
    },
    'blue': {
        'ranges': [
            {'h_min': 100, 'h_max': 130, 's_min': 50, 's_max': 255, 'v_min': 40, 'v_max': 255},
        ]
    },
    'green': {
        'ranges': [
            {'h_min': 35, 'h_max': 85, 's_min': 50, 's_max': 255, 'v_min': 40, 'v_max': 255},
        ]
    },
    'yellow': {
        'ranges': [
            {'h_min': 20, 'h_max': 35, 's_min': 70, 's_max': 255, 'v_min': 100, 'v_max': 255},
        ]
    },
    'orange': {
        'ranges': [
            {'h_min': 10, 'h_max': 22, 's_min': 100, 's_max': 255, 'v_min': 100, 'v_max': 255},
        ]
    },
    'pink': {
        'ranges': [
            {'h_min': 140, 'h_max': 170, 's_min': 30, 's_max': 180, 'v_min': 150, 'v_max': 255},
            {'h_min': 0, 'h_max': 10, 's_min': 30, 's_max': 150, 'v_min': 150, 'v_max': 255},
        ]
    },
    'purple': {
        'ranges': [
            {'h_min': 125, 'h_max': 155, 's_min': 50, 's_max': 255, 'v_min': 40, 'v_max': 255},
        ]
    },
    'brown': {
        'ranges': [
            {'h_min': 8, 'h_max': 25, 's_min': 60, 's_max': 255, 'v_min': 30, 'v_max': 150},
        ]
    },
    'black': {
        'ranges': [
            {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 255, 'v_min': 0, 'v_max': 50},
        ]
    },
    'white': {
        'ranges': [
            {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 40, 'v_min': 200, 'v_max': 255},
        ]
    },
    'gray': {
        'ranges': [
            {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 40, 'v_min': 50, 'v_max': 200},
        ]
    },
}

COLOR_MATCH_THRESHOLD = 0.15
COLOR_DOMINANT_THRESHOLD = 0.25

DRAW_COLORS = [
    (0, 255, 0),     # green
    (255, 100, 0),   # blue-ish
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (0, 165, 255),   # orange
    (255, 255, 0),   # cyan
    (147, 20, 255),  # pink
    (0, 128, 255),   # dark orange
    (255, 0, 0),     # blue
    (50, 205, 50),   # lime
    (238, 130, 238), # violet
    (0, 215, 255),   # gold
]


def extract_object_roi(image, bbox, padding_ratio=0.1):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    pad_x, pad_y = int(bw * padding_ratio), int(bh * padding_ratio)
    ix1, iy1 = x1 + pad_x, y1 + pad_y
    ix2, iy2 = x2 - pad_x, y2 - pad_y
    if ix2 <= ix1 or iy2 <= iy1:
        ix1, iy1, ix2, iy2 = x1, y1, x2, y2
    roi = image[iy1:iy2, ix1:ix2]
    return roi if roi.size > 0 else None


def get_color_percentages(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return {}
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    total = roi_hsv.shape[0] * roi_hsv.shape[1]
    if total == 0:
        return {}
    percentages = {}
    for color_name, info in COLOR_HSV_RANGES.items():
        mask = np.zeros((roi_hsv.shape[0], roi_hsv.shape[1]), dtype=np.uint8)
        for r in info['ranges']:
            lower = np.array([r['h_min'], r['s_min'], r['v_min']])
            upper = np.array([r['h_max'], r['s_max'], r['v_max']])
            mask = cv2.bitwise_or(mask, cv2.inRange(roi_hsv, lower, upper))
        percentages[color_name] = cv2.countNonZero(mask) / total
    return percentages


def get_dominant_color_hsv(image, bbox):
    roi = extract_object_roi(image, bbox, padding_ratio=0.15)
    if roi is None:
        return "unknown", 0.0
    pcts = get_color_percentages(roi)
    if not pcts:
        return "unknown", 0.0
    dominant = max(pcts, key=pcts.get)
    return dominant, pcts[dominant]


def matches_color(image, bbox, target_color):
    if not target_color:
        return True, 1.0, "any"
    target_color = target_color.lower().strip()
    if target_color not in COLOR_HSV_RANGES:
        return False, 0.0, "unknown"
    roi = extract_object_roi(image, bbox, padding_ratio=0.15)
    if roi is None:
        return False, 0.0, "no_roi"
    pcts = get_color_percentages(roi)
    if not pcts:
        return False, 0.0, "no_data"
    target_pct = pcts.get(target_color, 0.0)
    dominant = max(pcts, key=pcts.get)
    is_match = False
    reason = ""
    if target_pct >= COLOR_DOMINANT_THRESHOLD:
        is_match = True
        reason = f"dominant ({target_pct:.0%})"
    elif target_pct >= COLOR_MATCH_THRESHOLD and dominant == target_color:
        is_match = True
        reason = f"most common ({target_pct:.0%})"
    elif target_pct >= COLOR_MATCH_THRESHOLD and dominant in ('black', 'white', 'gray'):
        is_match = True
        reason = f"present ({target_pct:.0%})"
    elif target_color in ('black', 'white', 'gray') and target_pct >= 0.10:
        is_match = True
        reason = f"achromatic ({target_pct:.0%})"
    return is_match, target_pct, reason


def get_color_debug_info(image, bbox):
    roi = extract_object_roi(image, bbox, padding_ratio=0.15)
    if roi is None:
        return []
    pcts = get_color_percentages(roi)
    return [(n, p) for n, p in sorted(pcts.items(), key=lambda x: x[1], reverse=True) if p > 0.01]


# ========================================================================
#  OBJECT CATEGORIES
# ========================================================================

OBJECT_CATEGORIES = {
    "Weapons & Dangerous Items": [
        "knife", "gun", "pistol", "rifle", "sword", "machete", "axe",
        "hammer", "baseball bat", "explosive", "bomb", "grenade",
        "brass knuckles", "taser", "pepper spray", "crossbow",
        "dagger", "switchblade", "crowbar", "chain"
    ],
    "Electronics & Gadgets": [
        "laptop", "cell phone", "smartphone", "tablet", "ipad",
        "smartwatch", "watch", "camera", "webcam", "drone",
        "headphones", "earbuds", "airpods", "speaker", "bluetooth speaker",
        "power bank", "charger", "usb drive", "flash drive",
        "keyboard", "mouse", "monitor", "tv", "television",
        "remote control", "game controller", "joystick",
        "router", "modem", "hard drive", "ssd",
        "microphone", "projector", "printer", "scanner",
        "gopro", "action camera", "security camera", "ring doorbell",
        "walkie talkie", "radio", "gps device", "calculator"
    ],
    "Vehicles": [
        "car", "truck", "bus", "motorcycle", "bicycle", "scooter",
        "van", "suv", "sedan", "pickup truck", "ambulance",
        "fire truck", "police car", "taxi", "helicopter",
        "airplane", "jet", "boat", "ship", "yacht", "kayak",
        "skateboard", "segway", "golf cart", "forklift",
        "tractor", "excavator", "crane", "bulldozer",
        "train", "subway", "tram", "rickshaw"
    ],
    "People & Body Parts": [
        "person", "man", "woman", "child", "baby", "face",
        "hand", "fist", "finger", "foot", "head",
        "crowd", "group of people", "pedestrian"
    ],
    "Animals": [
        "dog", "cat", "bird", "horse", "cow", "sheep",
        "elephant", "bear", "zebra", "giraffe", "lion",
        "tiger", "monkey", "snake", "fish", "shark",
        "whale", "dolphin", "eagle", "hawk", "owl",
        "rabbit", "deer", "fox", "wolf", "rat", "mouse",
        "spider", "insect", "butterfly", "bee", "ant",
        "chicken", "duck", "goose", "pig", "goat",
        "parrot", "penguin", "turtle", "frog", "lizard",
        "crocodile", "alligator"
    ],
    "Clothing & Accessories": [
        "hat", "cap", "helmet", "mask", "ski mask", "balaclava",
        "sunglasses", "glasses", "eyeglasses", "goggles",
        "backpack", "bag", "handbag", "purse", "briefcase",
        "suitcase", "luggage", "duffel bag", "fanny pack",
        "shoe", "boot", "sneaker", "sandal", "high heel",
        "tie", "necktie", "bowtie", "scarf", "glove",
        "jacket", "coat", "hoodie", "vest", "dress",
        "shirt", "pants", "jeans", "shorts", "skirt",
        "belt", "wallet", "umbrella", "ring", "necklace",
        "bracelet", "earring", "badge", "id card", "lanyard"
    ],
    "Furniture & Home": [
        "chair", "table", "desk", "couch", "sofa", "bed",
        "shelf", "bookshelf", "cabinet", "drawer", "wardrobe",
        "lamp", "chandelier", "ceiling fan", "mirror",
        "curtain", "carpet", "rug", "pillow", "blanket",
        "clock", "painting", "picture frame", "poster",
        "vase", "flower pot", "potted plant", "candle",
        "door", "window", "staircase", "fence", "gate",
        "toilet", "sink", "bathtub", "shower", "faucet"
    ],
    "Kitchen & Food": [
        "bottle", "water bottle", "wine bottle", "beer bottle",
        "wine glass", "cup", "mug", "glass", "bowl", "plate",
        "fork", "knife", "spoon", "chopsticks",
        "pan", "pot", "wok", "kettle", "toaster", "blender",
        "microwave", "oven", "stove", "refrigerator", "dishwasher",
        "cutting board", "rolling pin", "whisk", "spatula",
        "banana", "apple", "orange", "pizza", "burger",
        "sandwich", "hot dog", "donut", "cake", "bread",
        "rice", "pasta", "salad", "soup", "sushi",
        "ice cream", "chocolate", "candy", "cookie",
        "coffee", "tea", "juice", "soda", "beer", "wine",
        "grocery bag", "lunch box", "thermos"
    ],
    "Tools & Equipment": [
        "screwdriver", "wrench", "pliers", "saw", "drill",
        "tape measure", "level", "ladder", "toolbox",
        "fire extinguisher", "flashlight", "torch", "lantern",
        "rope", "chain", "wire", "cable", "tape",
        "lock", "padlock", "key", "bolt", "nail", "screw",
        "shovel", "rake", "hoe", "wheelbarrow",
        "broom", "mop", "bucket", "hose", "sprinkler",
        "generator", "compressor", "welder"
    ],
    "Sports & Recreation": [
        "ball", "soccer ball", "football", "basketball", "tennis ball",
        "baseball", "volleyball", "golf ball", "cricket bat",
        "tennis racket", "badminton racket", "ping pong paddle",
        "hockey stick", "lacrosse stick", "pool cue",
        "skateboard", "surfboard", "snowboard", "ski",
        "bicycle helmet", "boxing glove", "punching bag",
        "dumbbell", "barbell", "weight plate", "kettlebell",
        "yoga mat", "jump rope", "resistance band",
        "swimming goggles", "snorkel", "life jacket",
        "fishing rod", "fishing net", "tent", "sleeping bag",
        "hiking pole", "compass", "binoculars"
    ],
    "Medical & Safety": [
        "first aid kit", "bandage", "syringe", "needle",
        "stethoscope", "thermometer", "blood pressure monitor",
        "wheelchair", "crutch", "walker", "stretcher",
        "medicine bottle", "pill bottle", "inhaler",
        "face mask", "surgical mask", "n95 mask",
        "safety vest", "hard hat", "safety goggles",
        "fire alarm", "smoke detector", "exit sign",
        "defibrillator", "oxygen tank", "iv bag"
    ],
    "Office & School": [
        "pen", "pencil", "marker", "eraser", "ruler",
        "scissors", "stapler", "paper clip", "rubber band",
        "notebook", "book", "textbook", "binder", "folder",
        "envelope", "stamp", "sticker", "tape dispenser",
        "whiteboard", "blackboard", "chalk", "marker board",
        "globe", "map", "calendar", "planner",
        "paper", "document", "newspaper", "magazine",
        "printer paper", "sticky note", "index card"
    ],
    "Containers & Packaging": [
        "box", "cardboard box", "package", "parcel",
        "crate", "barrel", "drum", "tank", "container",
        "jar", "can", "tin", "tube", "pouch",
        "plastic bag", "paper bag", "trash bag", "bin",
        "trash can", "recycling bin", "dumpster",
        "basket", "hamper", "cooler", "ice chest"
    ],
    "Signs & Symbols": [
        "stop sign", "traffic light", "speed limit sign",
        "exit sign", "no entry sign", "warning sign",
        "fire exit sign", "handicap sign", "parking sign",
        "license plate", "number plate", "barcode", "qr code",
        "flag", "banner", "billboard", "advertisement"
    ],
    "Infrastructure": [
        "traffic light", "street light", "lamp post",
        "fire hydrant", "mailbox", "phone booth",
        "bench", "parking meter", "bollard",
        "manhole cover", "drain", "gutter",
        "power line", "telephone pole", "antenna",
        "satellite dish", "solar panel", "wind turbine",
        "bridge", "tunnel", "overpass"
    ]
}

ALL_OBJECTS = sorted(list(set(
    item for items in OBJECT_CATEGORIES.values() for item in items
)))

YOLO_COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

MAX_TARGETS = 10


# ========================================================================
#  MULTI-TARGET MANAGEMENT
# ========================================================================

target_objects = []
target_objects_lock = threading.Lock()
_next_target_id = 1


def get_targets_snapshot():
    """Get a thread-safe copy of current targets"""
    with target_objects_lock:
        return [t.copy() for t in target_objects]


def get_targets_count():
    """Get current target count"""
    with target_objects_lock:
        return len(target_objects)


def clear_all_targets():
    """Clear all targets"""
    with target_objects_lock:
        target_objects.clear()


# ========================================================================
#  DETECTION BACKENDS (MULTI-CLASS)
# ========================================================================

def detect_yolo_world_multi(frame, class_names, confidence_threshold=0.3):
    """YOLO-World: open-vocabulary detection restricted to provided class_names."""
    try:
        if obj_model is None:
            return []
        obj_model.set_classes(class_names)

        yolo_kwargs = {"verbose": False, "conf": float(confidence_threshold)}
        # Force GPU when available (Ultralytics accepts device=0 for first CUDA device)
        try:
            import torch
            if torch.cuda.is_available():
                yolo_kwargs["device"] = 0
        except Exception:
            pass

        results = obj_model(frame, **yolo_kwargs)

        detections = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detected_class = class_names[class_id] if class_id < len(class_names) else class_names[0]
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "label": detected_class,
                    "confidence": confidence
                })
        return detections
    except Exception as e:
        print(f"YOLO-World multi error: {e}")
        return []


def detect_grounding_dino_multi(frame, class_names, confidence_threshold=0.3):
    import torch
    try:
        from PIL import Image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        text_prompt = ". ".join(class_names) + "."
        inputs = obj_processor(images=pil_image, text=text_prompt, return_tensors="pt").to(obj_device)
        with torch.no_grad():
            outputs = obj_model(**inputs)
        results = obj_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            target_sizes=[pil_image.size[::-1]]
        )[0]
        detections = []
        h, w = frame.shape[:2]
        for score, box, label in zip(results["scores"], results["boxes"], results["labels"]):
            confidence = float(score)
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
            x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
            detected_label = label.strip().lower()
            matched_class = detected_label
            for cn in class_names:
                if cn.lower() in detected_label or detected_label in cn.lower():
                    matched_class = cn
                    break
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'label': matched_class,
                'confidence': confidence
            })
        return detections
    except Exception as e:
        print(f"Grounding DINO multi error: {e}")
        return []


def detect_owlv2_multi(frame, class_names, confidence_threshold=0.2):
    import torch
    try:
        from PIL import Image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        texts = [[f"a photo of a {cn}" for cn in class_names]]
        inputs = obj_processor(text=texts, images=pil_image, return_tensors="pt").to(obj_device)
        with torch.no_grad():
            outputs = obj_model(**inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(obj_device)
        results = obj_processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]
        detections = []
        h, w = frame.shape[:2]
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            confidence = float(score)
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
            x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
            class_idx = int(label_id)
            detected_class = class_names[class_idx] if class_idx < len(class_names) else class_names[0]
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'label': detected_class,
                'confidence': confidence
            })
        return detections
    except Exception as e:
        print(f"OWLv2 multi error: {e}")
        return []


def detect_yolo_standard_multi(frame, class_names, confidence_threshold=0.5):
    """Standard YOLOv8 COCO model: filter detections to the requested class_names."""
    try:
        if obj_model is None:
            return []
        target_lower = [cn.lower() for cn in class_names]

        yolo_kwargs = {"verbose": False, "conf": float(confidence_threshold)}
        try:
            import torch
            if torch.cuda.is_available():
                yolo_kwargs["device"] = 0
        except Exception:
            pass

        results = obj_model(frame, **yolo_kwargs)

        detections = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id >= len(YOLO_COCO_CLASSES):
                    continue
                class_name = YOLO_COCO_CLASSES[class_id]
                confidence = float(box.conf[0])
                if class_name.lower() in target_lower:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "label": class_name,
                        "confidence": confidence
                    })
        return detections
    except Exception as e:
        print(f"YOLO standard multi error: {e}")
        return []


def detect_objects_multi(frame, class_names, confidence_threshold=0.3):
    """Route to appropriate multi-class detection backend"""
    if not class_names:
        return []
    if DETECTION_BACKEND == "yolo_world":
        return detect_yolo_world_multi(frame, class_names, confidence_threshold)
    elif DETECTION_BACKEND == "grounding_dino":
        return detect_grounding_dino_multi(frame, class_names, confidence_threshold)
    elif DETECTION_BACKEND == "owlv2":
        return detect_owlv2_multi(frame, class_names, confidence_threshold)
    elif DETECTION_BACKEND == "yolo_standard":
        return detect_yolo_standard_multi(frame, class_names, confidence_threshold)
    return []


# ========================================================================
#  FLASK ROUTES
# ========================================================================

def register_object_routes(app):
    """Register all object-detection routes on the Flask app"""

    @app.route("/get_available_objects")
    def get_available_objects():
        return jsonify({
            "objects": ALL_OBJECTS,
            "categories": OBJECT_CATEGORIES,
            "backend": DETECTION_BACKEND,
            "open_vocabulary": DETECTION_BACKEND in ("yolo_world", "grounding_dino", "owlv2"),
            "total_count": len(ALL_OBJECTS),
            "available_colors": list(COLOR_HSV_RANGES.keys()),
            "max_targets": MAX_TARGETS
        })


    @app.route("/get_object_catalog")
    def get_object_catalog():
        """Return object catalog for a requested detection mode (does not change the active backend).

        Query param:
            mode=fast      -> COCO catalog (YOLO standard / limited)
            mode=accurate  -> Full catalog (YOLO-World / open-vocabulary)
        """
        mode = (request.args.get("mode", "accurate") or "accurate").strip().lower()

        if mode == "fast":
            objs = YOLO_COCO_CLASSES
            cats = {"COCO Classes (Fast)": YOLO_COCO_CLASSES}
            backend = "yolo_standard"
            open_vocab = False
        else:
            objs = ALL_OBJECTS
            cats = OBJECT_CATEGORIES
            backend = "yolo_world"
            open_vocab = True

        return jsonify({
            "objects": objs,
            "categories": cats,
            "backend": backend,
            "open_vocabulary": open_vocab,
            "total_count": len(objs),
            "available_colors": list(COLOR_HSV_RANGES.keys()),
            "max_targets": MAX_TARGETS
        })


    @app.route("/search_objects")
    def search_objects():
        query = request.args.get("q", "").lower().strip()
        if not query:
            return jsonify({"results": ALL_OBJECTS[:50]})
        results = [obj for obj in ALL_OBJECTS if query in obj.lower()]
        if DETECTION_BACKEND in ("yolo_world", "grounding_dino", "owlv2"):
            if query not in results:
                results.insert(0, query)
        return jsonify({"results": results[:30]})

    @app.route("/add_target", methods=["POST"])
    def add_target():
        global _next_target_id

        data = request.json
        if not data:
            return jsonify({"success": False, "message": "No data provided"})

        obj = data.get("object", "").lower().strip()
        color = data.get("color", "").lower().strip()

        if not obj:
            return jsonify({"success": False, "message": "No object specified"})

        if color and color not in COLOR_HSV_RANGES:
            return jsonify({
                "success": False,
                "message": f"Color '{color}' not supported. Available: {', '.join(COLOR_HSV_RANGES.keys())}"
            })

        if DETECTION_BACKEND == "yolo_standard":
            if obj not in [c.lower() for c in YOLO_COCO_CLASSES]:
                return jsonify({
                    "success": False,
                    "message": f"Object '{obj}' not available with current model."
                })

        with target_objects_lock:
            if len(target_objects) >= MAX_TARGETS:
                return jsonify({
                    "success": False,
                    "message": f"Maximum {MAX_TARGETS} targets. Remove one first."
                })

            for t in target_objects:
                if t["object"] == obj and t["color"] == color:
                    return jsonify({
                        "success": False,
                        "message": f"'{obj}' with color '{color or 'any'}' already tracked."
                    })

            draw_color = DRAW_COLORS[len(target_objects) % len(DRAW_COLORS)]
            target = {
                "id": str(_next_target_id),
                "object": obj,
                "color": color,
                "draw_color": draw_color
            }
            _next_target_id += 1
            target_objects.append(target)

            print(f"Target added: {target}")

            return jsonify({
                "success": True,
                "target": {
                    "id": target["id"],
                    "object": target["object"],
                    "color": target["color"] if target["color"] else "any",
                    "draw_color": list(target["draw_color"])
                },
                "total_targets": len(target_objects)
            })

    @app.route("/remove_target", methods=["POST"])
    def remove_target():
        data = request.json
        if not data or "id" not in data:
            return jsonify({"success": False, "message": "Missing target ID"})

        target_id = str(data["id"])

        with target_objects_lock:
            before = len(target_objects)
            target_objects[:] = [t for t in target_objects if t["id"] != target_id]
            after = len(target_objects)

            for i, t in enumerate(target_objects):
                t["draw_color"] = DRAW_COLORS[i % len(DRAW_COLORS)]

        if after < before:
            print(f"Target removed: id={target_id}")
            return jsonify({"success": True, "total_targets": after})
        return jsonify({"success": False, "message": "Target not found"})

    @app.route("/clear_targets", methods=["POST"])
    def clear_targets_route():
        with target_objects_lock:
            target_objects.clear()
        print("All targets cleared")
        return jsonify({"success": True, "total_targets": 0})

    @app.route("/get_targets")
    def get_targets():
        with target_objects_lock:
            targets = [
                {
                    "id": t["id"],
                    "object": t["object"],
                    "color": t["color"] if t["color"] else "any",
                    "draw_color": list(t["draw_color"])
                }
                for t in target_objects
            ]
        return jsonify({"targets": targets, "total": len(targets)})

    @app.route("/set_target_object", methods=["POST"])
    def set_target_object_legacy():
        """Legacy endpoint: clears all and sets one"""
        data = request.json
        if not data:
            return jsonify({"success": False, "message": "No data provided"})

        obj = data.get("object", "").lower().strip()
        color = data.get("color", "").lower().strip()

        if not obj:
            return jsonify({"success": False, "message": "No object specified"})

        with target_objects_lock:
            target_objects.clear()

        # Reuse add_target logic via internal call
        return _add_target_internal(obj, color)

    @app.route("/analyze_color", methods=["POST"])
    def analyze_color():
        # Need access to latest_frame from app.py
        from app import get_latest_frame
        frame = get_latest_frame()
        if frame is None:
            return jsonify({"error": "No frame available"})
        data = request.json
        x1, y1 = data.get("x1", 0), data.get("y1", 0)
        x2, y2 = data.get("x2", frame.shape[1]), data.get("y2", frame.shape[0])
        debug_info = get_color_debug_info(frame, (x1, y1, x2, y2))
        dominant, confidence = get_dominant_color_hsv(frame, (x1, y1, x2, y2))
        return jsonify({
            "dominant_color": dominant,
            "dominant_confidence": round(confidence, 3),
            "all_colors": [{"color": n, "percentage": round(p, 3)} for n, p in debug_info]
        })


def _add_target_internal(obj, color):
    """Internal helper shared by add_target and set_target_object_legacy"""
    global _next_target_id

    if color and color not in COLOR_HSV_RANGES:
        return jsonify({"success": False, "message": f"Color '{color}' not supported."})

    if DETECTION_BACKEND == "yolo_standard":
        if obj not in [c.lower() for c in YOLO_COCO_CLASSES]:
            return jsonify({"success": False, "message": f"Object '{obj}' not available."})

    with target_objects_lock:
        draw_color = DRAW_COLORS[len(target_objects) % len(DRAW_COLORS)]
        target = {
            "id": str(_next_target_id),
            "object": obj,
            "color": color,
            "draw_color": draw_color
        }
        _next_target_id += 1
        target_objects.append(target)

    return jsonify({"success": True, "object": obj, "color": color if color else "any"})


# ========================================================================
#  OBJECT PROCESSING (called from app.py processing thread)
# ========================================================================

def process_objects(frame, gray):
    """
    Run multi-target object detection on a frame.
    Returns list of object data dicts with bbox, label, color, target_id.
    """
    try:
        current_targets = get_targets_snapshot()

        if not current_targets:
            return []

        _, brightness_check = is_dark(gray)
        if brightness_check <= CFG['darkness_threshold']:
            return []

        # Collect unique class names
        class_names = list(set(t["object"] for t in current_targets))

        # Build lookup: object_name -> list of targets
        target_lookup = {}
        for t in current_targets:
            if t["object"] not in target_lookup:
                target_lookup[t["object"]] = []
            target_lookup[t["object"]].append(t)

        # Single forward pass for ALL classes
        detections = detect_objects_multi(frame, class_names)
        objects_data = []

        for det in detections:
            bbox = det['bbox']
            det_label = det['label'].lower()
            det_confidence = det['confidence']

            matching_targets = target_lookup.get(det_label, [])

            if not matching_targets:
                for cn, targets in target_lookup.items():
                    if cn in det_label or det_label in cn:
                        matching_targets = targets
                        break

            if not matching_targets:
                continue

            for target in matching_targets:
                t_color = target["color"]
                draw_color = tuple(target["draw_color"])

                if t_color:
                    is_match, color_pct, reason = matches_color(frame, bbox, t_color)
                    if not is_match:
                        continue
                    color_label = f" [{t_color} {color_pct:.0%}]"
                else:
                    color_label = ""
                    detected_color, color_conf = get_dominant_color_hsv(frame, bbox)
                    if color_conf > 0.15:
                        color_label = f" [{detected_color}]"

                display_label = f"{det['label']}{color_label} ({det_confidence:.0%})"

                objects_data.append({
                    'bbox': bbox,
                    'label': display_label,
                    'color': draw_color,
                    'target_id': target["id"]
                })

        return objects_data

    except Exception as e:
        print(f"Object detection error: {e}")
        return []


def draw_objects(frame, objects_data):
    """Draw object detection results on display frame"""
    for obj_data in objects_data:
        x1, y1, x2, y2 = obj_data['bbox']
        label = obj_data['label']
        color = obj_data['color']

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame,
                      (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0] + 4, y1),
                      color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def get_detect_interval():
    """Return appropriate frame interval based on backend speed"""
    if DETECTION_BACKEND in ("grounding_dino", "owlv2"):
        return 5
    return 3

