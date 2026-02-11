"""
Tampering Detection Module (standalone)

Usage:
    python tampering_detection.py --source 0
    python tampering_detection.py --source path/to/video.mp4

Features implemented:
 - darkness / covered detection (mean brightness)
 - scene freeze detection (frame difference / low change for consecutive frames)
 - blur detection (variance of Laplacian)
 - camera movement detection (frame transform / translation via ECC)
 - snapshot + logging on tamper events
 - live overlay visualization, configurable thresholds

Notes:
 - This is lightweight and runs on CPU. Works with webcam or video files.
 - Tune thresholds at top of the script for your camera/environment.
 - Designed to be integrated later with your object-detection/face-recognition pipeline.


"""

import cv2
import numpy as np
import argparse
import time
import os
from collections import deque

# -------------------- Configuration / Thresholds --------------------
CFG = {
    # darkness: mean gray level below this -> dark/covered
    'darkness_threshold': 20.0,

    # freeze: count of consecutive frames with very small difference
    'freeze_frame_diff_thresh': 50,   # non-zero pixels threshold
    'freeze_frame_count': 50,         # number of consecutive frames (~2s @25fps)

    # blur: variance of Laplacian below this -> blurry
    'blur_threshold': 60.0,

    # camera movement: translation above this (pixels)
    'movement_shift_thresh': 8.0,

    # minimum seconds between repeated identical alerts of same type
    'alert_cooldown_seconds': 8,

    # output folders
    'snapshots_dir': 'tamper_snapshots',
    'log_file': 'tamper_log.txt',

    # visualization
    'display': True,
}

# -------------------- Utilities --------------------

def ensure_dirs():
    os.makedirs(CFG['snapshots_dir'], exist_ok=True)


def log_event(event_type, msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"{ts} | {event_type} | {msg}\n"
    with open(CFG['log_file'], 'a') as f:
        f.write(line)
    print(line.strip())


# -------------------- Detection Functions --------------------

def is_dark(frame_gray):
    mean_brightness = frame_gray.mean()
    return mean_brightness < CFG['darkness_threshold'], mean_brightness


def is_blurry(frame_gray):
    fm = cv2.Laplacian(frame_gray, cv2.CV_64F).var()
    return fm < CFG['blur_threshold'], fm


class FreezeDetector:
    def __init__(self, diff_thresh=CFG['freeze_frame_diff_thresh'], count_thresh=CFG['freeze_frame_count']):
        self.prev = None
        self.counter = 0
        self.diff_thresh = diff_thresh
        self.count_thresh = count_thresh

    def check(self, frame_gray):
        if self.prev is None:
            self.prev = frame_gray
            return False, 0

        diff = cv2.absdiff(frame_gray, self.prev)
        nonzero = np.count_nonzero(diff)
        if nonzero < self.diff_thresh:
            self.counter += 1
        else:
            self.counter = 0
        self.prev = frame_gray
        return (self.counter >= self.count_thresh), nonzero


class MovementDetector:
    def __init__(self, shift_thresh=CFG['movement_shift_thresh']):
        self.prev = None
        self.shift_thresh = shift_thresh
        # use small buffer of previous frames converted to gray & resized

    def check(self, frame_gray):
        if self.prev is None:
            self.prev = frame_gray
            return False, (0, 0)

        # use ECC-based alignment to estimate translation
        # warp_matrix 2x3
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp_matrix = cv2.findTransformECC(self.prev, frame_gray, warp_matrix,
                                                   cv2.MOTION_TRANSLATION,
                                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6))
        except cv2.error:
            # alignment may fail on low-texture images
            self.prev = frame_gray
            return False, (0, 0)

        shift_x = float(warp_matrix[0, 2])
        shift_y = float(warp_matrix[1, 2])
        self.prev = frame_gray
        moved = (abs(shift_x) > self.shift_thresh) or (abs(shift_y) > self.shift_thresh)
        return moved, (shift_x, shift_y)


# -------------------- Main loop --------------------

def run_detection(source=0, max_frames=None):
    ensure_dirs()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video source: {source}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    freeze_detector = FreezeDetector()
    movement_detector = MovementDetector()

    last_alert_time = {}

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('End of stream or cannot read frame')
            break

        frame_idx += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        alerts = []

        # darkness
        dark, brightness = is_dark(frame_gray)
        if dark:
            alerts.append(('DARKNESS', f'mean_brightness={brightness:.2f}'))

        # blur
        blurry, lap_var = is_blurry(frame_gray)
        if blurry:
            alerts.append(('BLUR', f'lap_var={lap_var:.2f}'))

        # freeze
        frozen, nonzero = freeze_detector.check(frame_gray)
        if frozen:
            alerts.append(('FREEZE', f'diff_nonzero={nonzero}'))

        # movement
        moved, shifts = movement_detector.check(frame_gray)
        if moved:
            alerts.append(('MOVE', f'shifts=({shifts[0]:.2f},{shifts[1]:.2f})'))

        # if any alerts, check cooldown and log + save snapshot
        now = time.time()
        for (atype, amsg) in alerts:
            last = last_alert_time.get(atype, 0)
            if now - last >= CFG['alert_cooldown_seconds']:
                last_alert_time[atype] = now
                # save snapshot
                fname = os.path.join(CFG['snapshots_dir'], f"{atype}_{int(now)}.jpg")
                cv2.imwrite(fname, frame)
                log_event(atype, f'{amsg} -> snapshot:{fname}')

        # Visualization overlay
        if CFG['display']:
            vis = frame.copy()
            y = 20
            cv2.putText(vis, f'Frame: {frame_idx}  FPS: {fps:.1f}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            y += 24
            cv2.putText(vis, f'Brightness: {brightness:.1f}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            y += 20
            cv2.putText(vis, f'LapVar: {lap_var:.1f}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            y += 20
            cv2.putText(vis, f'FreezeCounter: {freeze_detector.counter}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            y += 20
            cv2.putText(vis, f'MoveShift: {shifts[0]:.1f},{shifts[1]:.1f}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            # draw alert badges
            ax = 10
            ay = height - 30
            if alerts:
                cv2.rectangle(vis, (ax-6, ay-22), (ax+220, ay+6), (0,0,255), -1)
                cv2.putText(vis, 'TAMPER: ' + ', '.join([a[0] for a in alerts]), (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow('Tamper Detection', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                # manual snapshot
                fname = os.path.join(CFG['snapshots_dir'], f"MANUAL_{int(time.time())}.jpg")
                cv2.imwrite(fname, frame)
                log_event('MANUAL', f'snapshot:{fname}')

        # optional stop for testing
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    if CFG['display']:
        cv2.destroyAllWindows()


# -------------------- CLI --------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tampering Detection Module (standalone)')
    parser.add_argument('--source', type=str, default='0', help='video source (0 for webcam or path to file)')
    parser.add_argument('--max-frames', type=int, default=0, help='stop after N frames (for tests)')
    args = parser.parse_args()

    src = args.source
    if src.isdigit():
        src = int(src)

    if args.max_frames and args.max_frames <= 0:
        args.max_frames = None

    try:
        run_detection(source=src, max_frames=args.max_frames)
    except Exception as e:
        print('Error:', e)
        raise
