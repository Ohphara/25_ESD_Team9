import cv2
import numpy as np

FOCAL_LENGTH_PX = 529.56

CLASS_AVG_WIDTH_MM = {
    0: 600,
    1: 2500,
    2: 1800,
    3: 500,
    4: 800,
    5: 450,
    6: 600,
    7: 2500
}

TTC_THRESHOLD = 2.5
FRAME_SIZE = 480

def get_zone_pts(frame_size=FRAME_SIZE):
    top_y = int(frame_size * 0.55)
    bottom_y = int(frame_size * 0.95)
    center_x = frame_size // 2
    top_width = 100
    bottom_width = 220
    zone_pts = np.array([
        (center_x - top_width // 2, top_y),
        (center_x + top_width // 2, top_y),
        (center_x + bottom_width // 2, bottom_y),
        (center_x - bottom_width // 2, bottom_y),
    ], dtype=np.int32)
    return zone_pts

def compute_distance_m(cls, box_width_px):
    avg_width_mm = CLASS_AVG_WIDTH_MM.get(cls, 1000)
    distance_mm = (avg_width_mm * FOCAL_LENGTH_PX) / max(1, box_width_px)
    return distance_mm / 1000

def compute_ttc(prev_distance_m, distance_m, dt):
    dZ = prev_distance_m - distance_m
    v = dZ / dt if dt > 0 else 0
    ttc = distance_m / v if v > 0.01 else float('inf')
    return v, ttc

def draw_zone(frame, zone_pts):
    cv2.polylines(frame, [zone_pts], isClosed=True, color=(0, 255, 0), thickness=1)

def draw_detection(frame, x1, y1, x2, y2, label, warn):
    color = (0, 0, 255) if warn else (255, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
