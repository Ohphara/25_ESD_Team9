import cv2
import time
import numpy as np
from ultralytics import YOLO

SPEED_THRESHOLDS = {1: 5.0, 21: 5.0, 24: 2.0, 22: 4.0, 14: 3.0}
ALERT_CLASSES = SPEED_THRESHOLDS.keys()
FRAME_SIZE = 480

# 사다리꼴 상단/하단 기준 거리 (m)
TOP_DIST_M = 4.0
BOTTOM_DIST_M = 2.5

prev_info = {}
alert_cooldowns = {}

def should_alert(obj_id, risk_score, t_now):
    if risk_score >= 10:
        return True
    cooldown = 2 if risk_score >= 6 else 5
    last_alert = alert_cooldowns.get(obj_id, 0)
    if t_now - last_alert > cooldown:
        alert_cooldowns[obj_id] = t_now
        return True
    return False

def compute_risk_score(speed, distance_m, obj_class_name, approach_angle):
    if approach_angle < 0.5:  # 정면 접근
        dist_score = max(0, 5 - distance_m)
    else:  # 측면 접근
        dist_score = 2 if distance_m < 2.5 else 0
    speed_score = min(speed / 2, 5)
    obj_score = {'car': 3, 'bicycle': 2, 'bus': 3, 'truck': 3, 'motorcycle': 2}.get(obj_class_name, 1)
    return dist_score + speed_score + obj_score

def estimate_distance_from_y(y):
    return np.interp(y, [top_y, bottom_y], [4.0, 2.5])

model = YOLO("C:/Users/k9481/MyYoloProject/final_best3.pt")
cap = cv2.VideoCapture("C:/Users/k9481/Downloads/20250603_160952.mp4")
prev_time = time.time()

top_y = int(FRAME_SIZE * 0.55)
bottom_y = int(FRAME_SIZE * 0.95)
center_x = FRAME_SIZE // 2
top_width = 100
bottom_width = 220
zone_pts = np.array([
    (center_x - top_width // 2, top_y),
    (center_x + top_width // 2, top_y),
    (center_x + bottom_width // 2, bottom_y),
    (center_x - bottom_width // 2, bottom_y),
], dtype=np.int32)
zone_center = np.array([center_x, bottom_y])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    results = model.track(source=frame_resized, persist=True, stream=True, tracker="bytetrack.yaml")
    t_now = time.time()

    cv2.polylines(frame_resized, [zone_pts], isClosed=True, color=(0, 255, 0), thickness=1)

    for r in results:
        for box in r.boxes:
            if not hasattr(box, 'id') or box.id is None:
                continue

            obj_id = int(box.id.item())
            cls = int(box.cls.item())
            if cls not in model.names or cls not in ALERT_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bottom_center = (cx, y2)
            distance_m = estimate_distance_from_y(bottom_center[1])

            if obj_id in prev_info:
                px, py, t_prev, prev_speed = prev_info[obj_id]
                dt = t_now - t_prev
                dx, dy = cx - px, cy - py
                dist = np.hypot(dx, dy)
                speed = dist / dt if dt > 0.01 else 0
                if abs(speed - prev_speed) > 30:
                    speed = prev_speed
                approach_angle = abs(dx) / (abs(dy) + 1e-6)
            else:
                speed, approach_angle = 0, 0

            in_zone = cv2.pointPolygonTest(zone_pts, bottom_center, False) >= 0
            speed_thresh = SPEED_THRESHOLDS.get(cls, 1.5)

            warn = False
            if in_zone and speed > speed_thresh:
                risk_score = compute_risk_score(speed, distance_m, model.names[cls], approach_angle)
                if should_alert(obj_id, risk_score, t_now):
                    warn = True

            prev_info[obj_id] = (cx, cy, t_now, speed)

            label = f"{model.names[cls]} {speed:.1f}px/s {distance_m:.2f}m"
            color = (0, 0, 255) if warn else (255, 255, 0)
            if warn:
                label += " ⚠"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.line(frame_resized, bottom_center, zone_center, (0, 255, 255), 1)

    fps = 1.0 / (time.time() - prev_time + 1e-6)
    prev_time = time.time()
    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Alert System", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
