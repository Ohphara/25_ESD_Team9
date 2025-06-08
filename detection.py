# detection.py
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Danger speed thresholds by class index
SPEED_THRESHOLDS = {
    0: 2.0,  # bicycle
    1: 4.0,  # bus
    2: 5.0,  # car
    3: 1.5,  # carrier
    4: 3.0,  # motorcycle
    6: 2.5,  # scooter
    7: 5.0   # truck
}
ALERT_CLASSES = SPEED_THRESHOLDS.keys()

class DangerDetector:
    def __init__(self, model_path: str, tracker_cfg: str, frame_size: int = 480):
        self.model = YOLO(model_path)
        self.tracker_cfg = tracker_cfg
        self.frame_size = frame_size
        self.zone_pts = self._define_zone()
        self.prev_gray = None
        self.affine_matrix = None
        self.prev_info = {}
        self.prev_time = time.time()

    def _define_zone(self):
        top_y = int(self.frame_size * 0.55)
        bottom_y = int(self.frame_size * 0.95)
        center_x = self.frame_size // 2
        top_width = 100
        bottom_width = 220
        return np.array([
            (center_x - top_width // 2, top_y),
            (center_x + top_width // 2, top_y),
            (center_x + bottom_width // 2, bottom_y),
            (center_x - bottom_width // 2, bottom_y),
        ], dtype=np.int32)

    def update_affine(self, gray):
        if self.prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=80, qualityLevel=0.3, minDistance=7)
            if p0 is not None:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None)
                if p1 is not None:
                    good_p0 = p0[st == 1]
                    good_p1 = p1[st == 1]
                    if len(good_p0) >= 6:
                        self.affine_matrix, _ = cv2.estimateAffinePartial2D(good_p0, good_p1)
        self.prev_gray = gray.copy()

    def track_and_detect(self, frame):
        frame_resized = cv2.resize(frame, (self.frame_size, self.frame_size))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        self.update_affine(gray)

        results = self.model.track(source=frame_resized, persist=True, stream=True, tracker=self.tracker_cfg)
        now = time.time()
        alerts = []

        for r in results:
            for box in r.boxes:
                if not hasattr(box, 'id') or box.id is None:
                    continue

                obj_id = int(box.id.item())
                cls = int(box.cls.item())
                if cls not in ALERT_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                bottom_center = (cx, y2)

                if obj_id in self.prev_info:
                    px, py, t_prev, prev_speed, prev_warn = self.prev_info[obj_id]
                    dt = now - t_prev

                    if self.affine_matrix is not None:
                        prev_pt = np.array([[[px, py]]], dtype=np.float32)
                        corrected = cv2.transform(prev_pt, self.affine_matrix)[0][0]
                        dx, dy = cx - corrected[0], cy - corrected[1]
                    else:
                        dx, dy = cx - px, cy - py

                    dist = np.hypot(dx, dy)
                    speed = dist / dt if 0.02 < dt < 1.0 else 0
                    if abs(speed - prev_speed) > 30:
                        speed = prev_speed
                else:
                    speed = 0
                    prev_warn = False

                in_zone = cv2.pointPolygonTest(self.zone_pts, bottom_center, False) >= 0
                speed_thresh = SPEED_THRESHOLDS.get(cls, 1.5)
                current_warn = in_zone and speed > speed_thresh
                warn = prev_warn and current_warn

                self.prev_info[obj_id] = (cx, cy, now, speed, current_warn)
                if warn:
                    alerts.append((cls, (x1, y1, x2, y2), speed))

        return frame_resized, alerts

