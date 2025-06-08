import cv2
import time
from ultralytics import YOLO
from camera_input import CameraInput
from tts import speak
from utils import (
    FRAME_SIZE, TTC_THRESHOLD, get_zone_pts,
    compute_distance_m, compute_ttc,
    draw_zone, draw_detection, draw_fps
)

prev_info = {}
last_alert_obj = {}

model = YOLO("/home/pi/Desktop/project/yolo11n_480/yolo11n_480_best_ncnn_model")
cam = CameraInput(frame_size=640)
prev_time = time.time()

zone_pts = get_zone_pts(FRAME_SIZE)

prev_gray = None
affine_matrix = None

while True:
    frame = cam.get_frame()
    frame_resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=80, qualityLevel=0.3, minDistance=7)
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
            if p1 is not None:
                good_p0 = p0[st == 1]
                good_p1 = p1[st == 1]
                if len(good_p0) >= 6:
                    affine_matrix, _ = cv2.estimateAffinePartial2D(good_p0, good_p1)
    prev_gray = gray.copy()

    results = model.track(source=frame_resized, persist=True, stream=True, tracker="bytetrack.yaml")
    t_now = time.time()

    draw_zone(frame_resized, zone_pts)

    for r in results:
        for box in r.boxes:
            if not hasattr(box, 'id') or box.id is None:
                continue

            obj_id = int(box.id.item())
            cls = int(box.cls.item())
            if cls not in model.names:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_width_px = max(1, x2 - x1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bottom_center = (cx, y2)

            distance_m = compute_distance_m(cls, box_width_px)

            if obj_id in prev_info:
                px, py, t_prev, prev_distance_m, prev_ttc, prev_warn_candidate = prev_info[obj_id]
                dt = t_now - t_prev
                v, ttc = compute_ttc(prev_distance_m, distance_m, dt)
            else:
                v = 0
                ttc = float('inf')
                prev_warn_candidate = False

            in_zone = cv2.pointPolygonTest(zone_pts, bottom_center, False) >= 0
            current_warn_candidate = in_zone and ttc < TTC_THRESHOLD

            warn = prev_warn_candidate and current_warn_candidate

            prev_info[obj_id] = (cx, cy, t_now, distance_m, ttc, current_warn_candidate)

            last_warn = last_alert_obj.get(obj_id, False)
            if warn and not last_warn:
                class_name = model.names[cls] if hasattr(model, "names") else str(cls)
                tts_msg = f"Warning: possible collision with {class_name}, {distance_m:.1f} meters away, time to collision {ttc:.1f} seconds."
                print("TTS:", tts_msg)
                speak(tts_msg)
                last_alert_obj[obj_id] = True
            elif not warn:
                last_alert_obj[obj_id] = False

            label = f"{model.names[cls]} {distance_m:.1f}m TTC:{ttc:.1f}s"
            draw_detection(frame_resized, x1, y1, x2, y2, label, warn)

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    draw_fps(frame_resized, fps)

    cv2.imshow("YOLOv11n + ByteTrack Alert System", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
