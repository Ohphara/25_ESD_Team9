import cv2
import time
import numpy as np
from ultralytics import YOLO

# 위험 클래스 속도 임계값 (px/frame)
SPEED_THRESHOLDS = {
    1: 5.0,   # truck
    21: 5.0,  # car
    24: 2.0,  # bicycle
    22: 4.0,  # bus
    14: 3.0   # motorcycle
}
ALERT_CLASSES = SPEED_THRESHOLDS.keys()
FRAME_SIZE = 480
prev_info = {}

# YOLO 모델 및 영상 로드
model = YOLO("C:/Users/k9481/MyYoloProject/final_best3.pt")
cap = cv2.VideoCapture("C:/Users/k9481/Downloads/20250603_160952.mp4")
prev_time = time.time()

# 중심 위험 영역 정의 (가슴 높이 시점 기준)
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

# 회전 보정 관련 변수
prev_gray = None
affine_matrix = None
prev_affine_matrix = None
AFFINE_CHANGE_THRESHOLD = 8.0  # 너무 급격한 회전 무시

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # 특징점 기반 affine 보정 추정
    if prev_gray is not None:
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=80, qualityLevel=0.3, minDistance=7)
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
            if p1 is not None:
                good_p0 = p0[st == 1]
                good_p1 = p1[st == 1]
                if len(good_p0) >= 6:
                    new_affine, _ = cv2.estimateAffinePartial2D(good_p0, good_p1)
                    if new_affine is not None:
                        if prev_affine_matrix is not None:
                            diff = np.linalg.norm(new_affine - prev_affine_matrix)
                            if diff < AFFINE_CHANGE_THRESHOLD:
                                affine_matrix = new_affine
                        else:
                            affine_matrix = new_affine
                        prev_affine_matrix = affine_matrix
    prev_gray = gray.copy()

    # 객체 추적
    results = model.track(source=frame_resized, persist=True, stream=True, tracker="bytetrack.yaml")
    t_now = time.time()

    # 위험 영역 시각화
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

            # 속도 계산
            if obj_id in prev_info:
                px, py, t_prev, prev_speed = prev_info[obj_id]
                dt = t_now - t_prev

                if affine_matrix is not None:
                    prev_pt = np.array([[px, py]], dtype=np.float32)
                    corrected_pt = cv2.transform(prev_pt[None, :, :], affine_matrix)[0][0]
                    dx, dy = cx - corrected_pt[0], cy - corrected_pt[1]
                else:
                    dx, dy = cx - px, cy - py

                dist = np.hypot(dx, dy)
                if dt < 0.02 or dist > 100:
                    speed = 0
                else:
                    speed = dist / dt
                    # 속도 튐 방지
                    if abs(speed - prev_speed) > 10:
                        speed = prev_speed
            else:
                speed = 0

            prev_info[obj_id] = (cx, cy, t_now, speed)

            # 경보 조건
            in_zone = cv2.pointPolygonTest(zone_pts, (cx, cy), False) >= 0
            speed_thresh = SPEED_THRESHOLDS.get(cls, 1.5)
            warn = in_zone and speed > speed_thresh

            # 시각화
            label = f"{model.names[cls]} {speed:.1f}px/s"
            color = (0, 0, 255) if warn else (255, 255, 0)
            if warn:
                label += " ⚠"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # FPS 계산
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLOv8 + ByteTrack Alert System (Stabilized)", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
