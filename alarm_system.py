import cv2
import time
import numpy as np
from ultralytics import YOLO

    # 위험 클래스와 속도 임계값 (단위: px/frame)
    SPEED_THRESHOLDS = {
        1: 5.0,   # truck
        21: 5.0,  # car
        24: 1.5,  # bicycle
        22: 4.0,  # bus
        14: 3.0   # motorcycle
    }

    ALERT_CLASSES = SPEED_THRESHOLDS.keys()
    FRAME_SIZE = 480
    prev_info = {}  # 객체 ID → (cx, cy, timestamp, speed)

    # 모델 로드
    model = YOLO("C:/Users/k9481/MyYoloProject/final_best3.pt")

    # 영상 입력
    cap = cv2.VideoCapture("C:/Users/k9481/Downloads/20250603_160952.mp4")
    prev_time = time.time()

    # 역 사다리꼴 중심 영역 정의 (영상 해상도 기준)
    top_y = FRAME_SIZE // 3
    bottom_y = FRAME_SIZE
    center_x = FRAME_SIZE // 2
    top_width = 100
    bottom_width = 200
    zone_pts = np.array([
        (center_x - top_width // 2, top_y),
        (center_x + top_width // 2, top_y),
        (center_x + bottom_width // 2, bottom_y),
        (center_x - bottom_width // 2, bottom_y),
    ], dtype=np.int32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        results = model.track(source=frame_resized, persist=True, stream=True, tracker="bytetrack.yaml")
        t_now = time.time()

        # 중심 위험 영역 표시 (녹색 다각형)
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
                    dx, dy = cx - px, cy - py
                    dist = (dx ** 2 + dy ** 2) ** 0.5

                    if dt < 0.02 or dist > 100:
                        speed = 0
                    else:
                        speed = dist / dt
                        if abs(speed - prev_speed) > 30:
                            speed = prev_speed
                else:
                    speed = 0

                # 정보 갱신
                prev_info[obj_id] = (cx, cy, t_now, speed)

                # 중심영역 + 속도 초과 시 경보
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

        # FPS 계산 및 표시
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 결과 출력
        cv2.imshow("YOLOv8 + BoT-SORT Alert System", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
