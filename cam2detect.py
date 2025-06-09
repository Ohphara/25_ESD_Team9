import cv2
import numpy as np
from multiprocessing import shared_memory
import subprocess
import struct
import time

frame_shape = (480, 640, 3)
frame_bytes = np.prod(frame_shape)

# 1. shared memory 생성 (최초 1회)
shm = shared_memory.SharedMemory(name='cam_frame', create=True, size=frame_bytes)
img = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)

print("[INFO] shared memory created, now launching C++ detect ...")
time.sleep(10)  # 10초 대기: shm 생성 완료를 확실히 보장

# 2. detect 프로세스 시작 (pipe 연결)
proc = subprocess.Popen(['./yolo11n_ncnn'], stdout=subprocess.PIPE)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img[:] = frame[:]


    # 4. detect 결과 파이프로 읽기 (동기화 flag/세마포어 필요! 여기선 단순 예시)
    # (C++이 detection 끝내고 쏴줌)
    data = proc.stdout.read(4) # N (detection 개수)
    n = struct.unpack('I', data)[0]
    dets = []
    for _ in range(n):
        buf = proc.stdout.read(4 + 4 + 4*4) # classid+prob+x1+y1+x2+y2
        classid, prob, x1, y1, x2, y2 = struct.unpack('ifffff', buf)
        dets.append((classid, prob, x1, y1, x2, y2))

    # 5. 시각화
    for classid, prob, x1, y1, x2, y2 in dets:
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(frame, f'{classid}:{prob:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("YOLOv11n Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
shm.close()
shm.unlink()