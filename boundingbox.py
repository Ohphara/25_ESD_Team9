import pathlib
import sys

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath


import sys
import cv2
import torch

# yolov5 경로 추가
sys.path.append('C:/Users/k9481/MyYoloProject/yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# 모델 로드
device = select_device('')
model = DetectMultiBackend('C:/Users/k9481/MyYoloProject/best.pt', device=device)

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device)
    img = img.permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)
    img = img.unsqueeze(0)

    # 추론
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred)

    # 결과 출력
    print(pred)

    # 프레임 보여주기
    cv2.imshow('YOLOv5 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
