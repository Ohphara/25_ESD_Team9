from picamera2 import Picamera2, controls
from ultralytics import YOLO
import cv2

picam = Picamera2()
preview_config = picam.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 30}  # 여기에 FPS 설정
)
picam.configure(preview_config)
picam.start()

model = YOLO("yolo11n_ncnn_model")  # .pt, .onnx, ncnn 포맷 등

while True:
    frame = picam.capture_array()            # numpy RGB(480,640,3)
    res = model(frame, conf=0.5)             # Ultralytics API
    annotated = res[0].plot()
    cv2.imshow("YOLO", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
