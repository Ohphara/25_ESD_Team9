import time
from picamera2 import Picamera2

class CameraInput:
    def __init__(self, frame_size=640, warmup=1.0):
        self.frame_size = frame_size
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (frame_size, frame_size), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(warmup)

    def get_frame(self):
        return self.picam2.capture_array()

    def release(self):
        self.picam2.stop()

if __name__ == "__main__":
    import cv2
    cam = CameraInput(frame_size=640)
    print("Streaming live video. Press 'q' to quit.")
    while True:
        frame = cam.get_frame()
        cv2.imshow("CameraInput Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
