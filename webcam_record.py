from picamera2 import Picamera2
import cv2
import os

# Find the next available filename
def get_next_filename(base='output', ext='.mp4'):
    idx = 1
    while True:
        filename = f"{base}{idx}{ext}"
        if not os.path.exists(filename):
            return filename
        idx += 1

filename = get_next_filename()

picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(video_config)
picam2.start()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
width = 640
height = 480
out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

print(f"Recording started! Saving to {filename}. Press 'q' to stop.")

frame_count = 0

while True:
    frame = picam2.capture_array()
    print("Captured frame shape:", frame.shape)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
    frame_count += 1
    cv2.imshow('Picamera2', frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Total frames written: {frame_count}")
out.release()
picam2.stop()
cv2.destroyAllWindows()
print("Recording saved!")
