from picamera2 import Picamera2

# IMX219 sensor specifications -- hard coding info
FOCAL_LENGTH_MM = 3.04       # millimeters
SENSOR_WIDTH_MM = 3.674      # millimeters
SENSOR_HEIGHT_MM = 2.760     # millimeters

# Desired image resolution
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Initialize Picamera2 and set resolution
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (IMAGE_WIDTH, IMAGE_HEIGHT)})
picam2.configure(config)
picam2.start()

# Calculate focal length in pixels (horizontal direction)
focal_length_px = IMAGE_WIDTH * FOCAL_LENGTH_MM / SENSOR_WIDTH_MM

print("=== Camera Info (IMX219 typical) ===")
print(f"Focal Length (mm): {FOCAL_LENGTH_MM}")
print(f"Sensor Width (mm): {SENSOR_WIDTH_MM}")
print(f"Sensor Height (mm): {SENSOR_HEIGHT_MM}")
print(f"Image Width (pixels): {IMAGE_WIDTH}")
print(f"Focal Length (pixels): {focal_length_px:.2f}")

picam2.stop()

