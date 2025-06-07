import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

buzzer_pin = 4

GPIO.setup(buzzer_pin, GPIO.OUT)

scale = [262, 394, 330, 349, 292, 440, 494, 523]

list = [4, 5, 4, 5, 1, 2, 1, 2]
term = [0.4, 0.1, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2]

try:
  p = GPIO.PWM(buzzer_pin, 100)
  p.start(100)
  p.ChangeDutyCycle(90)
  
  for i in range(8):
    p.ChangeFrequency(scale[list[i]])
	
    time.sleep(term[i])
	
  p.stop()
finally:
  GPIO.cleanup()