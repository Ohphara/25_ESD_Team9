import RPi.GPIO as GPIO
import time

LED = 4

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED, GPIO.OUT, initial=GPIO.LOW)
for i in range(1, 6):
   GPIO.output(LED, True)
   time.sleep(1)
   GPIO.output(LED, False)
   time.sleep(1)