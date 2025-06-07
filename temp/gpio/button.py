import RPi.GPIO as GPIO
import time

LED = 4
KEY = 5

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED, GPIO.OUT)
GPIO.setup(KEY, GPIO.IN)
   
try:
	while True:
		if GPIO.input(KEY)==True:
			GPIO.output(LED, True)
		elif GPIO.input(KEY)==False:
			GPIO.output(LED, False)	
except KeyboardInterrupt:
      pass
finally:
      GPIO.cleanup()