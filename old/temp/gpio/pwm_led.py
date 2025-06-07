import RPi.GPIO as GPIO
import time

LED = 4

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED, GPIO.OUT)

try:
    pwm = GPIO.PWM(LED, 100)
    pwm.start(0)
    while True:
        for i in range(10):
             pwm.ChangeDutyCycle(i*10)
             time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
	pwm.stop()
	GPIO.cleanup()