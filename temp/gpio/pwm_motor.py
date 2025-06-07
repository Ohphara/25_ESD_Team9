import RPi.GPIO as GPIO
import time

MOTOR_P = 20
MOTOR_M = 21
KEY = 5

GPIO.setmode(GPIO.BCM)
GPIO.setup(KEY, GPIO.IN)
GPIO.setup(MOTOR_P, GPIO.OUT)
GPIO.setup(MOTOR_M, GPIO.OUT)

try:
    pwm_p = GPIO.PWM(MOTOR_P, 100)
    pwm_m = GPIO.PWM(MOTOR_M, 100)
    pwm_p.start(0)
    pwm_m.start(0)
    while True:
        if GPIO.input(KEY)==True:
            pwm_m.ChangeDutyCycle(0)
            pwm_p.ChangeDutyCycle(30)
        elif GPIO.input(KEY)==False:
            pwm_p.ChangeDutyCycle(0)
            pwm_m.ChangeDutyCycle(30)

        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    pwm_m.stop()
    pwm_p.stop()
    GPIO.cleanup()