import RPi.GPIO as GPIO
import time

servoPin          = 2
SERVO_MAX_DUTY    = 12
SERVO_MIN_DUTY    = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)
servo.start(0)


def servo_control(degree, delay):
  if degree > 180:
    degree = 180

  duty = SERVO_MIN_DUTY+(degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
  print("Degree: {} to {}(Duty)".format(degree, duty))
  servo.ChangeDutyCycle(duty)
  time.sleep(delay)
  
try :  
    for i in range(1, 180, 10):
        servo_control(i, 0.1)

except KeyboardInterrupt:
    pass
finally:
    servo.stop()
    GPIO.cleanup()