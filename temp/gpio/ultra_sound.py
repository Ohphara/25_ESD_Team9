import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

trig = 3
echo = 2

GPIO.setup(trig, GPIO.OUT)
GPIO.setup(echo, GPIO.IN)

try :   
    while True:
        GPIO.output(trig, False)
        time.sleep(0.5)

        GPIO.output(trig, True)
        time.sleep(0.00001) 
        GPIO.output(trig, False)    

        while GPIO.input(echo) == 0:
            pulse_start = time.time()

        while GPIO.input(echo) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * (340*100) / 2
        distance = round(distance, 2)

        print("Distance : ", distance, "cm")
except KeyboardInterrupt:
      pass
finally:
      GPIO.cleanup()