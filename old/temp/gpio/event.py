import RPi.GPIO as GPIO
import time

LED = 4
KEY = 5

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED, GPIO.OUT)
GPIO.setup(KEY, GPIO.IN)

def isr_key_event(pin):
    print("Key is pressed [%d]" %pin)
    if GPIO.input(LED)==True:
        GPIO.output(LED, False)
    elif GPIO.input(LED)==False:
        GPIO.output(LED, True)

GPIO.add_event_detect(KEY, GPIO.FALLING, callback=isr_key_event, bouncetime=300)
 
sec = 0
try:
    while True:
        print("sec = %d" %sec)
        sec = sec + 1
        time.sleep(1)
except KeyboardInterrupt:
      pass
finally:
      GPIO.cleanup()