import pyttsx3
import subprocess
import threading
import time
import queue
import sys

def speak_tts_worker(q):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    while True:
        text = q.get()
        if text is None:
            break
        print(f"[TTS] Speaking: {text}")
        engine.say(text)
        engine.runAndWait()

def main():
    # subprocess로 yolo11n 실행
    proc = subprocess.Popen(['./yolo11n'], stdout=subprocess.PIPE, text=True, bufsize=1)
    
    tts_queue = queue.Queue()
    tts_thread = threading.Thread(target=speak_tts_worker, args=(tts_queue,), daemon=True)
    tts_thread.start()

    last_alert_time = {}
    COOLDOWN_SEC = 3.0   # 같은 메시지 반복 알림 쿨다운 (초)

    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("ALERT:"):
                message = line[6:].strip()
                now = time.time()
                # 같은 메시지 쿨타임 관리
                if message not in last_alert_time or (now - last_alert_time[message]) > COOLDOWN_SEC:
                    tts_queue.put(message)
                    last_alert_time[message] = now
                else:
                    print(f"[INFO] Skip duplicate alert: {message}")
    except KeyboardInterrupt:
        print("종료 요청됨 (Ctrl+C)")
    finally:
        tts_queue.put(None)  # TTS 쓰레드 안전종료
        tts_thread.join()
        proc.terminate()
        proc.wait()
        print("프로그램 종료")

if __name__ == "__main__":
    main()
