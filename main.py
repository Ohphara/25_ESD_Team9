import pyttsx3
import subprocess
import threading
import time
import queue

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
    COOLDOWN_SEC = 3.0

    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("ALERT:"):
                message = line[6:].strip()
                now = time.time()
                # 쿨다운 체크
                if message not in last_alert_time or (now - last_alert_time[message]) > COOLDOWN_SEC:
                    print(f"[ALERT] {message}")  # ★ 터미널에도 출력!
                    tts_queue.put(message)
                    last_alert_time[message] = now
                else:
                    print(f"[INFO] Skip duplicate alert: {message}")
            else:
                print(line)  # ★ 일반 출력도 터미널로 pass-through
    except KeyboardInterrupt:
        print("종료 요청됨 (Ctrl+C)")
    finally:
        tts_queue.put(None)
        tts_thread.join()
        proc.terminate()
        proc.wait()
        print("프로그램 종료")

if __name__ == "__main__":
    main()
