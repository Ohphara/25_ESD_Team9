import pyttsx3
import subprocess
import threading
import time
import queue

def speak_tts_worker(q):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say("ready")
    engine.runAndWait()
    while True:
        text = q.get()
        if text is None:
            break
        print(f"[TTS] Speaking: {text}")
        engine.say("   " + text)
        engine.runAndWait()

def select_input():
    sel = input("카메라(0)/동영상(1) 중 선택하세요: ").strip()
    if sel == "1":
        video_path = "/home/pi/Desktop/FinalProject/20250603_160952.mp4"
        log_sel = input("로그 저장(1:저장, 0:저장안함)? ").strip()
        if log_sel not in ["0", "1"]:
            log_sel = "1"
        return [video_path, log_sel]
    else:
        cam_idx = input("카메라 인덱스(기본 8): ").strip()
        if cam_idx == "":
            cam_idx = "8"
        log_sel = input("로그 저장(1:저장, 0:저장안함)? ").strip()
        if log_sel not in ["0", "1"]:
            log_sel = "0"
        return [f"cam:{cam_idx}", log_sel]

def main():
    args = select_input()
    proc = subprocess.Popen(['./yolo11n'] + args, stdout=subprocess.PIPE, text=True, bufsize=1)

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
                if message not in last_alert_time or (now - last_alert_time[message]) > COOLDOWN_SEC:
                    print(f"[ALERT] {message}")
                    tts_queue.put("   " + message)
                    last_alert_time[message] = now
                else:
                    print(f"[INFO] Skip duplicate alert: {message}")
            elif line != "":
                print(line)
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