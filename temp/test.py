import cv2

cap = cv2.VideoCapture(0)  # 0은 첫 번째 웹캠

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 찾을 수 없습니다.")
        break
    cv2.imshow("USB Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
