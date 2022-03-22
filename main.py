import cv2
import time
from src.face_recognition import Face_Recognizer


face_recognizer = Face_Recognizer()
cam = cv2.VideoCapture(0)

cTime = 0
pTime = 0

while True:
    success, frame = cam.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_recognizer.recognize_frame(frameRGB)
    for pessoa in results:
        cv2.rectangle(frame, pessoa.position, (0, 255, 255), 2)
        cv2.putText(frame, f"{str(pessoa.name)} {int(pessoa.prob*100)}%", (pessoa.position[0], pessoa.position[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f"FPS:{int(fps)}", (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
