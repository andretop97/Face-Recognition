from typing import List
import cv2
from models.Face import Face


class OpenCv_detection:
    def __init__(self) -> None:
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                              'haarcascade_frontalface_default.xml')

    def detect_faces_by_frame(self, frame) -> List[Face]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(
            20, 20), flags=cv2.CASCADE_SCALE_IMAGE)

        return [Face(face) for face in faces]

    def detect_faces_by_path(self, path: str) -> List[Face]:
        image = cv2.imread(path)
        return self.detect_faces_by_frame(image)

    def crop_faces(self, path: str) -> list:
        image = cv2.imread(path)
        results = self.detect_faces_by_frame(image)

        images = list()

        for face in results:
            x1, y1, width, height = face.box
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = image[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            images.append(face)

        return images
