from mtcnn import MTCNN
from models.Face import Face
from typing import List
import cv2


class MTCNN_detection:
    def __init__(self) -> None:
        self.detector = MTCNN()

    def detect_faces_by_frame(self, frame) -> List[Face]:
        faces = self.detector.detect_faces(frame)
        return [Face(face["box"]) for face in faces]

    def detect_faces_by_path(self, path: str) -> List[Face]:
        image = cv2.imread(path)
        return self.detect_faces_by_frame(image)

    def crop_faces(self, path: str) -> list:
        image = cv2.imread(path)
        results = self.detect_faces_by_frame(image)

        images = list()

        for face in results:
            x1, y1, width, height = face['box']
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = image[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            images.append(face)

        return images
