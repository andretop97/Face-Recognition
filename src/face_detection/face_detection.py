from src.face_detection.MTCNN import MTCNN_detection
from src.face_detection.OpenCv import OpenCv_detection
from src.face_detection.Dlib import Dlib_detection
from src.face_detection.DlibCnn import DlibCnn_detection

from models.Face import Face
from typing import List


class Face_deteaction:
    def __init__(self, method: str = "OPENCV") -> None:
        if method == "MTCNN":
            self.detector = MTCNN_detection()
        elif method == "OPENCV":
            self.detector = OpenCv_detection()
        elif method == "DLIB":
            self.detector = Dlib_detection()
        elif method == "DLIBCNN":
            self.detector = DlibCnn_detection()
        else:
            raise Exception('Modelo invalido')

    def detect_faces_by_frame(self, frame) -> List[Face]:
        return self.detector.detect_faces_by_frame(frame)

    def detect_faces_by_path(self, path: str) -> List[Face]:
        return self.detector.detect_faces_by_path(path)

    def crop_faces(self, path: str) -> list:
        return self.detector.crop_faces(path)
