from src.face_detection.MTCNN import MTCNN_detection
from src.face_detection.OpenCv import OpenCv_detection
from src.face_detection.Dlib import Dlib_detection
from src.face_detection.DlibCnn import DlibCnn_detection


class Face_deteaction:
    def __init__(self, method: str) -> None:
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
