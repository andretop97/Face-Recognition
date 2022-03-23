from keras_facenet import FaceNet
from typing import List
from joblib import dump
from joblib import load
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from src.handle_data import load_dataset, save_dataset
from src.face_detection.face_detection import Face_deteaction
from models.Pessoas import Pessoa
import os
import cv2


class Face_Recognizer:

    def __init__(self, dataPath: str = "./data", modelPath: str = "./training_models", method: str = "OPENCV") -> None:

        self.modelPath = modelPath
        self.face_detection = Face_deteaction(method).detector
        self.embedder = FaceNet()
        try:
            self.model = load(modelPath + "\\model.joblib")
            self.label_encoder = load(modelPath + "\\label_encoder.joblib")
        except:
            os.system('cls')
            print("Modelo nao encontrado")
            self.train_model(dataPath)

    def get_embedding(self, faces):
        embedder = FaceNet()
        embeddings = embedder.embeddings(faces)
        return embeddings

    def train_model(self, dataPath: str):

        save_dataset(dataPath)
        x_train,  y_train, x_test, y_test = load_dataset()

        x_train = self.get_embedding(x_train)
        x_test = self.get_embedding(x_test)

        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)

        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

        model = SVC(kernel='linear', probability=True)
        model.fit(x_train, y_train)

        self.model = model
        self.label_encoder = label_encoder

        self.save_model(self.modelPath, model, label_encoder)

    def save_model(self, modelPath: str, model, label_encoder) -> None:
        if not os.path.exists(modelPath):
            os.makedirs(modelPath)
        dump(model, modelPath + "\\model.joblib")
        dump(label_encoder, modelPath + "\\label_encoder.joblib")

    def recognize_sample(self, sample) -> list:
        sample = self.get_embedding(sample)
        predicted = self.model.predict(sample)
        prob = self.model.predict_proba(sample)
        prob = prob[0, predicted[0]]
        return self.label_encoder.inverse_transform(predicted), prob

    def recognize_frame(self, frame: list) -> List[Pessoa]:
        faces = self.face_detection.detect_faces_by_frame(frame)
        pessoas = list()
        for face in faces:
            x1, y1, width, height = face.box
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            frame_slice = frame[y1:y2, x1:x2]
            frame_slice = cv2.resize(frame_slice, (160, 160))
            results, prob = self.recognize_sample([frame_slice])
            pessoa = Pessoa(results[0], face.box, prob)
            pessoas.append(pessoa)

        return pessoas
