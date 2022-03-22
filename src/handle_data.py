from typing import Tuple
from os import listdir, system
from os.path import isdir
from src.face_detection.MTCNN import MTCNN_detection
from sklearn.model_selection import train_test_split
from numpy import savez_compressed, load


def get_faces_from_directory(dirPath: str) -> list:

    face_detection = MTCNN_detection()
    faces_list = list()
    for filename in listdir(dirPath):
        path = dirPath + filename
        faces = face_detection.crop_faces(path)
        for face in faces:
            faces_list.append(face)
    return faces_list


def load_data(directory: str) -> Tuple[list, list]:
    X, y = list(), list()
    system('cls')

    for subdir in listdir(directory):
        path = directory + "/" + subdir + '/'

        if not isdir(path):
            continue

        faces = get_faces_from_directory(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return X, y


def save_dataset(directory: str, filename: str = "./data/dataset.npz") -> None:
    X, y = load_data(directory)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    savez_compressed(filename, x_train, y_train, x_test, y_test)


def load_dataset(path: str = "./data/dataset.npz") -> Tuple[list, list, list, list]:
    data = load(path, allow_pickle=True)
    return data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
