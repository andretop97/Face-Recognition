from typing import Tuple


class Face:
    def __init__(self, box: Tuple[int, int, int, int]) -> None:
        self.box = box
