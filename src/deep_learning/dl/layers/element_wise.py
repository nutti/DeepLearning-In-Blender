import numpy as np

from .layer_base import LayerBase


class AddLayer(LayerBase):
    def __init__(self):
        super().__init__()

        self.cache = {}

    def id(self):
        return "Add"

    def forward(self, x1, x2):
        y = x1 + x2

        return y

    def backward(self, dy):
        dx1 = dy
        dx2 = dy

        return dx1, dx2


class MulLayer(LayerBase):
    def __init__(self):
        super().__init__()

        self.cache = {}

    def id(self):
        return "Mul"

    def forward(self, x1, x2):
        y = x1 * x2

        self.cache["x1"] = x1
        self.cache["x2"] = x2

        return y

    def backward(self, dy):
        dx1 = dy * self.cache["x2"]
        dx2 = dy * self.cache["x1"]

        return dx1, dx2
