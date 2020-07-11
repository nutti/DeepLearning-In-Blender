import numpy as np

from .layer_base import LayerBase


class ReluLayer(LayerBase):
    def __init__(self):
        super().__init__()

        self.cache = {}

    def id(self):
        return "Relu"

    def forward(self, x):
        y = np.maximum(x, 0)

        self.cache["is_negative"] = (x < 0)

        return y

    def backward(self, dy):
        is_negative = self.cache["is_negative"]

        dx = dy
        dx[is_negative] = 0

        return dx


class SigmoidLayer(LayerBase):
    def __init__(self):
        super().__init__()

        self.cache = {}

    def id(self):
        return "Sigmoid"

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))

        self.cache["y"] = y

        return y

    def backward(self, dy):
        y = self.cache["y"]

        dx = y * (1 - y) * dy

        return dx


class TanhLayer(LayerBase):
    def __init__(self):
        super().__init__()

        self.cache = {}

    def id(self):
        return "Tanh"

    def forward(self, x):
        y = np.tanh(x)
        self.cache["y"] = y

        return y

    def backward(self, dy):
        y = self.cache["y"]

        dx = (1 - np.power(y, 2)) * dy

        return dx


class SoftmaxWithLossLayer(LayerBase):
    def __init__(self):
        super().__init__()

        self.cache = {}

    def id(self):
        return "SoftmaxWithLoss"

    def forward(self, x, target):
        batch_size = target.shape[0]

        c = np.max(x)
        y = np.exp(x - c) / np.sum(np.exp(x - c), axis=1, keepdims=True)
        loss = -np.sum(np.sum(target * np.log(y), axis=1)) / batch_size

        self.cache["target"] = target.copy()
        self.cache["y"] = y.copy()

        return loss

    def backward(self, dy=1):
        y = self.cache["y"].copy()
        target = self.cache["target"].copy()
        batch_size = target.shape[0]

        dx = dy * (y - target) / batch_size

        return dx

