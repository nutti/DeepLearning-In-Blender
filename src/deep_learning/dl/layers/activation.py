import numpy as np

from .layer_base import LayerBase

class TanhLayer(LayerBase):
    def __init__(self):
        super().__init__()

        self.cache = {}

    def forward(self, x):
        y = np.tanh(x)
        self.cache["y"] = y

        return y

    def backward(self, dy):
        y = self.cache["y"]

        return (1 - np.power(y, 2)) * dy


class SoftmaxLayer(LayerBase):
    def __init__(self):
        super().__init__()

        self.cache = {}

    def forward(self, x):
        c = np.max(x)
        sum_ = np.sum(np.exp(x - c), axis=0)
        y = np.exp(x - c) / np.sum(np.exp(x - c), axis=0)
        self.cache["y"] = y

        return y
    
    def backward(self, t, batch_size):
        y = self.cache["y"]
        return (y - t) / batch_size