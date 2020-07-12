import numpy as np

from .layer_base import LayerBase


class DropoutLayer(LayerBase):
    def __init__(self, drop_ratio=0.4):
        super().__init__()

        self.cache = {}

        self.hparams = {
            "drop_ratio": drop_ratio,
        }

    def id(self):
        return "Dropout"

    def forward(self, x):
        drop_ratio = self.hparams["drop_ratio"]

        mask = np.random.rand(*x.shape) > drop_ratio
        # TODO: add train flag.
        y = x * mask

        self.cache["mask"] = mask

        return y

    def backward(self, dy):
        mask = self.cache["mask"]

        dx = dy * mask

        return dx

