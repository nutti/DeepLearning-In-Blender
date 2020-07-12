import numpy as np

from .optimizer_base import OptimizerBase

class AdaGrad(OptimizerBase):
    def __init__(self, lr=0.01):
        super().__init__()

        self.lr = lr
        self.h = None

    def optimize(self, parameters, gradients):
        if self.h is None:
            self.h = {}
            for k, v in parameters.items():
                self.h[k] = np.zeros_like(v)

        for k in parameters.keys():
            self.h[k] += gradients[k] * gradients[k]
            parameters[k] -= self.lr * gradients[k] / np.sqrt(self.h[k] + 1e-7)

