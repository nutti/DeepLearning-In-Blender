import numpy as np

from .optimizer_base import OptimizerBase

class SGD(OptimizerBase):
    def __init__(self, lr=0.01, momentum=0):
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def optimize(self, parameters, gradients):
        if self.velocity is None:
            self.velocity = {}
            for k, v in parameters.items():
                self.velocity[k] = np.zeros_like(v)

        for k in parameters.keys():
            self.velocity[k] = self.momentum * self.velocity[k] - self.lr * gradients[k]
            parameters[k] += self.velocity[k]

