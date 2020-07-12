import numpy as np

from .initializer_base import InitializerBase


class StandardNormalInitializer(InitializerBase):
    def __init__(self, multiplier=0.01):
        super().__init__()

        self.multiplier = multiplier

    def init(self, shape):
        return np.random.randn(*shape) * self.multiplier

