import numpy as np

from .optimizer_base import OptimizerBase

class Adam(OptimizerBase):
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999):
        super().__init__()

        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = None
        self.v = None

    def optimize(self, parameters, gradients):
        if self.m is None:
            self.m = {}
            for k, v in parameters.items():
                self.m[k] = np.zeros_like(v)
        if self.v is None:
            self.v = {}
            for k ,v in parameters.items():
                self.v[k] = np.zeros_like(v)

        for k in parameters.keys():
            self.m[k] = self.beta_1 * self.m[k] + (1 - self.beta_1) * gradients[k]
            self.v[k] = self.beta_2 * self.v[k] + (1 - self.beta_2) * gradients[k]**2 
            m_hat = self.m[k] / (1 - self.beta_1)
            v_hat = self.v[k] / (1 - self.beta_2)
            delta_w = -self.alpha * m_hat / (np.sqrt(v_hat) + 1e-7)
            parameters[k] += delta_w

