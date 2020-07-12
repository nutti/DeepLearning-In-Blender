import numpy as np
from collections import OrderedDict

from .layer_base import LayerBase


class BatchNormalizationLayer(LayerBase):
    def __init__(self, gamma=1.0, beta=0.0):
        super().__init__()

        self.cache = {}

        self.params = OrderedDict()
        self.params["gamma"] = gamma
        self.params["beta"] = beta

        self.grads = OrderedDict()
        self.grads["gamma"] = 0
        self.grads["beta"] = 0

    def id(self):
        return "BatchNormalizationLayer"

    def parameters(self):
        return self.params

    def gradients(self):
        return self.grads

    def forward(self, x):
        gamma = self.params["gamma"]
        beta = self.params["beta"]

        mu = np.mean(x, axis=0)
        xmu = x - mu
        variance = np.mean(xmu**2, axis=0)
        sqrt_variance = np.sqrt(variance + 1e-7)
        inv_sqrt_variance = 1 / sqrt_variance
        xhat = xmu * inv_sqrt_variance
        y = gamma * xhat + beta

        self.cache["variance"] = variance
        self.cache["sqrt_variance"] = sqrt_variance
        self.cache["inv_sqrt_variance"] = inv_sqrt_variance
        self.cache["xhat"] = xhat
        self.cache["xmu"] = xmu
        self.cache["batch_size"] = x.shape[0]

        return y

    def backward(self, dy):
        xhat = self.cache["xhat"]
        xmu = self.cache["xmu"]
        inv_sqrt_variance = self.cache["inv_sqrt_variance"]
        sqrt_variance = self.cache["sqrt_variance"]
        variance = self.cache["variance"]
        batch_size = self.cache["batch_size"]
        gamma = self.params["gamma"]

        dbeta = dy * np.sum(dy, axis=0)
        dgamma = np.sum(dy * xhat, axis=0)
        dxhat = dy * gamma
        dxmu1 = dxhat * inv_sqrt_variance
        dinv_sqrt_variance = np.sum(dxhat * xmu, axis=0)
        dsqrt_variance = -dinv_sqrt_variance / sqrt_variance**2
        dvariance = 0.5 * dsqrt_variance / np.sqrt(variance + 1e-7)
        dmean = dvariance * np.ones(xmu.shape) / batch_size
        dxmu2 = 2.0 * xmu * dmean
        dx1 = dxmu1 + dxmu2
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
        dx2 = dmu * np.ones(xmu.shape) / batch_size
        dx = dx1 + dx2

        return dx
