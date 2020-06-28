import numpy as np

from .layer_base import LayerBase

class LinearLayer(LayerBase):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.cache = {}
        self.params = {
            "weight": np.zeros((out_features, in_features)),
            "bias": np.zeros((out_features, 1)),
        }
        self.grads = {
            "weight": np.zeros((out_features, in_features)),
            "bias": np.zeros((out_features, 1)),
        }
    
    def initialize_parameters(self):
        self.params["weight"] = np.random.randn(*self.params["weight"].shape) * 0.01
        self.params["bias"] = np.zeros(self.params["bias"].shape)

    def parameters(self):
        return self.params
    
    def gradients(self):
        return self.grads

    def forward(self, x):
        weight = self.params["weight"]
        bias = self.params["bias"]

        y = np.dot(weight, x) + bias

        self.cache["x"] = x

        return y
    
    def backward(self, dy):
        x = self.cache["x"]
        weight = self.params["weight"]

        dw = np.dot(dy, x.T)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = np.dot(weight.T, dy)

        self.grads["weight"] = dw
        self.grads["bias"] = db

        return dx
