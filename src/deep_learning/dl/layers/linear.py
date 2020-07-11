import numpy as np
from collections import OrderedDict

from .layer_base import LayerBase

class LinearLayer(LayerBase):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.cache = {}

        self.params = OrderedDict()
        self.params["weight"] = np.zeros((out_features, in_features))
        self.params["bias"] = np.zeros((out_features, ))

        self.grads = OrderedDict()
        self.params["weight"] = np.zeros((out_features, in_features))
        self.params["bias"] = np.zeros((out_features, ))

    def id(self):
        return "Linear"
    
    def initialize_parameters(self, weight=None, bias=None):
        if weight is None:
            self.params["weight"] = np.random.randn(*self.params["weight"].shape) * 0.01
        else:
            assert self.params["weight"].shape == weight.shape, "shape of 'weight' must be {}, but {}".format(self.params["weight"].shape, weight.shape)
            self.params["weight"] = weight

        if bias is None:
            self.params["bias"] = np.zeros(self.params["bias"].shape)
        else:
            assert self.params["bias"].shape == bias.shape, "shape of 'bias' must be {}, but {}".format(self.params["bias"].shape, bias.shape)
            self.params["bias"] = bias

    def parameters(self):
        return self.params
    
    def gradients(self):
        return self.grads

    def forward(self, x):
        weight = self.params["weight"]
        bias = self.params["bias"]

        y = np.dot(x, weight.T) + bias

        self.cache["x"] = x

        return y
    
    def backward(self, dy):
        x = self.cache["x"]
        weight = self.params["weight"]
        bias = self.params["bias"]

        dw = np.dot(dy.T, x)
        db = np.sum(dy, axis=0)
        dx = np.dot(dy, weight)

        assert x.shape == dx.shape, "Shape does not match between x and dx"
        assert weight.shape == dw.shape, "Shape does not match between weight and dw"
        assert bias.shape == db.shape, "Shape does not match between bias and db"

        self.grads["weight"] = dw
        self.grads["bias"] = db

        return dx

