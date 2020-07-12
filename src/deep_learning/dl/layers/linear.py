import numpy as np
from collections import OrderedDict

from .layer_base import LayerBase
from ..initializers.normal import StandardNormalInitializer

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
    
    def initialize_parameters(self, initializer_weight=None):
        if initializer_weight is None:
            self.params["weight"] = StandardNormalInitializer(0.01).init(self.params["weight"].shape)
        else:
            self.params["weight"] = initializer_weight.init(self.params["weight"].shape)

        self.params["bias"] = np.zeros(self.params["bias"].shape)

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

