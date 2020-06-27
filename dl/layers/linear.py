import numpy as np

class LinearLayer:
    def __init__(self):
        self.cache = {}
    
    def forward(self, x, weight, bias):
        y = np.dot(weight, x) + bias
        self.cache["weight"] = weight
        self.cache["x"] = x
        return y
    
    def backward(self, dy):
        x = self.cache["x"]
        weight = self.cache["weight"]

        dw = np.dot(dy, x.T)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = np.dot(weight.T, dy)

        return dx, dw, db
