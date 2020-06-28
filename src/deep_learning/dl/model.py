import numpy as np


class Model():
    def __init__(self):
        self.layers = []

    def initialize_params(self):
        for l in self.layers:
            l.initialize_parameters()

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_layers(self, layers):
        for l in layers:
            self.add_layer(l)

    def gradient(self, X, t):
        batch_size = X.shape[-1]

        tensor = X
        for l in self.layers:
            tensor = l.forward(tensor)

        tensor = self.layers[-1].backward(t, batch_size)
        for l in reversed(self.layers[:-1]):
            tensor = l.backward(tensor)

        grads = []
        for l in self.layers:
            grads.extend(l.gradients().values())

        return grads

    def predict(self, X):
        tensor = X
        for l in self.layers:
            tensor = l.forward(tensor)

        return tensor

    def loss(self, X, y):
        num_classes = 10
        batch_size = X.shape[-1]

        a2 = self.predict(X)
        l = -np.sum(np.sum(y * np.log(a2), axis=0)) / batch_size
        
        return l

    def update_paramters(self, grads, lr):
        for l in self.layers:
            for param_key in l.parameters().keys():
                l.parameters()[param_key] -= l.gradients()[param_key] * lr