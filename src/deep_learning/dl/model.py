import numpy as np

from . import gradient_check


class Model():
    def __init__(self):
        self.layers = []

    def initialize_params(self, initializer=None):
        for l in self.layers:
            l.initialize_parameters(initializer)

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_layers(self, layers):
        for l in layers:
            self.add_layer(l)

    def numerical_gradient(self, X, t):
        loss = lambda p : self.loss(X, t)

        grads = {}
        for l in self.layers:
            for param_key in l.parameters().keys():
                k = "{}:{}".format(l.name(), param_key)
                grads[k] = gradient_check.numerical_gradient(loss, l.parameters()[param_key])

        return grads

    def check_gradient(self, X, t):
        grads_1 = self.gradient(X, t)
        grads_2 = self.numerical_gradient(X, t)

        for k in grads_1.keys():
            diff = np.average(np.abs(grads_1[k] - grads_2[k]))
            assert diff < 1e-9, "Gradient check failed. (grad name: {}, diff: {})".format(k, diff)

    def gradient(self, X, t):
        batch_size = t.shape[0]

        tensor = X
        for l in self.layers[:-1]:
            tensor = l.forward(tensor)
        tensor = self.layers[-1].forward(tensor, t)

        tensor = np.ones((1, ))
        for l in reversed(self.layers):
            tensor = l.backward(tensor)

        grads = {}
        for l in self.layers:
            for param_key in l.parameters().keys():
                k = "{}:{}".format(l.name(), param_key)
                grads[k] = l.gradients()[param_key]

        return grads

    def predict(self, X):
        tensor = X
        for l in self.layers[:-1]:
            tensor = l.forward(tensor)

        return tensor

    def loss(self, X, t):
        p = self.predict(X)
        l = self.layers[-1].forward(p, t)

        return l

    def update_paramters(self, optimizer):
        params = {}
        grads = {}
        for l in self.layers:
            for param_key in l.parameters().keys():
                k = "{}:{}".format(l.name(), param_key)
                params[k] = l.parameters()[param_key]
                grads[k] = l.gradients()[param_key]

        optimizer.optimize(params, grads)

