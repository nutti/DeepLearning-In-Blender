import numpy as np
import warnings

from dl.layers.activation import TanhLayer, SoftmaxLayer
from dl.layers.linear import LinearLayer
from dl.datasets import mnist

warnings.resetwarnings()
warnings.simplefilter("error")


class Model():
    def __init__(self):
        self.initialize_params()

        self.dense_1 = LinearLayer()
        self.act_1 = TanhLayer()
        self.dense_2 = LinearLayer()
        self.act_2 = SoftmaxLayer()

        self.cache = {}

    def initialize_params(self):
        self.params = {
            "W1": np.random.randn(256, 28*28) * 0.01,
            "b1": np.zeros((256, 1)),
            "W2": np.random.randn(10, 256) * 0.01,
            "b2": np.zeros((10, 1)),
        }

    def gradient_check(self, X, y):
        orig_params = self.params.copy()
        epsilon = 1e-7
        num_classes = 10

        for k in self.params:
            # original
            self.params = orig_params.copy()
            grads = self.gradient(X, y)

            # theta plus
            self.params = orig_params.copy()
            self.params[k][0] += epsilon
            loss_plus = self.loss(X, y)

            # theta minus
            self.params = orig_params.copy()
            self.params[k][0] -= epsilon
            loss_minus = self.loss(X, y)

            grad = grads["d"+k]
            grad_approx = (loss_plus - loss_minus) / (2 * epsilon)
            
            numerator = np.linalg.norm(grad - grad_approx, ord=2)
            denominator = np.linalg.norm(grad, ord=2) + np.linalg.norm([grad_approx], ord=2)
            difference = numerator / denominator
            print("{}: {}".format(k, difference))

    def gradient(self, X, t):
        batch_size = X.shape[-1]

        z1 = self.dense_1.forward(X, self.params["W1"], self.params["b1"])
        a1 = self.act_1.forward(z1)
        z2 = self.dense_2.forward(a1, self.params["W2"], self.params["b2"])
        y = self.act_2.forward(z2)

        dz2 = self.act_2.backward(t, batch_size)
        da1, dw2, db2 = self.dense_2.backward(dz2)
        dz1 = self.act_1.backward(da1)
        dx1, dw1, db1 = self.dense_1.backward(dz1)

        grads = {
            "dW1": dw1,
            "db1": db1,
            "dW2": dw2,
            "db2": db2,
        }

        return grads

    def predict(self, X):
        a1 = self.dense_1.forward(X, self.params["W1"], self.params["b1"])
        z1 = self.act_1.forward(a1)
        a2 = self.dense_2.forward(z1, self.params["W2"], self.params["b2"])
        y = self.act_2.forward(a2)

        return y

    def loss(self, X, y):
        num_classes = 10
        batch_size = X.shape[-1]

        a2 = self.predict(X)
        l = -np.sum(np.sum(y * np.log(a2), axis=0)) / batch_size
        
        return l

    def update_paramters(self, grads, lr):
        self.params["W1"] -= grads["dW1"] * lr
        self.params["b1"] -= grads["db1"] * lr
        self.params["W2"] -= grads["dW2"] * lr
        self.params["b2"] -= grads["db2"] * lr


def simple_nn(images, labels, test_images, test_labels):
    learning_rate = 0.1
    num_classes = 10
    batch_size = 100
    num_batches = int(labels.shape[-1] / batch_size)
    epochs = 10

    model = Model()

    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        for batch in range(num_batches):
            batch_mask = np.random.choice(labels.shape[-1], batch_size)
            X_train = images[:, batch_mask]
            y_train = labels[:, batch_mask]

            # gradient
            grads = model.gradient(X_train, y_train)

            # calculate loss
            if batch % 100 == 0:
                loss = model.loss(X_train, y_train)
                print("Batch {}: Loss = {}".format(batch, loss))

            # update
            model.update_paramters(grads, learning_rate)


        # predict
        a2 = model.predict(test_images)
        print(np.sum(np.argmax(np.log(a2), axis=0) == np.argmax(test_labels, axis=0)) / test_labels.shape[-1])


if __name__ == "__main__":
    np.random.seed(1)
    data = mnist.load_mnist_data()

    data["train_images"] = mnist.normalize_data(data["train_images"]) 
    data["test_images"] = mnist.normalize_data(data["test_images"])
    data["train_labels"] = mnist.make_one_hot_label(data["train_labels"])
    data["test_labels"] = mnist.make_one_hot_label(data["test_labels"])

    simple_nn(data["train_images"], data["train_labels"],
              data["test_images"], data["test_labels"])