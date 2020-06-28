import numpy as np
import warnings

from dl.model import Model
from dl.layers.activation import TanhLayer, SoftmaxLayer
from dl.layers.linear import LinearLayer
from dl.datasets import mnist

warnings.resetwarnings()
warnings.simplefilter("error")


def simple_nn(images, labels, test_images, test_labels):
    learning_rate = 0.1
    batch_size = 100
    num_batches = int(labels.shape[-1] / batch_size)
    epochs = 10

    model = Model()
    model.add_layers([
        LinearLayer(28*28, 256),
        TanhLayer(),
        LinearLayer(256, 10),
        SoftmaxLayer(),
    ])
    model.initialize_params()

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