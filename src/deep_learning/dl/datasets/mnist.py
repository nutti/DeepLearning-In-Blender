import gzip
import urllib.request
import os
import numpy as np

DATASET_DIR = "./mnist"
MNIST_URL = "http://yann.lecun.com/exdb/mnist/"
MNIST_IMAGES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
}
MNIST_LABELS = {
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

def load_mnist_data():
    os.makedirs(DATASET_DIR, exist_ok=True)
    for f in MNIST_IMAGES.values():
        file_path = DATASET_DIR + "/" + f
        if os.path.exists(file_path):
            continue
        urllib.request.urlretrieve(MNIST_URL + f, file_path)
    
    for f in MNIST_LABELS.values():
        file_path = DATASET_DIR + "/" + f
        if os.path.exists(file_path):
            continue
        urllib.request.urlretrieve(MNIST_URL + f, file_path)

    data = {}
    for k, v in MNIST_IMAGES.items():
        file_path = DATASET_DIR + "/" + v
        with gzip.open(file_path, "rb") as f:
            data[k] = np.frombuffer(f.read(), np.uint8, offset=16)
            data[k] = data[k].reshape(-1, 28 * 28)
    for k, v in MNIST_LABELS.items():
        file_path = DATASET_DIR + "/" + v
        with gzip.open(file_path, "rb") as f:
            data[k] = np.frombuffer(f.read(), np.uint8, offset=8)
            data[k] = data[k].reshape(-1, 1)

    return data


def normalize_data(data):
    return data.astype(np.float32) / 255.0


def make_one_hot_label(label):
    max_label = np.max(label) + 1
    return np.eye(max_label)[label.flatten()]
