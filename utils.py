import os
import matplotlib.pyplot as plt


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(".", "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def load_mnist():
    from scipy.io import loadmat
    fpath = os.path.join(".", 'mldata', 'mnist-original.mat')
    mnist = loadmat(fpath)

    X = mnist['data'].transpose()
    y = mnist['label'].transpose()
    return X, y


def load_mnist_tf():
    import tensorflow as tf
    import numpy as np
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return (X_train, X_test, y_train, y_test)


def shuffle_batch(X, y, batch_size):
    import numpy as np
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


import pandas as pd


def load_data(path, filename):
    csv = os.path.join(path, filename)
    return pd.read_csv(csv)
