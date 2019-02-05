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