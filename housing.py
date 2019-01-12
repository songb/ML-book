import os
import tarfile
from six.moves import urllib

DOWNLOAD = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
HOUSING_PATH= os.path.join("datasets", "housing")

def fetching_data(url=DOWNLOAD, path=HOUSING_PATH):
    if not os.path.isdir(path):
        os.makedirs(path)

    fpath = os.path.join(path,  "housing.tgz")

    urllib.request.urlretrieve(url, fpath)
    tgzf = tarfile.open(fpath)
    tgzf.extractall(path=path)
    tgzf.close()

fetching_data()

import pandas as pd

def load_data(path=HOUSING_PATH):
    csv = os.path.join(path, "housing.csv")
    return pd.read_csv(csv)

housing = load_data()


# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

import numpy as np

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)

from sklearn.model_selection import StratifiedShuffleSplit

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from pandas.plotting import scatter_matrix