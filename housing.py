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


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing_X_train = train_set.drop("median_house_value", axis=1)

housing_X_lable = train_set["median_house_value"]