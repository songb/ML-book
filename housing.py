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

housing_X_lable = train_set["median_house_value"].copy()

housing_X_train_number_only = housing_X_train.drop('ocean_proximity', axis=1)

from sklearn.preprocessing import Imputer

imputer = Imputer()
temp = imputer.fit_transform(housing_X_train_number_only)
housing_X_train_number_only = pd.DataFrame(temp, columns=housing_X_train_number_only.columns)


#transfer text to num
housing_ocean_1d = housing_X_train['ocean_proximity']

import numpy as np
housing_ocean = np.reshape(housing_ocean_1d.values, (-1,1))

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_ocean_one_hot_encoded = encoder.fit_transform(housing_ocean)

pass