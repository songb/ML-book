import os
import tarfile
from six.moves import urllib

DOWNLOAD = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
HOUSING_PATH = os.path.join("datasets", "housing")


def fetching_data(url=DOWNLOAD, path=HOUSING_PATH):
    if not os.path.isdir(path):
        os.makedirs(path)

    fpath = os.path.join(path, "housing.tgz")

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


housing_X_test = test_set.drop('median_house_value', axis=1)
housing_y_test = test_set['median_house_value']

housing_X_train_number_only = housing_X_train.drop('ocean_proximity', axis=1)

from sklearn.preprocessing import Imputer

# replace empty with mean
imputer = Imputer()
temp = imputer.fit_transform(housing_X_train_number_only)
housing_X_train_number_only = pd.DataFrame(temp, columns=housing_X_train_number_only.columns)

# transfer text to num
housing_ocean_1d = housing_X_train['ocean_proximity']

import numpy as np

housing_ocean = np.reshape(housing_ocean_1d.values, (-1, 1))

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
housing_ocean_one_hot_encoded = encoder.fit_transform(housing_ocean)


from sklearn.base import BaseEstimator, TransformerMixin


class AddAttr(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        room_per_household = X[:, 3] / X[:, 6]
        pop_per_household = X[:, 5] / X[:, 6]
        bedroom_per_room = X[:, 4] / X[:, 3]
        return np.c_[X, room_per_household, pop_per_household, bedroom_per_room]


class KeepNumOnly(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop('ocean_proximity', axis=1)


class TextColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.reshape(X['ocean_proximity'].values, (-1,1))


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('num_only', KeepNumOnly()), ('imputer', Imputer(strategy='median')), ('add_attr', AddAttr()),
                         ('scaler', StandardScaler())])

text_pipeline = Pipeline([('text', TextColumns()), ('encoder', OneHotEncoder())])

from sklearn.pipeline import FeatureUnion
unioned_pipeline = FeatureUnion(transformer_list=[('number', num_pipeline), ('text', text_pipeline)])


final_housing_data = pd.DataFrame(unioned_pipeline.fit_transform(housing_X_train).toarray())


housing_test_processed = pd.DataFrame(unioned_pipeline.fit_transform(housing_X_test).toarray())


def train_test(train_X, train_y, test_X, test_y, model):
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    rmse = np.sqrt(mean_squared_error(test_y,pred))
    print(model, rmse)
    return rmse

from functools import partial
pf = partial(train_test, final_housing_data, housing_X_lable, housing_test_processed, housing_y_test)
#training linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
linear_regression = LinearRegression()
pf(linear_regression)


#training decision tree
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
pf(dt)

#training polynomial
from sklearn.preprocessing import PolynomialFeatures
p2features = PolynomialFeatures(degree=2, include_bias=False)
p2_housing_data = p2features.fit_transform(final_housing_data)
p2_housing_test = p2features.fit_transform(housing_test_processed)

train_test(p2_housing_data, housing_X_lable, p2_housing_test, housing_y_test, linear_regression)

#cross_validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(linear_regression, final_housing_data, housing_X_lable, cv=10, scoring='neg_mean_squared_error')
score_rt = np.sqrt(-score)
print(score_rt)

#grid search
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR
svr = SVR(kernel='rbf')
pf(svr)