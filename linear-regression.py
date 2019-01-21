import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

from utils import save_fig
import matplotlib.pyplot as plt

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
save_fig("generated_data_plot")


from sklearn.linear_model import SGDRegressor


from sklearn.linear_model import Ridge, Lasso

X_total = 5*np.random.rand(5000,1)
y_total = 9+3*X_total+np.random.rand(5000,1)/2

from sklearn.model_selection import train_test_split
X_train_raw, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_raw)


from sklearn.metrics import mean_squared_error
def verify(model, alpha, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    err = np.sqrt(mean_squared_error(y_test, y_predict))
    print("alpha=",alpha, " err=", err, " coef=", model.coef_, " intercept= ", model.intercept_)

for i in [0, 0.1, 1, 10]:
    ridge = Ridge(i)
    verify(ridge, i, X_scaled, y_train, X_test, y_test)

for i in [0, 0.1, 1, 10]:
    lasso = Lasso(i)
    verify(lasso, i, X_scaled, y_train, X_test, y_test)


from sklearn.linear_model import LogisticRegression


from sklearn import datasets
(X,y)=datasets.load_iris(True)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

pipeline = Pipeline([("scaler", StandardScaler()), ("svc", LinearSVC(loss='hinge'))])
pipeline.fit(X_train, y_train)

y_predict=pipeline.predict(X_test)

err=np.sqrt(mean_squared_error(y_test, y_predict))
print(err)

pass