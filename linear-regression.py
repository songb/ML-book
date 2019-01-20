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


from sklearn.linear_model import Ridge

X_total = 5*np.random.rand(500,1)
y_total = 9+3*X_total+np.random.rand(500,1)/2

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
    print("alpha=",alpha, " err=", err)

for i in [0, 0.1, 1, 10]:
    ridge = Ridge(i)
    verify(ridge, i, X_scaled, y_train, X_test, y_test)


pass