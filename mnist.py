from shutil import copyfileobj
from six.moves import urllib


import os

def fetch_mnist(data_home="."):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)



from scipy.io import loadmat
fpath = os.path.join(".", 'mldata', 'mnist-original.mat')
mnist = loadmat(fpath)

X= mnist['data'].transpose()
y=mnist['label'].transpose()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

y_train_5 = (y_train==5)

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state=42)

from sklearn.model_selection import cross_val_predict
y_train_predict = cross_val_predict(sgd, X_train, y_train_5, cv=3)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5, y_train_predict)

from sklearn.metrics import precision_recall_curve
y_score = cross_val_predict(sgd, X_train, y_train_5,cv=3, method='decision_function')
precision,recall,threshold = precision_recall_curve(y_train_5, y_score)


# sgd.fit(X_train, y_train)
# print(sgd.predict([X_test[1]]))


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


some_digit = X[1]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")

# save_fig("some_digit_plot")
# plt.show()

from sklearn.preprocessing import StandardScaler
stdScaler = StandardScaler()
X_train_scaled =stdScaler.fit_transform(X_train)
y_score = cross_val_predict(sgd, X_train, y_train,cv=3)
cm = confusion_matrix(y_train, y_score)
row_sum=cm.sum(axis=1, keepdims=True)


# KNeighbors model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

kn = KNeighborsClassifier()
kn.fit(X_train, y_train) #this one never finishes
y_pred = kn.predict(X_test)

from sklearn.metrics import accuracy_score
s = accuracy_score(y_test, y_pred)




pass