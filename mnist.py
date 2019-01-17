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

pass