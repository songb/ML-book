from DNNCreator import eval

from utils import load_mnist_tf

X_train, X_test, y_train, y_test = load_mnist_tf()

y_pred = eval(X_test, './models/cnn/mnist', training=False)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, y_pred)

print('final score=', score)
