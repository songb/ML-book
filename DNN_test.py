import tensorflow as tf
import numpy as np
from DNNCreator import eval

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:500], X_train[500:3000]
y_valid, y_train = y_train[:500], y_train[500:3000]

X_train1 = X_train[y_train < 5]
y_train1 = y_train[y_train < 5]
X_valid1 = X_valid[y_valid < 5]
y_valid1 = y_valid[y_valid < 5]
X_test1 = X_test[y_test < 5]
y_test1 = y_test[y_test < 5]


from DNNCreator import DNNClassifier
model_path='./models/dnncreator_test'

dnn_clf = DNNClassifier(random_state=42)
# dnn_clf.fit(X_train1, y_train1, model_path,n_epochs=40, X_valid=X_valid1, y_valid=y_valid1)

from sklearn.metrics import accuracy_score

y_pred = eval(X_test1, model_path)
score = accuracy_score(y_test1, y_pred)

print ('final test score=', score)
