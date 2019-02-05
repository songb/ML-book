import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_reshape = x_train.astype(np.float32).reshape(-1, 28*28)/255.0
x_test_reshape = x_test.astype(np.float32).reshape(-1, 28*28)/255.0

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

feature_cols = [tf.feature_column.numeric_column("X", shape=[28*28])]

dnn = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10, feature_columns=feature_cols)

input_fn = tf.estimator.inputs.numpy_input_fn(x={"X":x_train_reshape}, y=y_train, num_epochs=40, batch_size=500, shuffle=True)

dnn.train(input_fn=input_fn)


test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": x_test_reshape}, y=y_test, shuffle=False)
eval_results = dnn.evaluate(input_fn=test_input_fn)


print(eval_results)
