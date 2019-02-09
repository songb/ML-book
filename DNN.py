import tensorflow as tf
import numpy as np

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_output = 10

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    output = tf.layers.dense(hidden2, n_output, name="output")

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output, name="xentropy")
loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training = optimizer.minimize(loss)

correct = tf.nn.in_top_k(output, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

import utils
(X_train, X_test, y_train, y_test) = utils.load_mnist_tf()

def train_model():

    epoch = 20
    batch = 100

    with tf.Session() as sess:
        init.run()
        for e in range(epoch):
            for X_batch, y_batch in utils.shuffle_batch(X_train, y_train, batch):
                sess.run(training, feed_dict={X: X_batch, y: y_batch})
                a = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
                print("accuracy=", a)

        saver.save(sess, "./models/dnn.ckpt")


def run_test():
    with tf.Session() as sess:
        saver.restore(sess, "./models/dnn.ckpt")
        pred_array = output.eval(feed_dict={X:X_test})
        pred = np.argmax(pred_array, axis=1)
        return pred

pred = run_test()
print("pred ", pred[:30])
print("   y ", y_test[:30])