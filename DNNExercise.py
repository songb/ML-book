import tensorflow as tf
import numpy as np

he_intialization = tf.variance_scaling_initializer()


def create_dnn(inputs, n_hidden_layers, n_neurons, name, activation=tf.nn.elu, initializer=he_intialization):
    with tf.variable_scope(name, default_name='dnn'):
        for i in range(n_hidden_layers):
            inputs = tf.layers.dense(inputs, n_neurons, activation=activation, kernel_initializer=initializer,
                                     name='hidden%d' % (i + 1))
        # last layer output
        return inputs


n_inputs = 28 * 28
n_outputs = 5

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')

outputs = create_dnn(X, 5, 100, 'my_dnn')

logits = tf.layers.dense(outputs, n_outputs, kernel_initializer=he_intialization, name='logits')
y_pred = tf.nn.softmax(logits, name='y_pred')

learning_rate = 0.01

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate)
training = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from utils import load_mnist_tf

X_train, X_test, y_train, y_test = load_mnist_tf()

X_train1 = X_train[y_train < 5]
y_train1 = y_train[y_train < 5]
X_test1 = X_test[y_test < 5]
y_test1 = y_test[y_test < 5]

epochs = 100
batch = 20

no_progress_limit = 20
no_progreee = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()

    for i in range(epochs):
        r = np.random.permutation(len(X_train1))

        for ind in np.array_split(r, len(X_train1) / batch):
            X_batch = X_train1[ind]
            y_batch = y_train1[ind]
            sess.run(training, feed_dict={X: X_batch, y: y_batch})

        loss_v, accuracy_v = sess.run([loss, accuracy], feed_dict={X: X_test1, y: y_test1})
        if (loss_v < best_loss):
            best_loss = loss_v
            saver.save(sess, './minst_0_4.ckpt')
            no_progreee = 0
        else:
            no_progreee = no_progreee + 1
            if (no_progreee > no_progress_limit):
                print('no progress')
                break

        print('epoch{}, loss:{:.6f}, accuracy:{:.2f}'.format(i, best_loss, accuracy_v * 100))
