# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
from io import open
import tensorflow as tf

# Common imports
import numpy as np
import os


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# To plot pretty figures

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "cnn"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


from sklearn.datasets import load_sample_image

china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
image = china[150:220, 130:250]
height, width, channels = image.shape
image_grayscale = image.mean(axis=2).astype(np.float32)
images = image_grayscale.reshape(1, height, width, 1)

reset_graph()
fmap = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
fmap[:, 3, 0, 0] = 1
fmap[3, :, 0, 1] = 1

X = tf.placeholder(tf.float32, shape=(None, height, width, 1))
feature_maps = tf.constant(fmap)
convolution = tf.nn.conv2d(X, feature_maps, strides=[1, 1, 1, 1], padding="SAME")

reset_graph()
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape

X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2, 2],
                        padding="SAME")

dataset = np.array([china, flower], dtype=np.float32)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    output = sess.run(conv, feed_dict={X: dataset})

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line

X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


def create_cnn(n_inputs, height, width, channels, n_outputs, conv_maps):
    reset_graph()
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
    X_shaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name='y')
    training = tf.placeholder(tf.bool, shape=(), name='training')

    layer_inputs = X_shaped
    i = 0
    for m in conv_maps:
        i = i + 1
        layer_inputs = tf.layers.conv2d(layer_inputs, name='layer%d' % i, activation=tf.nn.relu, **m)

    # pool
    pool = tf.nn.max_pool(layer_inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    pool_reshape = tf.reshape(pool, shape=[-1, 64 * 14 * 14])
    pool_drop = tf.layers.dropout(pool_reshape, training=training)

    dense = tf.layers.dense(pool_drop, 128, activation=tf.nn.relu, name='dense')
    dense_drop = tf.layers.dropout(dense, training=training)

    logits = tf.layers.dense(dense_drop, n_outputs, name='logits')
    y_prob = tf.nn.softmax(logits, name='y_prob')

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return {'X_ph': X, 'y_ph': y, 'training_ph': training, 'accuracy': accuracy, 'loss': loss,
            'training_op': training_op, 'y_prob': y_prob}


def fit_model(X, y, model_path, n_epochs=40, batch_size=50):
    from utils import shuffle_batch

    inputs = X.shape[1]
    conv1 = {'filters': 32, 'kernel_size': 3, 'strides': 1, 'padding': 'SAME'}
    conv2 = {'filters': 64, 'kernel_size': 3, 'strides': 1, 'padding': 'SAME'}
    convs = [conv1, conv2]
    cnn_model = create_cnn(inputs, 28, 28, 1, 10, convs)

    training_op = cnn_model['training_op']
    accuracy = cnn_model['accuracy']
    loss = cnn_model['loss']

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    best_loss = np.infty
    no_progress = 0

    with tf.Session() as sess:
        init.run()
        for i in range(n_epochs):
            c = 0
            for X_batch, y_batch in shuffle_batch(X, y, batch_size):
                if c == 0:
                    X_valid, y_valid = X_batch, y_batch
                    c = c + 1
                    continue
                c = c + 1
                sess.run(training_op, feed_dict={cnn_model['X_ph']: X_batch, cnn_model['y_ph']: y_batch,
                                                 cnn_model['training_ph']: False})
                print(c)
                if c % 500 == 0:
                    loss_val = loss.eval(feed_dict={cnn_model['X_ph']: X_valid, cnn_model['y_ph']: y_valid,
                                                    cnn_model['training_ph']: False})
                    acc = accuracy.eval(feed_dict={cnn_model['X_ph']: X_valid, cnn_model['y_ph']: y_valid,
                                                   cnn_model['training_ph']: False})
                    if loss_val < best_loss:
                        best_loss = loss_val
                        saver.save(sess, model_path)
                        no_progress = 0
                        print(acc)
                    else:
                        if no_progress > 2:
                            return
                        else:
                            no_progress = no_progress + 1


from utils import load_mnist_tf

X_train, X_test, y_train, y_test = load_mnist_tf()

fit_model(X_train, y_train, './models/cnn/mnist')

print('done')
