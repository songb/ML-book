from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import numpy as np

he_init = tf.variance_scaling_initializer()


class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden_layers=5, n_neurons=100, optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.01, activation=tf.nn.elu, initializer=he_init,
                 batch_norm_momentum=None, dropout_rate=None, random_state=None):
        """Initialize the DNNClassifier by simply storing all the hyperparameters."""
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None

    def _build_dnn(self, inputs):
        for i in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs, rate=self.dropout_rate, training=self._training)
            inputs = tf.layers.dense(inputs, self.n_neurons, activation=self.activation,
                                     kernel_initializer=self.initializer, name='hidden%d' % (i + 1))
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum,
                                                       training=self._training)

        return inputs

    def _build_graph(self, n_inputs, n_outputs):
        if self.random_state:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
        y = tf.placeholder(tf.int32, shape=(None), name='y')

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None

        top_hidden_layer_output = self._build_dnn(X)
        logits = tf.layers.dense(top_hidden_layer_output, n_outputs, kernel_initializer=self.initializer, name='logits')
        y_prob = tf.nn.softmax(logits, name='y_prob')

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self._X = X
        self._y = y
        self._loss = loss
        self._y_prob = y_prob
        self._training_op = training_op
        self._accuracy = accuracy
        self._saver = saver
        self._init = init

    def fit(self, X, y, model_path, n_epochs=40, batch_size=20, X_valid=None, y_valid=None):
        if self._session:
            self._session.close()

        n_inputs = X.shape[1]
        n_outputs = len(np.unique(y))

        self._graph = tf.Graph()

        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        max_no_progress = 20
        no_progress = 0
        best_loss = np.infty

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for i in range(n_epochs):
                ind = np.random.permutation(len(X))
                for batch_ind in np.array_split(ind, len(X) // batch_size):
                    X_batch, y_batch = X[batch_ind], y[batch_ind]
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = False
                    sess.run(self._training_op, feed_dict=feed_dict)

                    if update_ops:
                        sess.run(update_ops, feed_dict=feed_dict)

                    if X_valid is None:
                        X_valid = X_batch
                    if y_valid is None:
                        y_valid = y_batch

                    loss, accuracy = sess.run([self._loss, self._accuracy],
                                              feed_dict={self._X: X_valid, self._y: y_valid})
                    print('loss=', loss, '  accuracy=', accuracy)

                    if loss < best_loss:
                        best_loss = loss
                        no_progress = 0
                        self._saver.save(sess, model_path)
                        print('save model')
                    else:
                        no_progress += 1
                        if no_progress > max_no_progress:
                            print('stop training')
                            return


# def eval(X_test, model_path):
#     tf.reset_default_graph()
#     graph = tf.Graph()
#     with graph.as_default():
#     # with self._session.as_default() as sess:
#         with tf.Session(graph=graph) as sess:
#             # self._saver.restore(sess, model_path)
#             tf.saved_model.loader.load(sess, [tag_constants.SERVING], './models/saved_dnn/12')
#             self._X = graph.get_tensor_by_name('X:0')
#             self._y_prob = graph.get_tensor_by_name('dense/BiasAdd:0')
#             y_prod = self._y_prob.eval(feed_dict={self._X: X_test})
#             y_pred = np.argmax(y_prod, axis=1)
#             return y_pred


def eval(X_test, model_path):
    import os
    sess = tf.Session()
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('{0}.meta'.format(model_path))
    path, _ = os.path.split(model_path)
    saver.restore(sess, tf.train.latest_checkpoint(path))

    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    feed_dict = {X: X_test}

    # Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("y_prob:0")

    y_prod = sess.run(op_to_restore, feed_dict)

    y_pred = np.argmax(y_prod, axis=1)
    return y_pred
