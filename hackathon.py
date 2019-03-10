import os
from utils import load_data
import numpy as np

path = os.path.join("datasets", 'breastcancer')

data = load_data(path, 'bc-data.csv')

# d = np.reshape(data['diagnosis'].values, (-1,1))
#
# from sklearn.preprocessing import OrdinalEncoder
# enc = OrdinalEncoder()
# t = [['M', 1], ['B', 0]]
# enc.fit(t)
# diag_enc = enc.transform(d)
#
# data['diagnosis_enc'] = diag_enc


y = data['diagnosis']

# X = data.drop(['diagnosis_enc', 'id', 'Unnamed: 32', 'diagnosis'], axis=1)
X = data.drop(['id', 'diagnosis'], axis=1)

X.columns

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

X = scale.fit_transform(X)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

X_train = X_train.astype(np.float32)

X_test = X_test.astype(np.float32)

for i in X_train:
    for j in i:
        if j<-100000:
            print (j)

for i in y_train:
    if type(i) != int :
        print(i)

import tensorflow as tf

feature_cols = [tf.feature_column.numeric_column("X", shape=[X_train.shape[1]])]

dnn = tf.estimator.DNNClassifier(hidden_units=[30, 10], feature_columns=feature_cols, model_dir='./models/hackathon-bc')

input_fn = tf.estimator.inputs.numpy_input_fn(x={"X": X_train}, y=y_train, num_epochs=40, batch_size=500, shuffle=True)


def train_model():
    dnn.train(input_fn=input_fn)


def test_model():
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"X": X_test}, y=y_test, shuffle=False)
    # eval_results = dnn.evaluate(input_fn=test_input_fn)
    # print(eval_results)
    pred = dnn.predict(input_fn=test_input_fn)
    i = 0
    r = []
    for v in pred:
        r.append(np.argmax(v.get('logits')))
        i = i + 1
        if (i == 100):
            break
    return r


#test_model()
#
# import csv
#
# ##############################
# def transform_instance(row):
#     cur_row = []
#     cur_row.append(row[0])
#     #cur_row.extend(nltk.word_tokenize(row[2].lower()))
#     return cur_row
#
#
# from random import shuffle
# from multiprocessing import Pool
# import multiprocessing
#
# def preprocess(input_file, output_file, keep=1):
#     all_rows = []
#     with open(input_file, 'r') as csvinfile:
#         csv_reader = csv.reader(csvinfile, delimiter=',')
#         for row in csv_reader:
#             all_rows.append(row)
#     shuffle(all_rows)
#     all_rows = all_rows[:int(keep * len(all_rows))]
#     pool = Pool(processes=multiprocessing.cpu_count())
#     transformed_rows = pool.map(transform_instance, all_rows)
#     pool.close()
#     pool.join()
#
#     with open(output_file, 'w') as csvoutfile:
#         csv_writer = csv.writer(csvoutfile, delimiter=' ', lineterminator='\n')
#         csv_writer.writerows(transformed_rows)
#
#
# preprocess('datasets/mbti_2.csv', 'fffff')


import tensorflow as tf
import numpy as np

n_inputs = 30
n_hidden1 = 30
n_hidden2 = 10
n_output = 2

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

epoch = 20
batch = 100

with tf.Session() as sess:
    init.run()
    for e in range(epoch):
        for X_batch, y_batch in utils.shuffle_batch(X_train, y_train, batch):
            sess.run(training, feed_dict={X: X_batch, y: y_batch})
            a = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
            print("accuracy=", a)






