import tensorflow as tf

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./models/dnn.ckpt")
    pred = sess.run