#wap to initialize a varibale with random number of size [5x3]
import numpy as np
import tensorflow as tf
g = tf.Graph()

with g.as_default():
    tf_x  = tf.Variable(np.random.rand(5,3))
    print(tf_x)

with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf_x))
