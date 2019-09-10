import tensorflow as tf
import numpy as np
g = tf.Graph()
with g.as_default():
  w = tf.Variable(np.array([[1,2,3,4,5],[6,7,8,9,10]]))
  print(w)
with tf.Session(graph = g) as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run(w))
