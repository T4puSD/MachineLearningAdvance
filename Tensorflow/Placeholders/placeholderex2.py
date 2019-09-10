import tensorflow as tf
import numpy as np
g = tf.Graph()

with g.as_default():
  tf_a = tf.placeholder(tf.int32,shape = [],name = 'tf_a')
  tf_b = tf.placeholder(tf.int32,shape = [],name = 'tf_b')
  tf_c = tf.placeholder(tf.int32,shape = [],name = 'tf_c')
  z = 2*(tf_a-tf_b)+tf_c

with tf.Session(graph = g) as sess:
  feed = {tf_a:1,tf_b:10,tf_c:5}
  print('Output:',sess.run(z,feed_dict=feed))
