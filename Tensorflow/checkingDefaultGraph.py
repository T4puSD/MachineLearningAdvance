import numpy as np
import tensorflow as tf
g = tf.Graph()
with g.as_default():
  t1 = tf.constant(np.pi)
  print(t1.graph)
  print(t1.graph is tf.get_default_graph())
  print(t1)
