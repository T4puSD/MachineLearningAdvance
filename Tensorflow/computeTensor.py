import tensorflow as tf

g = tf.Graph()
with g.as_default():
  a = tf.constant(1)
  b = tf.constant(2)
  c = tf.constant(3)
  
  z = z = 2*(a-b)+c
  
with tf.Session(graph = g) as sess:
  print('Output: ',sess.run(z))
