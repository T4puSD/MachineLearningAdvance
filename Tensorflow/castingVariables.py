import tensorflow as tf
import numpy as np
g=tf.Graph()
a = tf.constant(4.3)
print(a)
with g.as_default():
  a=tf.constant(5,name='a')
  b=tf.constant(2,name='b')
  c=tf.constant(3,name='c')
  d=tf.multiply(a,b)
  e = tf.add(c,b)
  f = tf.subtract(d,e)
  print(a)
  
  print(b.dtype)
  b = tf.cast(b,tf.float32)
  print(b.dtype)
  
  with tf.Session(graph = g)as sess:
    print(sess.run(f))
    print(sess.run(b))