import tensorflow as tf
import numpy as np
g = tf.Graph()
with g.as_default():
    t1=tf.constant(np.pi)
    t2 = tf.constant([1,2,4,5])
    t3 = tf.constant([[1,2],[4,6]])
    
    x = np.arange(0,18)
    t4 = tf.reshape(x,shape = (2,3,3)) # 3 dimentional vector

    r1 = tf.rank(t1)
    r2 = tf.rank(t2)
    r3 = tf.rank(t3)
    r4 = tf.rank(t4)

    s1 = t1.get_shape()
    s2 = t2.get_shape()
    s3 = t3.get_shape()
    s4 = t4.get_shape()

    print(t1,t2,t3)
    print(s1,s2,s3)
    print(r1,r2,r3)

with tf.Session(graph = g) as sess:
    print('Rank :',r1.eval(),r2.eval(),r3.eval(),r4.eval())
    print('Val :',t1.eval(),t2.eval(),t3.eval(),t4.eval())
