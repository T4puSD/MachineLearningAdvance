#wap to impoliment an equation y = 2x^2+3x+5 in a tensorflow using placeholder

import tensorflow as tf
g = tf.Graph()

with g.as_default():
    tf_x = tf.placeholder(tf.int32,shape = [])
    y = 2*tf_x**2+3*tf_x+5

with tf.Session(graph = g) as sess:
    feed = {tf_x:2}
    print('output:',sess.run(y,feed_dict = feed))
    
