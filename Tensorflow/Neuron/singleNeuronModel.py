import tensorflow as tf
g = tf.Graph()
with g.as_default():
    w = tf.placeholder(tf.float32,shape = (None,10),name = 'w')
    b = tf.Variable(1)
    z = 
with tf.Session(graph  = g) as sess:
    sess.run(tf.global_variables_initializer())
    x = [2,3,4,5,1,3,5,7,1,4]
    sess.run(w,feed_dict={w:x})
    
