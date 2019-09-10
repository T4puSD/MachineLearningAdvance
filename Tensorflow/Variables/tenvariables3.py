import tensorflow as tf

g = tf.Graph()

with g.as_default():
    w1 = tf.Variable(1,name = 'w1')
    w2 = tf.Variable(2,name = 'w2')
    w3 = tf.Variable(3,name = 'w3')

    init_op = tf.global_variables_initializer()

with tf.Session(graph = g) as sess:
    sess.run(init_op)
    print(sess.run(w1),sess.run(w2),sess.run(w3))
