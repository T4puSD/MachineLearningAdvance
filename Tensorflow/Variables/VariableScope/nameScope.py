import tensorflow as tf

g = tf.Graph()

with g.as_default():
    with tf.name_scope('net_a'):
        with tf.name_scope('layer_1'):
            w1 = tf.Variable(tf.random_normal(shape =(5,2)),name ='w1')
            w2 = tf.Variable(tf.random_normal(shape = (5,2)),name = 'w2')
        with tf.name_scope('layer_2'):
            w3 = tf.Variable(tf.random_normal(shape =(5,2)),name = 'w3')
    with tf.variable_scope('net_b'):
        #with tf.variable_scope('layer_1'):
        w4 = tf.Variable(tf.random_normal(shape = (5,2)),name = 'w4')
    print(w1)
    print(w2)
    print(w3)
    print(w4)
    print(tf.get_variable('w4',shape = (5,2)))

with tf.Session(graph = g) as sess:
   #saver = tf.train.Saver() 
   sess.run(tf.global_variables_initializer())
   print(sess.run(w1),'\n',
         sess.run(w2),'\n',
         sess.run(w3),'\n',
         sess.run(w4))
   #saver.save(sess,'./variableScopeSession')
