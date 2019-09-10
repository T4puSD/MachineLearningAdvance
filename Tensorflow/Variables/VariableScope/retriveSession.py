import tensorflow as tf
g1  = tf.Graph()

with g1.as_default():
    w1 = tf.get_variable('w1',shape=(5,2))
    new_saver = tf.train.Saver()
with tf.Session(graph = g1) as sess:
    new_saver = tf.train.import_meta_graph('variableScopeSession.meta')
    new_saver.restore(sess,'variableScopeSession')
    print('w1',sess.run(w1))
