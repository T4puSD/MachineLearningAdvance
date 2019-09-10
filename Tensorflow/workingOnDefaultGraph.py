import tensorflow as tf

c = tf.linspace(0.0,4.5,5)
d = tf.zeros(shape = (2,5))
e = tf.zeros(shape= (2,5),dtype = tf.int64)
f = tf.ones(shape=(2,5))
g = tf.random_normal(shape = (2,2))
h = tf.random_uniform(shape = (2,5))

a = tf.constant([1,2,3])
print(tf.shape(a))
b = tf.expand_dims(a,1)
print(tf.shape(a))

sess = tf.InteractiveSession()
print("c = {}".format(c.eval()))
print("d = {}".format(d.eval()))
print("e = {}".format(e.eval()))
print("f = {}".format(f.eval()))
print("g = {}".format(g.eval()))

print("h = {}".format(a.eval()))
print("h1 = {}".format(b.eval()))

print(sess.run(c))

sess.close()
