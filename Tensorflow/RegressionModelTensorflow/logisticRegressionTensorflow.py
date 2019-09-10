import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2
#noise = np.random.randn(1,2000)*0.1
wxb = np.matmul(w_real,x_data.T)+b_real
y_data_pre_noise  = sigmoid(wxb)
y_data = np.random.binomial(1,y_data_pre_noise)

print('real weight and biases')
print(w_real,b_real)
print('estimated weight and biases at step')

NUM_STEPS =51

import tensorflow as tf
g = tf.Graph()
with g.as_default():
    x=tf.placeholder(tf.float32,shape=(None,3),name = 'x')
    y_true = tf.placeholder(tf.float32,shape = None, name = 'y_true')
    with tf.name_scope('LogisticRegression') as scope:
        w=tf.Variable([[0,0,0]],dtype = tf.float32,name='weight')
        b = tf.Variable(0,dtype = tf.float32,name = 'bias')
        y_pred = tf.matmul(w,tf.transpose(x))+b
    with tf.name_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
        loss = tf.reduce_mean(loss)
    with tf.name_scope('train') as scope:
        learning_rate = 0.3
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train= optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()
with tf.Session(graph = g) as sess:
    sess.run(init_op)
    #creating event for tensorboard
    writer = tf.summary.FileWriter('./graphlogistic',sess.graph)

    for step in range(NUM_STEPS):
        sess.run(train,feed_dict = {x:x_data,y_true:y_data})
        if step%5==0:
            print(step,sess.run([w,b]))
            
