# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: XORV2.py
   create time: Sun 16 Jul 2017 07:00:20 AM EDT
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
   function: improved model can fit XOR
'''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

w1_1 = tf.Variable(tf.random_normal([2, 1]))
w1_2 = tf.Variable(tf.random_normal([2, 1]))
w2 = tf.Variable(tf.random_normal([2, 1]))

b1_1 = tf.constant(0.1, shape=[1])
b1_2 = tf.constant(0.1, shape=[1])
b2 = tf.constant(0.1, shape=[1])

h1 = tf.nn.relu(tf.matmul(x, w1_1) + b1_1)
h2 = tf.nn.relu(tf.matmul(x, w1_2) + b1_2)

hidden = tf.concat([h1, h2], 1)
out = tf.matmul(hidden, w2) + b2

loss = tf.reduce_mean(tf.square(out - y))

train = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        for j in range(4):
            sess.run(train, feed_dict={x: np.expand_dims(X[j], 0), y: np.expand_dims(Y[j], 0)})
        loss_ = sess.run(loss, feed_dict={x: X, y: Y})
        print("step: %d, loss: %.3f"%(i, loss_))
    print("X: %r"%X)
    print("pred: %r"%sess.run(out, feed_dict={x: X}))