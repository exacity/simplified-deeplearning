# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: dropout.py
   create time: Fri 29 Sep 2017 03:00:16 AM EDT
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
#mnist data
import tensorflow.examples.tutorials.mnist as mnist
mnist = mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
#ground truth
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float", [None,10])

#dropout
p = tf.placeholder(tf.float32)

#weight and bias
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#nn
h1 = tf.nn.dropout(x, keep_prob = p)
y = tf.nn.softmax(tf.matmul(h1,W) + b)

#loss and train
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, p : 0.95})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, p : 1.0}))
