# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: RNN.py
   create time: Sun 31 Oct 2017 07:52:12 AM EDT
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
#An implement of RNN
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist

batch_size = 512
n_input = 28
n_step = 28
n_hidden = 64

data = mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)

#grondtruth image and label
GTX = tf.placeholder(tf.float32, [None, n_step, n_input])
GTY = tf.placeholder(tf.float32, [None, 10])

#input to hidden layer
w1 = tf.get_variable("w1", shape = [n_input, n_hidden], initializer=tf.truncated_normal_initializer(stddev=0.1))
b1 = tf.get_variable("b1", shape = [n_hidden], initializer=tf.constant_initializer(0.1))

#hidden layer to class
w2 = tf.get_variable("w2", shape = [n_hidden, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
b2 = tf.get_variable("b2", shape = [10], initializer=tf.constant_initializer(0.1))

if __name__ == "__main__":
    # step, batchsize, input
    X = tf.transpose(GTX, [1, 0, 2])
    # step * batchsize, input
    X = tf.reshape(X, [-1, n_input])
    # step * batchsize, hidden
    X = tf.matmul(X, w1) + b1
    # step * (batchsize, hidden)
    X = tf.split(X, n_step, 0)

    # build RNN
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    h, states = tf.nn.static_rnn(rnn_cell, X, dtype=tf.float32)

    # hidden layer to class
    prob = tf.matmul(h[-1], w2) + b2

    # loss
    loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = GTY, logits = prob)))
    train = tf.train.AdamOptimizer(0.01).minimize(loss)

    # evaluate
    correct_num = tf.equal(tf.arg_max(prob, 1), tf.arg_max(GTY, 1))
    acc = tf.reduce_mean(tf.cast(correct_num, dtype = tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            batch_X, batch_Y = data.train.next_batch(batch_size)
            batch_X = batch_X.reshape((batch_size, 28, 28))
            sess.run(train, feed_dict = {GTX: batch_X, GTY: batch_Y})
            if i % 100 == 0:
                testacc = sess.run(acc, feed_dict = {GTX: data.test.images.reshape((-1, 28, 28)), GTY: data.test.labels})
                print("step %d: %.3f"%(i, testacc))
