# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: CNN.py
   create time: Fri 21 Jul 2017 07:52:12 AM EDT
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
#An implement of CNN
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist

batch_size = 512

data = mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)

GTX = tf.placeholder(tf.float32, [None, 28 * 28])
GTY = tf.placeholder(tf.float32, [None, 10])

def conv_layer(name, input, k_size, channel_in, channel_out, strides, padding = "VALID",):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [k_size, k_size, channel_in, channel_out],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", shape = [channel_out], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input, w, strides = strides, padding=padding)
        return tf.nn.relu(tf.nn.bias_add(conv, b))

def fc_layer(name, input, inD, outD, relu = True):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inD, outD], initializer = tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", shape=[outD], initializer=tf.constant_initializer(0.1))
        if relu == True:
            return tf.nn.relu(tf.nn.bias_add(tf.matmul(input, w), b))
        else:
            return tf.nn.bias_add(tf.matmul(input, w), b)

if __name__ == "__main__":
    GTX2 = tf.reshape(GTX, [-1, 28, 28, 1])
    #conv1
    conv1 = conv_layer("conv1", GTX2, 5, 1, 6, [1, 1, 1, 1], "SAME")
    #pooling2
    pool2 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
    #conv3
    conv3 = conv_layer("conv3", pool2, 5, 6, 16, [1, 1, 1, 1])
    #pooling4
    pool4 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

    shape = pool4.get_shape().as_list()
    fc_in = tf.reshape(pool4, [-1, shape[1] * shape[2] * shape[3]])

    fc5 = fc_layer("fc5", fc_in, shape[1] * shape[2] * shape[3], 120)
    fc6 = fc_layer("fc6", fc5, 120, 84)
    fc7 = fc_layer("fc7", fc6, 84, 10, False)
    prob = tf.nn.softmax(fc7)

    loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = GTY, logits = prob)))
    train = tf.train.AdamOptimizer(0.01).minimize(loss)

    correct_num = tf.equal(tf.arg_max(prob, 1), tf.arg_max(GTY, 1))
    acc = tf.reduce_mean(tf.cast(correct_num, dtype = tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000000):
            batch_X, batch_Y = data.train.next_batch(batch_size)
            sess.run(train, feed_dict = {GTX: batch_X, GTY: batch_Y})
            if i % 100 == 0:
                testacc = sess.run(acc, feed_dict = {GTX: data.test.images, GTY: data.test.labels})
                print("step %d: %.3f"%(i, testacc))
