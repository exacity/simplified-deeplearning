import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

# Import MNIST data
mnist = input_data.read_data_sets("data/", one_hot=True)

# set training parms
lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

## RNN parms
n_input = 28
n_step = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_step, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
w = tf.Variable(tf.random_normal([n_hidden, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))
weights = {
    # (28,128)
    'in': tf.Variable(tf.random_normal([n_input,n_hidden])),
    # (128,10)
    'out': tf.Variable(tf.random_normal([n_hidden,n_classes]))
}
biases = {
    # (128)
    'in': tf.Variable(tf.constant(0.1,shape=[n_hidden,])),
    # (10,)
    'out': tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def rnn_model(X,weights,biases):
    # X ==> (128 batch*28 steps,28 inputs)
    X = tf.reshape(X,[-1,n_input])
    # X_in = (128 batch*28 steps,128 hidden)
    X_in = tf.matmul(X,weights['in']+biases['in'])
    # X_in ==> (128 batch,28 steps,128 hidden)
    X_in = tf.reshape(X_in,[-1,n_step,n_hidden])

    #use basic LSTM Cell
    lstm_cell = rnn.BasicLSTMCell(n_hidden,forget_bias=1.0,
                                  state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,
                                            time_major=False)
    results = tf.matmul(final_state[1],weights['out'] + biases['out'])
    return results

pred = rnn_model(x,weights,biases)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', cost)
summaries = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('logs/', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 1
    while batch_size * step < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape(batch_size, n_step, n_input)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc, loss = sess.run(
                [accuracy, cost], feed_dict={x: batch_x,
                                             y: batch_y})
            print("Iteration " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        if step % 100 == 0:
            s = sess.run(summaries, feed_dict={x: batch_x, y: batch_y})
            train_writer.add_summary(s, global_step=step)

        step += 1
    print("Optimization Finished!")

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_step, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_data, y: test_label}))