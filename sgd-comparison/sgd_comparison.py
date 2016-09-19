import gzip
import os
import sys
import timeit

import matplotlib.pyplot as plt
import six.moves.cPickle as pickle

from LogisticRegression import LogisticRegression
from MLP import MLP
from algorithms import *


def load_data(dataset):
    """Loads the dataset
    This code is downloaded from
    http://deeplearning.net/tutorial/code/logistic_sgd.py

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def smooth(x, window_len=10, window='hamming'):
    x = numpy.array(x)
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


def sgd_optimization_mnist(updates, classifier, n_epochs=10,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate various stochastic gradient descent optimization
    algorithms of a log-linear model

    This is demonstrated on MNIST.
    :type updates: list
    :param updates: updates for params which calculated by algorithms

    :type classifier: object
    :param classifier: model 

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    classifier = classifier(x)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gradients = [T.grad(cost, param) for param in classifier.params]
    parameters = classifier.params

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates(parameters, gradients),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')

    validation_frequency = n_train_batches
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    minibatch_avg_cost_list = []
    validation_losses_list = []
    while (epoch < n_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            minibatch_avg_cost_list.append(minibatch_avg_cost)
            # iteration number
            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                validation_losses_list.append(this_validation_loss)
                if best_validation_loss > this_validation_loss:
                    best_validation_loss = this_validation_loss

                print(
                    'epoch %i, validation error %f %%' %
                    (
                        epoch,
                        this_validation_loss * 100.
                    )
                )

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%'
        )
        % (best_validation_loss * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

    return minibatch_avg_cost_list, validation_losses_list


def test_model(classifier, shapes, fig_num=1):
    learning_rate = 0.1
    f = plt.figure(fig_num, figsize=(20, 10))
    ax1 = f.add_subplot(211)
    ax1.set_title('training cost')
    ax1.set_xlabel('training iterations')
    ax1.set_ylabel('cost')
    ax2 = f.add_subplot(212)
    ax2.set_title('validation error')
    ax2.set_xlabel('training epochs')
    ax2.set_ylabel('error rate')
    #ax2.set_xlim((0, 10))
    print('start test SGD...')
    sgd_updates = lambda parameters, gradients: sgd(parameters, gradients, learning_rate)
    sgd_train_costs, sgd_val_errors = sgd_optimization_mnist(sgd_updates, classifier)
    ax1.plot(smooth(sgd_train_costs), 'r', label="SGD")
    ax1.legend()
    ax2.plot(sgd_val_errors, 'ro-', label='SGD')
    ax2.legend()
    plt.pause(1)

    print('start test momentum...')
    momentum_updates = lambda parameters, gradients: momentum(parameters, gradients, shapes)
    momentum_train_costs, momentum_val_errors = sgd_optimization_mnist(momentum_updates, classifier)
    ax1.plot(smooth(momentum_train_costs), 'b', label='momentum')
    ax1.legend()
    ax2.plot(momentum_val_errors, 'bo-', label='momentum')
    ax2.legend()
    plt.pause(1)

    print('start test NAG...')
    NAG_updates = lambda parameters, gradients: NAG(parameters, gradients, shapes)
    NAG_train_costs, NAG_val_errors = sgd_optimization_mnist(NAG_updates, classifier)
    ax1.plot(smooth(NAG_train_costs), 'g', label='NAG')
    ax1.legend()
    ax2.plot(NAG_val_errors, 'go-', label='NAG')
    ax2.legend()
    plt.pause(1)

    print('start test AdaGrad...')
    AdaGrad_updates = lambda parameters, gradients: AdaGrad(parameters, gradients, shapes)
    AdaGrad_train_costs, AdaGrad_val_errors = sgd_optimization_mnist(AdaGrad_updates, classifier)
    ax1.plot(smooth(AdaGrad_train_costs), 'y', label='AdaGrad')
    ax1.legend()
    ax2.plot(AdaGrad_val_errors, 'yo-', label='AdaGrad')
    ax2.legend()
    plt.pause(1)

    print('start test RMSProp...')
    RMSProp_updates = lambda parameters, gradients: RMSProp(parameters, gradients, shapes)
    RMSProp_train_costs, RMSProp_val_errors = sgd_optimization_mnist(RMSProp_updates, classifier)
    ax1.plot(smooth(RMSProp_train_costs), 'c', label='RMSProp')
    ax1.legend()
    ax2.plot(RMSProp_val_errors, 'co-', label='RMSProp')
    ax2.legend()
    plt.pause(1)

    print('start test AdaDelta...')
    AdaDelta_updates = lambda parameters, gradients: AdaDelta(parameters, gradients, shapes)
    AdaDelta_train_costs, AdaDelta_val_errors = sgd_optimization_mnist(AdaDelta_updates, classifier)
    ax1.plot(smooth(AdaDelta_train_costs), 'm', label='AdaDelta')
    ax1.legend()
    ax2.plot(AdaDelta_val_errors, 'mo-', label='AdaDelta')
    ax2.legend()
    plt.pause(1)

    print('start test Adam...')
    Adam_updates = lambda parameters, gradients: Adam(parameters, gradients, shapes)
    Adam_train_costs, Adam_val_errors = sgd_optimization_mnist(Adam_updates, classifier)
    ax1.plot(smooth(Adam_train_costs), 'k', label='Adam')
    ax1.legend()
    ax2.plot(Adam_val_errors, 'ko-', label='Adam')
    ax2.legend()
    plt.pause(1)

    print('start test Adamax...')
    Adamax_updates = lambda parameters, gradients: Adamax(parameters, gradients, shapes)
    Adamax_train_costs, Adamax_val_errors = sgd_optimization_mnist(Adamax_updates, classifier)
    ax1.plot(smooth(Adamax_train_costs), color='#895CCC', label='Adamax')
    ax1.legend()
    ax2.plot(Adamax_val_errors, 'o-', color='#895CCC', label='Adamax')
    ax2.legend()
    plt.pause(1)

    plt.show(block=False)


if __name__ == '__main__':
    image_size = 28 * 28
    classes = 10
    # test on logistic regression
    shapes = [(image_size, classes), (classes,)]
    # construct the logistic regression class
    classifier = lambda x: LogisticRegression(input=x, n_in=image_size, n_out=classes)
    print('start test on logistic regression......')
    test_model(classifier, shapes, fig_num=1)

    # test on mlp
    n_hidden = 500
    shapes = [(image_size, n_hidden), (n_hidden,), (n_hidden, classes),
              (classes,)]
    rng = numpy.random.RandomState(1234)
    # construct the mlp
    classifier = lambda x: MLP(
        rng=rng,
        input=x,
        n_in=image_size,
        n_hidden=n_hidden,
        n_out=classes
    )
    print()
    print()
    print('start test on logistic regression......')
    test_model(classifier, shapes, fig_num=2)
    plt.show()
