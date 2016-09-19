import numpy
import theano
import theano.tensor as T

# http://cs229.stanford.edu/proj2015/054_report.pdf
# http://sebastianruder.com/optimizing-gradient-descent/


def sgd(parameters, gradients, eta=1.0):
    """
    Basic Mini-batch Stochastic Gradient Descent.
    :param parameters:
    :param gradients:
    :param eta: learning rate
    :return: updates of parameters
    """
    updates = [(parameters[i], parameters[i] - eta * gradients[i])
               for i in range(len(parameters))]
    return updates


def momentum(parameters, gradients, shapes, eta=.05, gamma=.9):
    """
    Momentum-based Stochastic Gradient Descent
    :param parameters:
    :param gradients:
    :param shapes: shape of parameters
    :param eta: learning rate
    :param gamma: decay factor
    :return: updates of parameters
    """
    para_num = len(parameters)
    v = []
    t = theano.shared(numpy.float32(0.), 't')
    for shape in shapes:
        v.append(theano.shared(
            value=numpy.zeros(
                shape,
                dtype=theano.config.floatX
            ),
            name='v' + str(shape),
            borrow=True))

    next_v = [gamma * v[i] + eta * gradients[i] for i in range(para_num)]
    updates = [(v[i], next_v[i]) for i in range(para_num)]
    updates.extend([(parameters[i], parameters[i] - next_v[i])
                    for i in range(para_num)])
    updates.extend([(t, t + 1)])
    return updates


def NAG(parameters, gradients, shapes, eta=.05, gamma=.9):
    """
    Nesterov accelerated gradient
    :param parameters:
    :param gradients:
    :param shapes: shape of parameters
    :param eta: learning rate
    :param gamma: decay factor
    :return: updates of parameters
    """
    para_num = len(parameters)
    v = []
    t = theano.shared(numpy.float32(0.), 't')
    for shape in shapes:
        v.append(theano.shared(
            value=numpy.zeros(
                shape,
                dtype=theano.config.floatX
            ),
            name='v' + str(shape),
            borrow=True))

    next_v = [gamma * v[i] + eta * gradients[i] for i in range(para_num)]
    updates = [(v[i], next_v[i]) for i in range(para_num)]
    updates.extend([(parameters[i], parameters[i] - gamma * next_v[i] - eta * gradients[i])
                    for i in range(para_num)])
    updates.extend([(t, t + 1)])
    return updates


def AdaGrad(parameters, gradients, shapes, eta=0.01, epsilon=1e-8):
    """
    Adaptive subGradient
    :param parameters:
    :param gradients:
    :param shapes: shape of parameters
    :param eta: learning rate
    :param epsilon:
    :return: updates of parameters
    """
    para_num = len(parameters)
    G = []
    for shape in shapes:
        G.append(theano.shared(
            value=numpy.zeros(
                shape,
                dtype=theano.config.floatX
            ),
            name='G' + str(shape),
            borrow=True))
    next_G = [G[i] + T.sqr(gradients[i]) for i in range(para_num)]
    updates = [(G[i], next_G[i]) for i in range(para_num)]
    updates.extend([(parameters[i], parameters[i] - eta * gradients[i] / T.sqrt(next_G[i] + epsilon)) for i in
                    range(para_num)])
    return updates


def RMSProp(parameters, gradients, shapes, eta=0.001, gamma=0.9, epsilon=1e-8):
    """
    RMSProp: Divide the gradient by a running average of its recent magnitude
    :param parameters:
    :param gradients:
    :param shapes: shape of parameters
    :param eta: learning rate
    :param gamma: decay factor
    :param epsilon:
    :return: updates of parameters
    """
    para_num = len(parameters)
    G = []
    for shape in shapes:
        G.append(theano.shared(
            value=numpy.zeros(
                shape,
                dtype=theano.config.floatX
            ),
            name='G' + str(shape),
            borrow=True))
    next_G = [gamma * G[i] + (1 - gamma) * T.sqr(gradients[i]) for i in range(para_num)]
    updates = [(G[i], next_G[i]) for i in range(para_num)]
    updates.extend([(parameters[i], parameters[i] - eta * gradients[i] / T.sqrt(next_G[i] + epsilon)) for i in
                    range(para_num)])
    return updates


def AdaDelta(parameters, gradients, shapes, gamma=0.95, epsilon=1e-6):
    """
    AdaDelta: An Adaptive Learning Rate
    :param parameters:
    :param gradients:
    :param shapes: shape of parameters
    :param gamma: decay factor
    :param epsilon:
    :return: updates of parameters
    """
    para_num = len(parameters)
    G = []
    dx = []
    for shape in shapes:
        G.append(theano.shared(
            value=numpy.zeros(
                shape,
                dtype=theano.config.floatX
            ),
            name='G' + str(shape),
            borrow=True))
    for shape in shapes:
        dx.append(theano.shared(
            value=numpy.zeros(
                shape,
                dtype=theano.config.floatX
            ),
            name='dx' + str(shape),
            borrow=True))
    next_G = [gamma * G[i] + (1 - gamma) * T.sqr(gradients[i]) for i in range(para_num)]
    updates = [(G[i], next_G[i]) for i in range(para_num)]
    next_dx = [T.sqrt(dx[i] + epsilon) / T.sqrt(next_G[i] + epsilon) for i in range(para_num)]
    updates.extend([(parameters[i], parameters[i] - next_dx[i] * gradients[i]) for i in
                    range(para_num)])
    updates.extend([(dx[i], gamma * dx[i] + (1 - gamma) * T.sqr(next_dx[i] * gradients[i])) for i in range(para_num)])
    return updates


def Adam(parameters, gradients, shapes, eta=0.002, gamma=0.999, beta=0.9, epsilon=1e-8):
    """
    Adam: adaptive estimates of lower-order moments
    :param parameters:
    :param gradients:
    :param shapes: shape of parameters
    :param eta: learning rate
    :param beta: mean decay factor 
    :param gamma: variance decay factor
    :param epsilon:
    :return: updates of parameters
    """
    para_num = len(parameters)
    m = []
    t = theano.shared(numpy.float32(1.), 't')
    for shape in shapes:
        m.append(theano.shared(
            value=numpy.zeros(
                shape,
                dtype=theano.config.floatX
            ),
            name='m' + str(shape),
            borrow=True))

    G = []
    for shape in shapes:
        G.append(theano.shared(
            value=numpy.zeros(
                shape,
                dtype=theano.config.floatX
            ),
            name='G' + str(shape),
            borrow=True))

    next_m = [beta * m[i] + (1 - beta) * gradients[i] for i in range(para_num)]
    updates = [(m[i], next_m[i]) for i in range(para_num)]
    next_G = [gamma * G[i] + (1 - gamma) * T.sqr(gradients[i]) for i in range(para_num)]
    updates.extend([(G[i], next_G[i]) for i in range(para_num)])
    updates.extend([(parameters[i], parameters[i] - eta * T.sqrt(1 - gamma ** t) / (1 - beta ** t) * next_m[i] / T.sqrt(
        next_G[i] + epsilon))
                    for i in range(para_num)])
    updates.extend([(t, t + 1)])
    return updates


def Adamax(parameters, gradients, shapes, eta=0.004, gamma=0.999, beta=0.9, epsilon=1e-8):
    """
    Adamax: adaptive estimates of lower-order moments
    :param parameters:
    :param gradients:
    :param shapes: shape of parameters
    :param eta: learning rate
    :param beta: mean decay factor 
    :param gamma: variance decay factor
    :param epsilon: 
    :return: updates of parameters
    """
    para_num = len(parameters)
    m = []
    t = theano.shared(numpy.float32(1.), 't')
    for shape in shapes:
        m.append(theano.shared(
            value=numpy.zeros(
                shape,
                dtype=theano.config.floatX
            ),
            name='m' + str(shape),
            borrow=True))

    G = []
    for shape in shapes:
        G.append(theano.shared(
            value=numpy.zeros(
                shape,
                dtype=theano.config.floatX
            ),
            name='G' + str(shape),
            borrow=True))

    next_m = [beta * m[i] + (1 - beta) * gradients[i] for i in range(para_num)]
    updates = [(m[i], next_m[i]) for i in range(para_num)]
    next_G = [T.maximum(gamma * G[i], abs(gradients[i])) for i in range(para_num)]
    updates.extend([(G[i], next_G[i]) for i in range(para_num)])
    updates.extend([(parameters[i], parameters[i] - eta / (1 - beta ** t) * next_m[i] / (next_G[i] + epsilon))
                    for i in range(para_num)])
    updates.extend([(t, t + 1)])
    return updates
