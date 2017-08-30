# coding: utf-8
import numpy 

import matplotlib
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.datasets import mnist


batch_size = 128
data_dim = 784
mid_dim = 512
sample_dim = 100
dropout_rate = 0.2


# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, data_dim)
X_test = X_test.reshape(10000, data_dim)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255


discriminator = Sequential()
discriminator.add(Dense(mid_dim, input_dim=data_dim, activation='tanh'))
discriminator.add(Dropout(dropout_rate))
discriminator.add(Dense(mid_dim / 2, activation='tanh'))
discriminator.add(Dropout(dropout_rate))
discriminator.add(Dense(mid_dim / 4, activation='tanh'))
discriminator.add(Dropout(dropout_rate))
discriminator.add(Dense(1, activation='sigmoid'))

 
generator = Sequential()
generator.add(Dense(mid_dim / 4, input_dim=sample_dim, activation='tanh'))
generator.add(Dropout(dropout_rate))
generator.add(Dense(mid_dim / 2, activation='tanh'))
generator.add(Dropout(dropout_rate))
generator.add(Dense(mid_dim, activation='tanh'))
generator.add(Dropout(dropout_rate))
generator.add(Dense(data_dim, activation='sigmoid'))
# generate fake sample
sample_fake = K.function([generator.input, K.learning_phase()], generator.output)



discriminator.trainable = False
generator.add(Dropout(dropout_rate))
generator.add(discriminator)


opt_g = Adam(lr=.0001)
generator.compile(loss='binary_crossentropy', optimizer=opt_g)

opt_d = Adam(lr=.002) #the learning rate of discriminator should be faster
discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy', optimizer=opt_d)


u_dist = numpy.random.uniform(-1, 1, (1000, sample_dim)).astype('float32')
gn_dist = sample_fake([u_dist, 0])

true_n_fake = numpy.vstack([X_train[0:1000], gn_dist])
y_batch = numpy.hstack([numpy.ones((1000, )),
                             numpy.zeros((1000, ))])


# pre train discriminator
loss1 = discriminator.fit(true_n_fake, y_batch, 
                          batch_size=batch_size, nb_epoch=1)

        

i = 0
# generate from these fixed noise
fixed_noise = numpy.random.uniform(-1, 1, (9, sample_dim)).astype('float32')

# start train
for j in range(5000):
        # train discriminator more often
        for k in range(10):
            i = (i + 1) % (60000//batch_size-1)
            n_dist = X_train[i*batch_size:(i+1)*batch_size]
            u_dist = numpy.random.uniform(-1, 1, (batch_size, sample_dim)).astype('float32')
            gn_dist = sample_fake([u_dist, 0])
            true_n_fake = numpy.vstack([n_dist, gn_dist])

            y_batch = numpy.hstack([numpy.ones((batch_size, )),
                                 numpy.zeros((batch_size, ))])
            loss1 = discriminator.train_on_batch(true_n_fake, y_batch)
                              
        # train generator once
        all_fake = numpy.ones((batch_size, )).astype('float32')
        u_dist = numpy.random.uniform(-1, 1, (batch_size, sample_dim)).astype('float32')
        loss0 = generator.train_on_batch(u_dist, all_fake)

        # print and save middle results as figure
        if j % 60 == 0:
            fixed_fake = sample_fake([fixed_noise, 0])
            plt.clf()
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow(fixed_fake[i].reshape((28,28)), cmap='gray')
                plt.axis('off')

            plt.show(block=False)
            plt.savefig('%05d.jpg'%j)
            plt.pause(1)
            print('generator loss: %.4f\t'%loss0, 
                  'discriminator loss: %.4f\t'%loss1)
