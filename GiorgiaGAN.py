from __future__ import print_function, division

#from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from scipy.stats import entropy
from numpy.linalg import norm
import MDOFload as mdof

import matplotlib.pyplot as plt

import sys

import numpy as np


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated signal samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class GiorgiaGAN():
    """
        Flexible implementation of GAN based auto-Fx
    """
    def __init__(self):

        """
            Setup
        """
        self.stride = 2
        self.kernel = 3
        self.nlayers = 5
        self.Xsize = 1024
        self.Zsize = self.Xsize//(self.stride**self.nlayers)
        self.nXchannels = 2
        self.Xshape = (self.Xsize, self.nXchannels)
        self.latentZdim = 256
        self.latentCidx = list(range(5))
        self.latentSidx = list(range(5,7))
        self.latentNidx = list(range(7,self.latentZdim))
        self.latentCdim = len(self.latentCidx)
        self.latentSdim = len(self.latentSidx)
        self.latentNdim = len(self.latentNidx)
        self.batchSize = 128
        self.n_critic = 5
        self.clipValue = 0.01
        self.data_root_ID = '/gpfs/workdir/invsem07/damaged_1_8P' 
        self.ID_string = 'U'
        self.ID_pb_string = 'BC'
        self.case = 'train_model'

        # assert self.latentZdim >= self.stride**(self.Xsize//self.Zsize)

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        ths = model(noise)

        return Model(noise, ths)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv1D(32, kernel_size=3, strides=2, input_shape=self.ths_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding1D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        ths = Input(shape=self.ths_shape)
        validity = model(ths)

        return Model(ths, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        #X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of signals
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            thss = X_train[idx]

            # Sample noise and generate a batch of new signals
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_thss = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(thss, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_thss, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake signals as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated signal samples
            if epoch % save_interval == 0:
                self.save_thss(epoch)

    def save_thss(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_thss = self.generator.predict(noise)

        # Rescale signals 0 - 1
        gen_thss = 0.5 * gen_thss + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_thss[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("signals/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = GiorgiaGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
