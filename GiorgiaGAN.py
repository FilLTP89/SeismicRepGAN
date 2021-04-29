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

        """
            Optimizers
        """
        adam_optimizer = Adam(0.0002, 0.5)
        rmsprop_optimizer = RMSprop(lr=0.00005)

        """
            Build the discriminators
        """
        self.Dx = self.build_Dx()
        self.Dc = self.build_Dc()
        self.Ds = self.build_Ds()
        self.Dn = self.build_Dn()
        """
            Build Fx/Gz (generators)
        """
        self.Fx = self.build_Fx()
        self.Gz = self.build_Gz()

        #------------------------------------------------
        #           Construct Computational Graph
        #               for the Discriminator
        # Adversarial Losses: LadvX, Ladvc, Ladvs, Ladvn
        #------------------------------------------------

        # Freeze generators' layers while training critics
        self.Fx.trainable = False
        self.Gz.trainable = False


        # The generator takes the signal, encodes it and reconstructs it
        # from the encoding
        # Real
        realX = Input(shape=self.Xshape) # X data
        realZ = Input(shape=(self.latentZdim,))
        realC = realZ[self.latentCidx] # C  
        realS = realZ[self.latentSidx] # S 
        realN = realZ[self.latentNidx] # N 

        # Fake
        fakeZ = self.Fx(realX) # encoded z = Fx(X)
        fakeC = fakeZ[self.latentCidx] # C = Fx(X)|C 
        fakeS = fakeZ[self.latentSidx] # S = Fx(X)|S
        fakeN = fakeZ[self.latentNidx] # N = Fx(X)|N
        fakeX = self.Gz(realZ) # fake X = Gz(Fx(X))
        

        # Discriminator determines validity of the real and fake X
        (fakeXcritic, realXcritic) = self.Dx(fakeX), self.Dx(realX)

        # Discriminator determines validity of the real and fake C
        (fakeCcritic, realCcritic) = self.Dc(fakeC), self.Dc(realC)

        # Discriminator determines validity of the real and fake S
        (fakeScritic, realScritic) = self.Ds(fakeS), self.Ds(realS)

        # Discriminator determines validity of the real and fake N
        (fakeNcritic, realNcritic) = self.Dn(fakeN), self.Dn(realN)


        self.RepGANcritic = Model(inputs  = [realX,realZ],
            outputs = [realXcritic,fakeXcritic,realCcritic,fakeCcritic,
            realScritic,fakeScritic,realNcritic,fakeNcritic])        
        

        self.RepGANcritic.compile(loss=[self.wasserstein_loss,self.wasserstein_loss,
            self.wasserstein_loss,self.wasserstein_loss, self.wasserstein_loss,self.wasserstein_loss,
            self.wasserstein_loss,self.wasserstein_loss], optimizer=rmsprop_optimizer,
            loss_weights=[1, 1, 1, 1, 1, 1, 1, 1])

        #----------------------------------------
        #      Construct Computational Graph
        #               for Generator
        # Reconstruction Losses: RecX, Recc, Recs
        #----------------------------------------

        # Freeze critics' layers while training generators
        self.Fx.trainable = True
        self.Gz.trainable = True
        self.Dx.trainable = False
        self.Dc.trainable = False
        self.Ds.trainable = False
        self.Dn.trainable = False

        # Fake
        fakeZ = self.Fx(realX) # encoded z = Fx(X)
        fakeC = fakeZ[self.latentCidx] # C = Fx(X)|C 
        fakeS = fakeZ[self.latentSidx] # S = Fx(X)|S
        fakeN = fakeZ[self.latentNidx] # N = Fx(X)|N
        fakeX = self.Gz(realZ) # fake X = Gz(Fx(X))
        
        # Discriminator determines validity of the real and fake X
        fakeXcritic = self.Dx(fakeX)

        # Discriminator determines validity of the real and fake C
        fakeCcritic = self.Dc(fakeC)

        # Discriminator determines validity of the real and fake S
        fakeScritic = self.Ds(fakeS)

        # Discriminator determines validity of the real and fake N
        fakeNcritic = self.Dn(fakeN)

        # Reconstruction
        recX  = self.Gz(fakeZ)
        recZ  = self.Fx(fakeX)
        recC = recZ[self.latentCidx]
        recS = recZ[self.latentSidx]

        # The Representative GAN model
        self.RegGANgenerative = Model(input = [realX, realZ], 
            output = [realXcritic,fakeXcritic,realCcritic,fakeCcritic,
            realScritic,fakeScritic,realNcritic,fakeNcritic,
            recX,recC,recS])

        self.RepGANgenerative.compile(loss=[self.wasserstein_loss,self.wasserstein_loss,
                                            self.wasserstein_loss,self.wasserstein_loss,
                                            self.wasserstein_loss,self.wasserstein_loss,
                                            self.wasserstein_loss,self.wasserstein_loss,
                                            'mse','binary_crossentropy',self.gaussian_nll],
                                        optimizer=rmsprop_optimizer,
                                        loss_weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gaussian_nll(true, pred):
        """
         Gaussian negative loglikelihood loss function 
        """
        n_dims = int(int(pred.shape[1])/2)
        mu = pred[:, 0:n_dims]
        logsigma = pred[:, n_dims:]
        
        mse = -0.5*K.sum(K.square((true-mu)/K.exp(logsigma)),axis=1)
        sigma_trace = -K.sum(logsigma, axis=1)
        log2pi = -0.5*n_dims*np.log(2*np.pi)
        
        log_likelihood = mse+sigma_trace+log2pi

        return K.mean(-log_likelihood)

    def build_Fx(self):
        """
            Conv1D Fx structure
        """
        model = Sequential()
        for n in range(self.nlayers):
            model.add(Conv1D(self.latentZdim*self.stride**self.nlayers-n,
                self.kernel,self.stride,padding="same",activation="tanh"))
            model.add(LeakyReLU(alpha=0.5))
            model.add(BatchNormalization())
        
        #model.add(Conv1D(self.latentZdim,self.kernel,1,padding="same"))
        model.add(Flatten())
        model.add(Dense(self.latentZdim))
        model.summary()

        X = Input(shape=(self.Xsize,self.nXchannels))
        z = model(X)

        # variable s
        mu = Dense(self.latentSdim)(z[self.latentSidx])
        log_var = Dense(self.latentSdim)(z[self.latentSidx])

        mu = BatchNormalization(mu)
        log_var = BatchNormalization(log_var)

        s = merge([mu, log_var],
            mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
            output_shape=lambda p: p[0])

        # variable c
        c = Dense(self.latentCdim, activation='softmax')(z[self.latentCidx])
        c = BatchNormalization(c)
        
        # variable n
        n = Dense(self.latentNdim)(z[self.latentNidx])
        n = BatchNormalization(n)

        # concatenation of variables c, s and n
        z = Input(shape=(self.latentZdim,))
        z[self.latentCidx] = c
        z[self.latentSidx] = s
        z[self.latentNidx] = n

        return Model(X,z) 

    def build_Gz(self):
        """
            Conv1D Gz structure
        """
        model = Sequential()          
        model.add(Dense(self.Zsize*self.latentZdim,
            activation="tanh",
            input_dim=self.latentZdim,
            use_bias=False))
        model.add(Reshape((self.Zsize,self.latentZdim)))
        model.add(BatchNormalization())
        for n in range(self.Xsize//self.Zsize):
            model.add(Conv1DTranspose(self.latentZdim//self.stride**n,
                self.kernel,self.stride,padding="same",activation="relu"))
            model.add(BatchNormalization())
        
        model.add(Conv1DTranspose(1,self.kernel,1,padding="same"))

        model.summary()

        z = Input(shape=(self.latentZdim,))
        X = model(z)

        return Module(z,X)
        
    def build_Dx(self):
        """
            Conv1D discriminator structure
        """
        model = Sequential()
        model.add(Conv1D(32,self.kernel,self.stride,
            input_shape=self.Xshape,padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(64,self.kernel,self.stride,padding="same"))
        model.add(ZeroPadding1D(padding=((0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(128,self.kernel,self.stride,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.0))
        model.add(Dropout(0.25))
        model.add(Conv1D(256,self.kernel,strides=1,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1,activation='sigmoid'))
        # model.add(Conv1D(64,self.kernel,self.stride,
        #     input_shape=self.Xshape,padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv1D(128,self.kernel,self.stride,
        #     input_shape=self.Xshape,padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024,activation='LeakyReLU'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1,activation='sigmoid'))

        model.summary()

        X = Input(shape=(self.Xshape))
        D_X = model(X)

        return Model(X,D_X)


    def build_Dc(self):
        """
            Dense discriminator structure
        """
        model = Sequential()
        model.add(Dense(3000)) 
        model.add(LeakyReLU())
        model.add(Dense(3000)) 
        model.add(LeakyReLU())
        model.add(Dense(1))  

        model.summary()

        c = Input(shape=(self.latentCdim,))
        D_c = model(c)

        return Model(c,D_c)

    def build_Dn(self):
        """
            Dense discriminator structure
        """
        model = Sequential()
        model.add(Dense(3000)) 
        model.add(LeakyReLU())
        model.add(Dense(3000)) 
        model.add(LeakyReLU()) 
        model.add(Dense(1))  

        model.summary()

        n = Input(shape=(self.latentNdim,))
        D_n = model(n)

        return Model(n,D_n)

    def build_Ds(self):
        """
            Dense discriminator structure
        """
        model = Sequential()
        model.add(Dense(3000)) 
        model.add(LeakyReLU())
        model.add(Dense(3000)) 
        model.add(LeakyReLU())
        model.add(Dense(1))  

        model.summary()

        s = Input(shape=(self.latentSdim,))
        D_s = model(s)

        return Model(s,D_s)

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
