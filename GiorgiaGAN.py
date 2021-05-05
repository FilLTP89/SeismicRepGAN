from __future__ import print_function, division

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Layer, Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout 
from tensorflow.keras.layers import Lambda, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding1D
from tensorflow.keras.layers import LeakyReLU, ReLU, Softmax
from tensorflow.keras.layers import UpSampling1D, Conv1D, Conv1DTranspose

from tensorflow.keras.optimizers import Adam, RMSprop
from numpy.linalg import norm
import MDOFload as mdof
import matplotlib.pyplot as plt
import visualkeras

import sys

import numpy as np


#class RandomWeightedAverage(_Merge):
    #"""Provides a (random) weighted average between real and generated signal samples"""
    #def _merge_function(self, inputs):
    #    alpha = K.random_uniform((32, 1, 1, 1))
    #    return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class RandomWeightedAverage(Layer):
    """Provides a (random) weighted average between real and generated signal samples"""
    def _merge_function(self, inputs, **kwargs):
        alpha = tf.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class RepGAN(Model):



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
        self.nXchannels = 2
        self.Xshape = (self.Xsize, self.nXchannels)

        self.latentZdim = 2048
        self.Zsize = self.Xsize//(self.stride**self.nlayers)
        self.nZchannels = self.latentZdim//self.Zsize

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

        assert self.nZchannels >= 1
        assert self.nZchannels >= self.stride**self.nlayers
        assert self.latentZdim >= self.Xsize//(self.stride**self.nlayers)

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

        realC = np.zeros((self.batchSize,self.latentCdim))
        rand_idx = np.random.randint(0,self.latentCdim,self.batchSize)
        realC[np.arange(self.batchSize),rand_idx]=1.0

        realS = np.random.normal(loc=0.0, scale=0.5, size=(self.batchSize,self.latentSdim))

        realN = np.random.normal(loc=0.0, scale=0.3, size=(self.batchSize,self.latentNdim))

        #realC = realZ[self.latentCidx] # C  
        #realS = realZ[self.latentSidx] # S 
        #realN = realZ[self.latentNidx] # N 

        # Fake
        (fakeC,fakeS,fakeN) = self.Fx(realX) # encoded z = Fx(X)

        fakeX = self.Gz(realZ) # fake X = Gz(Fx(X))
        

        # Discriminator determines validity of the real and fake X
        fakeXcritic = self.Dx(fakeX)
        realXcritic = self.Dx(realX)

        # Discriminator determines validity of the real and fake C
        fakeCcritic = self.Dc(fakeC)
        realCcritic = self.Dc(realC)

        # Discriminator determines validity of the real and fake S
        fakeScritic = self.Ds(fakeS)
        realScritic = self.Ds(realS)

        # Discriminator determines validity of the real and fake N
        fakeNcritic = self.Dn(fakeN)
        realNcritic = self.Dn(realN)

        # # Use Python partial to provide loss function with additional
        # # 'averaged_samples' argument
        # partial_gp_loss = partial(self.gradient_penalty_loss,
        #                   averaged_samples=interpolated_img)
        # partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names


        # # Construct weighted average between real and fake X
        # interpolatedX = RandomWeightedAverage()([realX, fakeX])
        # # Determine validity of weighted sample X
        # validity_interpolatedX = self.critic(interpolatedX)

        # # Construct weighted average between real and fake c
        # interpolatedC = RandomWeightedAverage()([realC, fakeC])
        # # Determine validity of weighted sample C
        # validity_interpolatedC = self.critic(interpolatedC)

        # # Construct weighted average between real and fake S
        # interpolatedS = RandomWeightedAverage()([realS, fakeS])
        # # Determine validity of weighted sample S
        # validity_interpolatedS = self.critic(interpolatedS)

        # # Construct weighted average between real and fake N
        # interpolatedN = RandomWeightedAverage()([realN, fakeN])
        # # Determine validity of weighted sample N
        # validity_interpolatedN = self.critic(interpolatedN)


        #self.adversarialCritic = Model(inputs=[real_img, z_disc],
        #                    outputs=[valid, fake, validity_interpolated])

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
        (fakeC,fakeS,fakeN) = self.Fx(realX) # encoded z = Fx(X)
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
        fakeZ = Concatenate(axis=1)([fakeC,fakeS,fakeN])
        recX  = self.Gz(fakeZ)
        (recC,recS,_)  = self.Fx(fakeX)

        # The Representative GAN model
        self.RegGANgenerative = RepGAN(input = [realX, realZ], 
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

        # # Compile the model
        # self.RepGANgenerative.compiledoubleopt(realXoptimizer = rmsprop_optimizer,fakeXoptimizer = rmsprop_optimizer,
        #                                        realCoptimizer = rmsprop_optimizer,fakeCoptimizer = rmsprop_optimizer,
        #                                        realSoptimizer = rmsprop_optimizer,fakeSoptimizer = rmsprop_optimizer,
        #                                        realNoptimizer = rmsprop_optimizer,fakeNoptimizer = rmsprop_optimizer,
        #                                        recXoptimizer = adam_optimizer,recCoptimizer = adam_optimizer,
        #                                        recSoptimizer = adam_optimizer,Advloss = wasserstein_loss, 
        #                                        recSloss = gaussian_nll)


        ## For the combined model we will only train the generator
        #self.discriminator.trainable = False

        ## The discriminator takes generated signals as input and determines validity
        #valid = self.discriminator(ths)

        ## The combined model  (stacked generator and discriminator)
        ## Trains the generator to fool the discriminator
        #self.combined = Model(z, valid)
        #self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def compiledoubleopt(self,realXoptimizer,fakeXoptimizer,realCoptimizer,
                         fakeCoptimizer,realSoptimizer,fakeSoptimizer,realNoptimizer,
                         fakeNoptimizer,recXoptimizer,recCoptimizer,recSoptimizer,
                         Advloss,recSloss):

        super(RepGANgenerative, self).compiledoubleopt()
        self.realXoptimizer = realXoptimizer
        self.fakeXoptimizer = fakeXoptimizer
        self.realCoptimizer = realCoptimizer
        self.fakeCoptimizer = fakeCoptimizer
        self.realSoptimizer = realSoptimizer
        self.fakeSoptimizer = fakeSoptimizer
        self.realNoptimizer = realNoptimizer
        self.fakeNoptimizer = fakeNoptimizer
        self.recXoptimizer = recXoptimizer
        self.recCoptimizer = recCoptimizer
        self.recSoptimizer = recSoptimizer

        self.Advloss = Advloss
        self.recSloss = recSloss
        self.recXloss = keras.losses.MeanSquareError()
        self.recCloss = keras.losses.BinaryCrossentropy()

    
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
        
        X = Input(shape=self.Xshape)

        # for n in range(self.nlayers):
        #     X = Conv1D((self.latentZdim/self.Zsize)*self.stride**(-self.nlayers+n),
        #         self.kernel,self.stride,padding="same",
        #         input_shape=self.Xshape,data_format="channels_last")(X)
        #     X = BatchNormalization(momentum=0.95)(inputs=X)
        #     X = LeakyReLU(alpha=0.1)(X)
        #     X = Dropout(rate=0.2)(X)
        # z = Flatten()(X)
        # z = Dense(self.latentZdim)(z)
        # z = BatchNormalization(momentum=0.95)(inputs=z)
        # z = LeakyReLU(alpha=0.1)(z)


        model = Sequential()
        for n in range(self.nlayers):
            model.add(Conv1D((self.latentZdim/self.Zsize)*self.stride**(-self.nlayers+n),
                self.kernel,self.stride,padding="same",
                input_shape=self.Xshape,data_format="channels_last"))
            model.add(BatchNormalization(momentum=0.95))
            model.add(LeakyReLU(alpha=0.1))
            model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(self.latentZdim))
        model.add(BatchNormalization(momentum=0.95))
        model.add(LeakyReLU(alpha=0.1))
        model.summary()
        
        z = model(X)
        
               

        # sampleS = Sequential()
        # sampleS.add(Lambda(lambda t: t,(self.latentSidx[0]),(self.latentSidx[-1])))

        # sampleC = Sequential()
        # sampleC.add(Lambda(lambda t: t,(self.latentCidx[0]),(self.latentCidx[-1])))

        # sampleN = Sequential()
        # sampleN.add(Lambda(lambda t: t,(self.latentNidx[0]),(self.latentNidx[-1])))


        # variable s 
        # MuS = Sequential()
        # LVS = Sequential()
        # MuS.add(Dense(self.latentSdim))
        # MuS.add(BatchNormalization(momentum=0.95))
        # LVS.add(Dense(self.latentSdim))
        # LVS.add(BatchNormalization(momentum=0.95))

        # mu = MuS(z)
        # lv = LVS(z)

        h = Dense(self.latentSdim)(z)
        Zmu = BatchNormalization(momentum=0.95)(inputs=h)

        h = Dense(self.latentSdim)(z)
        Zlv = BatchNormalization(momentum=0.95)(inputs=h)

        s = Sampling()([Zmu,Zlv])

        # variable c
        sampleC = Sequential()
        sampleC.add(Dense(self.latentCdim))
        sampleC.add(BatchNormalization(momentum=0.95))
        sampleC.add(Softmax())
        c = sampleC(z)
        
        # variable n
        sampleN = Sequential()
        sampleN.add(Dense(self.latentNdim))
        sampleN.add(BatchNormalization(momentum=0.95))
        n = sampleN(z)

        # concatenation of variables c, s and n
        # z = Concatenate()([c,s,n])


        return keras.Model(X,(c,s,n))

    def build_Gz(self):
        """
            Conv1D Gz structure
        """
        model = Sequential()          
        model.add(Dense(self.Zsize*self.nZchannels,input_dim=self.latentZdim,
            use_bias=False))
        model.add(Reshape((self.Zsize,self.nZchannels)))
        model.add(BatchNormalization(momentum=0.95))
        model.add(ReLU())
        for n in range(self.nlayers):
            model.add(Conv1DTranspose(self.latentZdim//self.stride**n,
                self.kernel,self.stride,padding="same"))
            model.add(BatchNormalization(momentum=0.95))
            model.add(ReLU())
        
        model.add(Conv1DTranspose(self.nXchannels,self.kernel,1,padding="same"))
        model.summary()

        z = Input(shape=(self.latentZdim,))
        X = model(z)

        return keras.Model(z,X)
        
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
        model.add(BatchNormalization(momentum=0.95))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(128,self.kernel,self.stride,padding="same"))
        model.add(BatchNormalization(momentum=0.95))
        model.add(LeakyReLU(alpha=0.0))
        model.add(Dropout(0.25))
        model.add(Conv1D(256,self.kernel,strides=1,padding="same"))
        model.add(BatchNormalization(momentum=0.95))
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
        # model.add(BatchNormalization(momentum=0.95))
        # model.add(Dense(1024,activation='LeakyReLU'))
        # model.add(BatchNormalization(momentum=0.95))
        # model.add(Dense(1,activation='sigmoid'))

        #model.summary()

        X = Input(shape=(self.Xshape))
        D_X = model(X)

        return keras.Model(X,D_X)


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

        #model.summary()

        c = Input(shape=(self.latentCdim,))
        D_c = model(c)

        return keras.Model(c,D_c)

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

        #model.summary()

        n = Input(shape=(self.latentNdim,))
        D_n = model(n)

        return keras.Model(n,D_n)

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

        #model.summary()

        s = Input(shape=(self.latentSdim,))
        D_s = model(s)

        return keras.Model(s,D_s)


    def train(self, epochs, batchSize=128, save_interval=50):

        # Load the dataset
        Xtrn,  Xvld, params_trn, params_vld, Ctrn, Cvld, Strn, Svld, Ntrn, Nvld = mdof.load_data()

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, Xtrn.shape[0], batchSize)
                realX = Xtrn[idx]
                
                # Sample noise as generator input
                realZ = np.random.normal(0, 1, (batchSize, self.latentZdim))

                # Train Discriminator
                LadvD = self.RepGANcritic.train_on_batch([realX, realZ],
                    [valid,fake,valid,fake,valid,fake,valid,fake])
                
                # Clip critic weights
                for l in self.Dx.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                    l.set_weights(weights)
                for l in self.Dc.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                    l.set_weights(weights)
                for l in self.Ds.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                    l.set_weights(weights)
                for l in self.Dn.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            LadvG = self.RepGANgenerative.train_on_batch([realX,realZ],
                    [valid,fake,valid,fake,valid,fake,valid,fake,
                     realX,realC,realS])


            # Plot the progress
            print ("{} [D loss: {}] [G loss: {}]" % (epoch, LadvD, LadvG))

            # If at save interval => save generated image samples
            # if epoch % sample_interval == 0:
            #     self.sampleX(epoch)

    # def sampleX(self, epoch):
    #     r, c = 5, 5
    #     noise = np.random.normal(0, 1, (r * c, self.latent_dim))
    #     gen_imgs = self.generator.predict(noise)

    #     # Rescale images 0 - 1
    #     gen_imgs = 0.5 * gen_imgs + 0.5

    #     fig, axs = plt.subplots(r, c)
    #     cnt = 0
    #     for i in range(r):
    #         for j in range(c):
    #             axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
    #             axs[i,j].axis('off')
    #             cnt += 1
    #     fig.savefig("images/mnist_%d.png" % epoch)
    #     plt.close()


    # def RepGAN_loss(true, pred):

    #     # Adversarial part of RepGAN loss
    #     Adv_c = tf.keras.losses.BinaryCrossentropy(c_true,c_pred)
    #     Adv_n = tf.keras.losses.BinaryCrossentropy(n_true,n_pred)
    #     Adv_s = tf.keras.losses.BinaryCrossentropy(s_true,s_pred)
    #     Adv_X = tf.keras.losses.BinaryCrossentropy(X_true,X_pred)
    #     Adv_loss = Adv_c + Adv_s + Adv_n + Adv_X

    #     # Reconstruction part of RepGAN loss
    #     Rec_X = -np.linalg.norm([X_true,X_pred])
    #     Rec_c = -tf.keras.losses.CategoricalCrossentropy(c_true,c_pred)
    #     Rec_s = gaussian_nll(s_true,s_pred)
    #     Rec_loss = Rec_loss = Rec_c + Rec_s + Rec_X

    #     # RepGAN loss
    #     RepGAN_loss = Adv_loss + Rec_loss

    #     return RepGAN_loss


if __name__ == '__main__':
    dcgan = GiorgiaGAN()
    dcgan.train(epochs=4000, batchSize=32, save_interval=50)

visualkeras.layered_view(GiorgiaGAN)
