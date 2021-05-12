# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
__author__ = "Filippo Gatti"
__copyright__ = "Copyright 2020, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti-centralesupelec.fr"
__status__ = "Beta"


import sys
import argparse
from os.path import join as opj
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Layer
from tensorflow.keras.layers import Lambda, Concatenate,concatenate, Activation
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding1D
from tensorflow.keras.layers import LeakyReLU, ReLU, Softmax
from tensorflow.keras.layers import UpSampling1D, Conv1D, Conv1DTranspose
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.constraints import Constraint, min_max_norm
from numpy.linalg import norm
import MDOFload as mdof
import matplotlib.pyplot as plt
import visualkeras
import keras.backend as K
# tf.compat.v1.disable_eager_execution()
AdvDLoss_tracker = keras.metrics.Mean(name="loss")
AdvGLoss_tracker = keras.metrics.Mean(name="loss")


def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=1000,help='Number of epochs')
    parser.add_argument("--kernel",type=int,default=3,help='CNN kernel size')
    parser.add_argument("--stride",type=int,default=2,help='CNN stride')
    parser.add_argument("--nCnnLayers",type=int,default=5,help='Number of CNN layers per Coupling Layer')
    parser.add_argument("--Xsize",type=int,default=1024,help='Data space size')
    parser.add_argument("--nXchannels",type=int,default=2,help="Number of data channels")
    parser.add_argument("--latentZdim",type=int,default=2048,help="Latent space dimension")
    parser.add_argument("--batchSize",type=int,default=24,help='input batch size')
    parser.add_argument("--nCritic",type=int,default=5,help='number of discriminator training steps')
    parser.add_argument("--clipValue",type=float,default=0.01,help='clip weight for WGAN')
    parser.add_argument("--dataroot",type=str,default="/gpfs/workdir/invsem07/damaged_1_8P",help="Data root folder")
    parser.add_argument("--which_channel_1",type=int,default=21,help="Channel 1")
    parser.add_argument("--which_channel_2",type=int,default=39,help="Channel 2")
    parser.add_argument("--n_params",type=str,default=2,help="Number of parameters")
    parser.add_argument("--seq_len_input",type=int,default=1000,help="Sequence length input")
    parser.add_argument("--seq_len",type=int,default=1024,help="Sequence length")
    parser.add_argument("--seq_len_start",type=int,default=0,help="Sequence length start")
    parser.add_argument("--seq_sampling",type=int,default=1,help="Sequence Sampling")
    parser.add_argument("--case",type=str,default="train_model",help="case")
    parser.add_argument("--id",type=str,default="U",help="case id")
    parser.add_argument("--pb",type=str,default="BC",help="case pb")

       
    options = parser.parse_args().__dict__
    

    options['Xshape'] = (options['Xsize'], options['nXchannels'])
    options['Zsize']  = options['Xsize']//(options['stride']**options['nCnnLayers'])
    options['latentCidx'] = list(range(5))
    options['latentSidx'] = list(range(5,7))
    options['latentNidx'] = list(range(7,options['latentZdim']))
    options['latentCdim'] = len(options['latentCidx'])
    options['latentSdim'] = len(options['latentSidx'])
    options['latentNdim'] = len(options['latentNidx'])
    options['nZchannels'] = options['latentZdim']//options['Zsize']
    options['nCchannels'] = options['latentCdim']//options['Zsize']
    options['nSchannels'] = options['latentSdim']//options['Zsize']
    options['nNchannels'] = options['latentNdim']//options['Zsize']

    return options

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


def WassersteinDiscriminatorLoss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def WassersteinGeneratorLoss(fake_img):
    return -tf.reduce_mean(fake_img)


# class GANMonitor(keras.callbacks.Callback):
#     def __init__(self,num_img=6,latent_dim=128):
#         self.num_img = num_img
#         self.latent_dim = latent_dim

#     def on_epoch_end(self, epoch, logs=None):
#         random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
#         generated_images = self.model.generator(random_latent_vectors)

#         for i in range(self.num_img):
#             img = generated_images[i].numpy()
#             img = keras.preprocessing.image.array_to_img(img)
#             img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))


def GaussianNLL(true, pred):
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


class RepGAN(Model):

    def __init__(self,options):
        super(RepGAN, self).__init__()
        """
            Setup
        """
        self.__dict__.update(options)

        assert self.nZchannels >= 1
        assert self.nZchannels >= self.stride**self.nCnnLayers
        assert self.latentZdim >= self.Xsize//(self.stride**self.nCnnLayers)

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

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [AdvDLoss_tracker,AdvGLoss_tracker]

    def compile(self,optimizers,losses):
        super(RepGAN, self).compile()
        """
            Optimizers
        """
        self.__dict__.update(optimizers)
        """
            Losses
        """
        self.__dict__.update(losses)

    def GradientPenaltyX(self,batchSize,realX,fakeX):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batchSize, 1, 1], 0.0, 1.0)
        diffX = fakeX - realX
        intX = realX + alpha * diffX

        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(intX)
            # 1. Get the discriminator output for this interpolated image.
            predX = self.Dx(intX,training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        GradX = gp_tape.gradient(predX, [intX])[0]
        # 3. Calculate the norm of the gradients.
        NormGradX = tf.sqrt(tf.reduce_sum(tf.square(GradX), axis=[1, 2]))
        gp = tf.reduce_mean((NormGradX - 1.0) ** 2)
        return gp

    def train_step(self, realXC):

        realX, realC = realXC
        
        # Get the batch size
        #batchSize = tf.shape(realX)[0]
        #if self.batchSize != batchSize:
        self.batchSize = tf.shape(realX)[0]

        #------------------------------------------------
        #           Construct Computational Graph
        #               for the Discriminator
        #------------------------------------------------

        # Freeze generators' layers while training critics
        self.Fx.trainable = False
        self.Gz.trainable = False
        self.Dx.trainable = True
        self.Dc.trainable = True
        self.Ds.trainable = True
        self.Dn.trainable = True



        for _ in range(self.nCritic):

            # Sample noise as generator input
            # realZ = tf.random.normal(shape=[self.batchSize, self.latentZdim], mean=0.0, stddev=1.0)
            # realS = tf.random.normal(shape=[self.batchSize, self.latentSdim], mean=0.0, stddev=1.0)
            # realN = tf.random.normal(shape=[self.batchSize, self.latentNdim], mean=0.0, stddev=1.0)
        # The generator takes the signal, encodes it and reconstructs it
        # from the encoding
        # Real c,s,n

        # realC = tf.zeros(shape=(self.batchSize,self.latentCdim))
        # rand_idx = np.random.randint(0,self.latentCdim,self.batchSize)
        # realC[np.arange(self.batchSize),rand_idx]=1.0

            with tf.GradientTape(persistent=True) as tape:

                # realC = tf.zeros(shape=(self.batchSize,self.latentCdim))
                # rand_idx = np.random.randint(0,self.latentCdim,self.batchSize)
                #realC[np.arange(self.batchSize),rand_idx]=1.0
                #realC = tf.random.categorical(tf.math.log([[batchSize, 2]]), self.latentCdim)
                realS = tf.random.normal(mean=0.0,stddev=0.5,shape=[self.batchSize,self.latentSdim])
                realN = tf.random.normal(mean=0.0,stddev=0.3,shape=[self.batchSize,self.latentNdim])

                # # Generate fake latent code from real signals
                (fakeC,fakeS,fakeN) = self.Fx(realX) # encoded z = Fx(X)

                fakeX = self.Gz((realC,realS,realN)) # fake X = Gz(Fx(X))
                

                # Discriminator determines validity of the real and fake X
                fakeXcritic = self.Dx(fakeX)
                realXcritic = self.Dx(realX)

                # Discriminator determines validity of the real and fake C
                fakeCcritic = self.Dc(fakeC)
                realCcritic = self.Dc(realC)

                # Discriminator determines validity of the real and fake N
                fakeNcritic = self.Dn(fakeN)
                realNcritic = self.Dn(realN) 

                # Discriminator determines validity of the real and fake S
                fakeScritic = self.Ds(fakeS)
                realScritic = self.Ds(realS)

                # Calculate the discriminator loss using the fake and real logits
                AdvDlossX = self.AdvDloss(realXcritic,fakeXcritic)*self.PenAdvXloss
                AdvDlossC = self.AdvDloss(realCcritic,fakeCcritic)*self.PenAdvCloss
                AdvDlossS = self.AdvDloss(realScritic,fakeScritic)*self.PenAdvSloss
                AdvDlossN = self.AdvDloss(realNcritic,fakeNcritic)*self.PenAdvNloss
                AdvDlossPenGradX = self.GradientPenaltyX(self.batchSize,realX,fakeX)*self.PenGradX
                AdvDloss = AdvDlossX + AdvDlossC + AdvDlossS + AdvDlossN + AdvDlossPenGradX

            # Get the gradients w.r.t the discriminator loss
            
            gradDx, gradDc, gradDs, gradDn = tape.gradient(AdvDloss, 
                (self.Dx.trainable_variables, self.Dc.trainable_variables, 
                self.Ds.trainable_variables, self.Dn.trainable_variables))
            
            # Update the weights of the discriminator using the discriminator optimizer
            self.DxOpt.apply_gradients(zip(gradDx,self.Dx.trainable_variables))
            self.DcOpt.apply_gradients(zip(gradDc,self.Dc.trainable_variables))
            self.DsOpt.apply_gradients(zip(gradDs,self.Ds.trainable_variables))
            self.DnOpt.apply_gradients(zip(gradDn,self.Dn.trainable_variables))

            # Clip critic weights
            # for l in self.Dx.layers:
            #     weights = l.get_weights()
            #     weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
            #     l.set_weights(weights)
            # for l in self.Dc.layers:
            #     weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
            #     l.set_weights(weights)
            # for l in self.Ds.layers:
            #     weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
            #     l.set_weights(weights)
            # for l in self.Dn.layers:
            #     weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                # l.set_weights(weights)

        #----------------------------------------
        #      Construct Computational Graph
        #               for Generator
        #----------------------------------------

        # Freeze critics' layers while training generators
        self.Fx.trainable = True
        self.Gz.trainable = True
        self.Dx.trainable = False
        self.Dc.trainable = False
        self.Ds.trainable = False
        self.Dn.trainable = False

        with tf.GradientTape(persistent=True) as tape:
            # Fake
            (fakeC,fakeS,fakeN) = self.Fx(realX) # encoded z = Fx(X)
            fakeX = self.Gz((fakeC,fakeS,fakeN)) # fake X = Gz(Fx(X))
            
            # Discriminator determines validity of the real and fake X
            fakeXcritic = self.Dx(fakeX)

            # Discriminator determines validity of the real and fake C
            fakeCcritic = self.Dc(fakeC)

            # Discriminator determines validity of the real and fake S
            fakeScritic = self.Ds(fakeS)

            # Discriminator determines validity of the real and fake N
            fakeNcritic = self.Dn(fakeN)

            # Reconstruction
            # fakeZ = Concatenate([fakeC,fakeS,fakeN])
            recX  = self.Gz((fakeC,fakeS,fakeN))
            (recC,recS,_)  = self.Fx(fakeX)

            AdvGlossX = self.AdvGloss(fakeXcritic)*self.PenAdvXloss
            AdvGlossC = self.AdvGloss(fakeCcritic)*self.PenAdvCloss
            AdvGlossS = self.AdvGloss(fakeScritic)*self.PenAdvSloss
            AdvGlossN = self.AdvGloss(fakeNcritic)*self.PenAdvNloss
            RecGlossX = self.RecXloss(realX,recX)*self.PenRecXloss
            RecGlossC = self.RecCloss(realC,recC)*self.PenRecCloss
            RecGlossS = self.RecSloss(realS,recS)*self.PenRecSloss
            AdvGloss = AdvGlossX + AdvGlossC + AdvGlossS + AdvGlossN + RecGlossX + RecGlossC + RecGlossS

        # Get the gradients w.r.t the generator loss
        gradFx, gradGz = tape.gradient(AdvGloss, 
                (self.Fx.trainable_variables, self.Gz.trainable_variables))
        # Update the weights of the generator using the generator optimizer
        self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))
        self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))

        ## Compute our own metrics
        #AdvDloss_tracker.update_state(AdvDloss)
        #AdvGloss_tracker.update_state(AdvGloss)

        #return {"AdvDloss": AdvDloss_tracker.result(), 
        #    "AdvGloss": AdvGloss_tracker.results()}

        
        return {"AdvDloss": AdvDloss, 
            "AdvGloss": AdvGloss}


    def build_Fx(self):
        """
            Conv1D Fx structure
        """
        
        X = Input(shape=self.Xshape)

        # for n in range(self.nCnnLayers):
        #     X = Conv1D((self.latentZdim/self.Zsize)*self.stride**(-self.nCnnLayers+n),
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
        for n in range(self.nCnnLayers):
            model.add(Conv1D((self.latentZdim/self.Zsize)*self.stride**(-self.nCnnLayers+n),
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


        return keras.Model(X,(c,s,n),name="Fx")

    def build_Gz(self):
        """
            Conv1D Gz structure
        """
        c = tf.keras.Input(shape=(self.latentCdim,))
        s = tf.keras.Input(shape=(self.latentSdim,))
        n = tf.keras.Input(shape=(self.latentNdim,))
             
        GzC = Dense(self.Zsize*self.nCchannels,use_bias=False)(c)
        GzC = Reshape((self.Zsize,self.nCchannels))(GzC)
        GzC = Model(c,GzC)

        GzS = Dense(self.Zsize*self.nSchannels,use_bias=False)(s)
        GzS = Reshape((self.Zsize,self.nSchannels))(GzS)
        GzS = Model(s,GzS)

        GzN = Dense(self.Zsize*self.nNchannels,use_bias=False)(n)
        GzN = Reshape((self.Zsize,self.nNchannels))(GzN)
        GzN = Model(n,GzN)

        z = concatenate([GzC.output,GzS.output,GzN.output])


        #Gz = Reshape((self.Zsize,self.nZchannels))(z)
        Gz = BatchNormalization(axis=-1,momentum=0.95)(z)
        Gz = Activation('relu')(Gz)

        for n in range(self.nCnnLayers):
            Gz = Conv1DTranspose(self.latentZdim//self.stride**n,
                self.kernel,self.stride,padding="same")(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)
            Gz = Activation('relu')(Gz)
        
        Gz = Conv1DTranspose(self.nXchannels,self.kernel,1,padding="same")(Gz)

        #model.summary()
         
              
        return keras.Model([GzC.input,GzS.input,GzN.input],Gz,name="Gz")
        
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
        model.add(Dense(1,activation=None))
        # model.add(Dense(1,activation='sigmoid'))

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

        model.summary()

        X = Input(shape=(self.Xshape))
        Dx = model(X)

        return keras.Model(X,Dx,name="Dx")


    def build_Dc(self):
        """
            Dense discriminator structure
        """
        model = Sequential()
        model.add(Dense(3000,kernel_constraint=min_max_norm(2.)))
        model.add(LeakyReLU())
        model.add(Dense(3000,kernel_constraint=min_max_norm(2.)))
        model.add(LeakyReLU())
        model.add(Dense(1))  

        #model.summary()

        c = Input(shape=(self.latentCdim,))
        Dc = model(c)

        return keras.Model(c,Dc,name="Dc")

    def build_Dn(self):
        """
            Dense discriminator structure
        """
        model = Sequential()
        model.add(Dense(3000,kernel_constraint=min_max_norm(2.)))
        model.add(LeakyReLU())
        model.add(Dense(3000,kernel_constraint=min_max_norm(2.)))
        model.add(LeakyReLU()) 
        model.add(Dense(1))  

        #model.summary()

        n = Input(shape=(self.latentNdim,))
        Dn = model(n)

        return keras.Model(n,Dn,name="Dn")

    def build_Ds(self):
        """
            Dense discriminator structure
        """
        model = Sequential()
        model.add(Dense(3000,kernel_constraint=min_max_norm(2.)))
        model.add(LeakyReLU())
        model.add(Dense(3000,kernel_constraint=min_max_norm(2.)))
        model.add(LeakyReLU())
        model.add(Dense(1))  

        #model.summary()

        s = Input(shape=(self.latentSdim,))
        Ds = model(s)

        return keras.Model(s,Ds,name="Ds")


if __name__ == '__main__':

    options = ParseOptions()

    optimizers = {}
    optimizers['DxOpt'] = RMSprop(lr=0.00005)
    optimizers['DcOpt'] = RMSprop(lr=0.00005)
    optimizers['DsOpt'] = RMSprop(lr=0.00005)
    optimizers['DnOpt'] = RMSprop(lr=0.00005)
    optimizers['FxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    optimizers['GzOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    losses = {}
    losses['AdvDloss'] = WassersteinDiscriminatorLoss
    losses['AdvGloss'] = WassersteinGeneratorLoss
    losses['RecSloss'] = GaussianNLL
    losses['RecXloss'] = tf.keras.losses.MeanSquaredError()
    losses['RecCloss'] = tf.keras.losses.BinaryCrossentropy()
    losses['PenAdvXloss'] = 1.
    losses['PenAdvCloss'] = 1.
    losses['PenAdvSloss'] = 1.
    losses['PenAdvNloss'] = 1.
    losses['PenRecXloss'] = 1.
    losses['PenRecCloss'] = 1.
    losses['PenRecSloss'] = 1.
    losses['PenGradX'] = 10.
    

    # extra_options = {}
    # extra_options['metrics'] = [tf.keras.metrics.Accuracy()]
    # extra_options['weighted_metrics'] = None
    # extra_options['loss_weights'] = None
    # extra_options['run_eagerly'] = None
    # extra_options['steps_per_execution'] = None
    # extra_options['target_tensors'] = None
    # extra_options['sample_weight_mode'] = None
    # extra_options['experimental_run_tf_function'] = None
    #metrics = {}
    #metrics['Accuracy'] = tf.keras.metrics.Accuracy()

    #weighted_metrics = {}

    #loss_weights = {}

    #run_eagerly = {}
    #run_eagerly['run_eagerly'] = None

    #steps_per_execution = {}
    #steps_per_execution['steps_per_execution'] = None

    #target_tensors = {}
    #target_tensors['target_tensors'] = None

    #sample_weight_mode = {}
    #sample_weight_mode['sample_weight_mode'] = None

    #experimental_run_tf_function = {}
    #experimental_run_tf_function['experimental_run_tf_function'] = None

    # Instantiate the RepGAN model.
    GiorgiaGAN = RepGAN(options)

    # Compile the RepGAN model.
    GiorgiaGAN.compile(optimizers,losses)


    # Load the dataset
    Xtrn,  Xvld, params_trn, params_vld, Ctrn, Cvld, Strn, Svld, Ntrn, Nvld = mdof.load_data(**options)


    Xtrn = Xtrn.astype("float32") 
    Xvld = Xvld.astype("float32")
    
    # Xtrn = tf.data.Dataset.from_tensor_slices(Xtrn)
    # Xtrn = Xtrn.shuffle(buffer_size=1024).batch(options['batchSize'])

    # Start training the model.
    #GiorgiaGAN.fit(x=Xtrn,y=Ctrn,batch_size=options["batchSize"],epochs=options["epochs"])
    GiorgiaGAN.fit(Xtrn,Ctrn,batch_size=options["batchSize"],epochs=options["epochs"])