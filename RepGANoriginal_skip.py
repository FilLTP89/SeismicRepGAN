## -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
__author__ = "Giorgia Colombera, Filippo Gatti"
__copyright__ = "Copyright 2021, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Giorgia Colombera,Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__Maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from utils import *
import math as mt

import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
# Spectral normalisation tf implementation
import tensorflow_addons as tfa
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Layer
from tensorflow.keras.layers import Lambda, Concatenate, concatenate, ZeroPadding1D, Add
from tensorflow.keras.layers import LeakyReLU, ReLU, Softmax, Activation, Average, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.constraints import Constraint, min_max_norm

import tensorflow_probability as tfp

import timeit
# from tensorflow.python.eager import context
# import keras_tuner as kt
# from keras_tuner.tuners import RandomSearch
# from keras_tuner import HyperModel
import MDOFload as mdof
import matplotlib.pyplot as plt
import h5py
from copy import deepcopy
from plot_tools import *

from RepGAN_losses import *

tf.keras.backend.set_floatx('float32')

AdvDLoss_tracker = keras.metrics.Mean(name="loss")
AdvDlossX_tracker = keras.metrics.Mean(name="loss")
AdvDlossC_tracker = keras.metrics.Mean(name="loss")
AdvDlossS_tracker = keras.metrics.Mean(name="loss")
AdvDlossN_tracker = keras.metrics.Mean(name="loss")
AdvGLoss_tracker = keras.metrics.Mean(name="loss")
AdvGlossX_tracker = keras.metrics.Mean(name="loss")
AdvGlossC_tracker = keras.metrics.Mean(name="loss")
AdvGlossS_tracker = keras.metrics.Mean(name="loss")
AdvGlossN_tracker = keras.metrics.Mean(name="loss")
RecGlossX_tracker = keras.metrics.Mean(name="loss")
RecGlossC_tracker = keras.metrics.Mean(name="loss")
RecGlossS_tracker = keras.metrics.Mean(name="loss")
Qloss_tracker = keras.metrics.Mean(name="loss")

#gpu_devices = tf.config.experimental.list_physical_devices('GPU')

checkpoint_dir = "/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/01_06"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


class RandomWeightedAverage(Layer):
    """Provides a (random) weighted average between real and generated signal samples"""
    def _merge_function(self,inputs,**kwargs):
        alpha = tf.random_uniform((32,1,1,1))
        return (alpha*inputs[0])+((1.0-alpha)*inputs[1])

class SamplingFxS(Layer):
    """Uses (z_mean, z_std) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_std = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        #logz = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        #return tf.exp(logz)
        # return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_mean + z_std * epsilon

class SamplingFxNormSfromSigma(Layer):
    def call(self, inputs):
        z_mean, z_std = inputs
        # z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        #z = z_mean + tf.multiply(z_std,tf.random.normal([1]))
        z = z_mean + tf.multiply(z_std,tf.random.normal(shape=(batch,dim),dtype=tf.float32))
        return z

class SamplingFxNormSfromVariance(Layer):
    def call(self, inputs):
        z_mean, z_var = inputs
        z_std = tf.math.sqrt(z_var)
        epsilon = tf.random.normal(shape=tf.shape(z_mean),mean=0.0,stddev=1.0)
        z = z_mean + tf.math.multiply(z_std,epsilon)
        return z

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
    # clip model weights to hypercube
    def __call__(self, weights):
        if self.clip_value is not None:
            return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)
        else:             
            return tf.keras.backend.clip(weights, None, None)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    idx = np.random.randint(low=0, high=dataset.shape[0], size=n_samples)
    #X[tf.arange(size_batch), rand_idx] = dataset
    X = tf.gather(dataset, idx)
    # idx = np.random.randint(0, size=(dataset.shape[0], n_samples))
    # select images
    # X = dataset[idx]

    return X


class RepGAN(Model):

    def __init__(self,options):
        super(RepGAN, self).__init__()
        """
            Setup
        """
        self.__dict__.update(options)

        # define the constraint
        self.ClipD = ClipConstraint(0.01)
        """
            Build the discriminators
        """
        self.Dx = self.BuildDx()
        self.Dc = self.BuildDc()
        self.Ds = self.BuildDs()
        self.Dn = self.BuildDn()
        """
            Build Fx/Gz (generators)
        """

        self.Fx = self.BuildFx() 
        self.Gz = self.BuildGz()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.Xshape#self.size
        })
        return config

    @property
    def metrics(self):
        return [AdvDLoss_tracker,AdvGLoss_tracker,AdvDlossX_tracker,AdvDlossC_tracker,AdvDlossS_tracker,
            AdvDlossN_tracker,AdvGlossX_tracker,AdvGlossC_tracker,AdvGlossS_tracker,AdvGlossN_tracker,
            RecGlossX_tracker,RecGlossC_tracker,RecGlossS_tracker,Qloss_tracker]

    def compile(self,optimizers,losses): #run_eagerly
        super(RepGAN, self).compile()
        """
            Optimizers
        """
        self.__dict__.update(optimizers)
        """
            Losses
        """
        self.__dict__.update(losses)

    def GradientPenaltyX(self,batchSize,X,fakeX):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batchSize, 1, 1], 0.0, 1.0)
        diffX = fakeX - X
        intX = X + alpha * diffX

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(intX)
            # 1. Get the discriminator output for this interpolated image.
            predX = self.Dx(intX,training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        GradDx = gp_tape.gradient(predX, [intX])[0]
        # 3. Calculate the norm of the gradients.
        NormGradX = tf.sqrt(tf.reduce_sum(tf.square(GradDx), axis=[1]))
        gp = tf.reduce_mean((NormGradX - 1.0) ** 2)
        return gp

    def GradientPenaltyS(self,batchSize,X,fakeX):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batchSize, 1], 0.0, 1.0)
        diffX = fakeX - X
        intX = X + alpha * diffX

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(intX)
            # 1. Get the discriminator output for this interpolated image.
            predX = self.Ds(intX,training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        GradX = gp_tape.gradient(predX, [intX])[0]
        # 3. Calculate the norm of the gradients.
        NormGradX = tf.sqrt(tf.reduce_sum(tf.square(GradX), axis=[1]))
        gp = tf.reduce_mean((NormGradX - 1.0) ** 2)
        return gp

    def train_XZX(self,X,c):

        # Create labels for BCE in GAN loss
        realBCE_C = tf.ones((self.batchSize,1), dtype=tf.float32)
        fakeBCE_C = -tf.ones((self.batchSize,1), dtype=tf.float32)
        realBCE_S = tf.ones((self.batchSize,1), dtype=tf.float32)
        fakeBCE_S = -tf.ones((self.batchSize,1), dtype=tf.float32)
        realBCE_N = tf.ones((self.batchSize,1), dtype=tf.float32)
        fakeBCE_N = -tf.ones((self.batchSize,1), dtype=tf.float32)

        # Train generators
        self.Fx.trainable = True
        self.Gz.trainable = True
        self.Dx.trainable = False
        self.Dc.trainable = False
        self.Ds.trainable = False
        self.Dn.trainable = False

        # Train nGenerator times the generators
        for _ in range(self.nGenerator):

            with tf.GradientTape(persistent=True) as tape:

                # Encode real signals
                _,_,s_fake,c_fake,n_fake = self.Fx(X,training=True)

                # Reconstruct real signals
                X_rec = self.Gz((s_fake,c_fake,n_fake),training=True)

                # Compute reconstruction loss
                RecGlossX = self.RecXloss(X,X_rec)*self.PenRecXloss
        
            # Get the gradients w.r.t the generator loss
            gradFx, gradGz = tape.gradient(RecGlossX,
            (self.Fx.trainable_variables,self.Gz.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Update the weights of the generator using the generator optimizer
            self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))
            self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))

        # Train discriminators
        self.Fx.trainable = False
        self.Gz.trainable = False
        self.Dx.trainable = False
        self.Dc.trainable = True
        self.Ds.trainable = True
        self.Dn.trainable = True

        # Train nCritic times the discriminators
        for _ in range(self.nCritic):
            
            # Sample factorial prior S
            s_prior = tf.random.normal(mean=0.0,stddev=1.0,
                shape=[self.batchSize,self.latentSdim],
                dtype=tf.float32)

            # Sample factorial prior N
            n_prior = tf.random.normal(mean=0.0,stddev=1.0,
                shape=[self.batchSize,self.latentNdim],
                dtype=tf.float32)
            
            with tf.GradientTape(persistent=True) as tape:

	            # Encode real signals X
                [_,_,s_fake,c_fake,n_fake] = self.Fx(X,training=True)

                # Discriminates real and fake S
                s_fakecritic = self.Ds(s_fake,training=True)
                s_priorcritic = self.Ds(s_prior,training=True)

                # Discriminates real and fake N
                n_fakecritic = self.Dn(n_fake,training=True)
                n_priorcritic = self.Dn(n_prior,training=True)

                # Discriminates real and fake C
                c_fakecritic = self.Dc(c_fake,training=True)
                c_critic = self.Dc(c,training=True)

                # Compute XZX adversarial loss (JS(s),JS(n),JS(c))
                AdvDlossC  = self.AdvDlossDz(realBCE_C,c_critic)*self.PenAdvCloss
                AdvDlossC += self.AdvDlossDz(fakeBCE_C,c_fakecritic)*self.PenAdvCloss
                AdvDlossS  = self.AdvDlossDz(realBCE_S,s_priorcritic)*self.PenAdvSloss
                AdvDlossS += self.AdvDlossDz(fakeBCE_S,s_fakecritic)*self.PenAdvSloss
                AdvDlossN  = self.AdvDlossDz(realBCE_N,n_priorcritic)*self.PenAdvNloss
                AdvDlossN += self.AdvDlossDz(fakeBCE_N,n_fakecritic)*self.PenAdvNloss
                
                AdvDloss = 0.5*(AdvDlossC + AdvDlossS + AdvDlossN)

            # Compute the discriminator gradient
            gradDc, gradDs, gradDn = tape.gradient(AdvDloss,
                    (self.Dc.trainable_variables,self.Ds.trainable_variables, 
                    self.Dn.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Update discriminators' weights
            self.DcOpt.apply_gradients(zip(gradDc,self.Dc.trainable_variables))
            self.DsOpt.apply_gradients(zip(gradDs,self.Ds.trainable_variables))
            self.DnOpt.apply_gradients(zip(gradDn,self.Dn.trainable_variables))
        

        self.Fx.trainable = True
        self.Gz.trainable = False
        self.Dx.trainable = False
        self.Dc.trainable = False
        self.Ds.trainable = False
        self.Dn.trainable = False

        with tf.GradientTape(persistent=True) as tape:

            # Encode real signals
            _,_,s_fake,c_fake,n_fake = self.Fx(X,training=True)

            # Discriminate fake latent space
            s_fakecritic = self.Ds(s_fake,training=True)
            c_fakecritic = self.Dc(c_fake,training=True)
            n_fakecritic = self.Dn(n_fake,training=True)

            # Compute adversarial loss for generator
            AdvGlossC = self.AdvGlossDz(realBCE_C,c_fakecritic)*self.PenAdvCloss
            AdvGlossS = self.AdvGlossDz(realBCE_S,s_fakecritic)*self.PenAdvSloss
            AdvGlossN = self.AdvGlossDz(realBCE_N,n_fakecritic)*self.PenAdvNloss

            # Compute total generator loss
            AdvGloss = AdvGlossC + AdvGlossS + AdvGlossN
            
        # Get the gradients w.r.t the generator loss
        gradFx = tape.gradient(AdvGloss,(self.Fx.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Update the weights of the generator using the generator optimizer
        self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))

        return RecGlossX,AdvDloss,AdvDlossC,AdvDlossS,AdvDlossN,AdvGloss,AdvGlossC,AdvGlossS,AdvGlossN,\
                c_fakecritic,s_fakecritic,n_fakecritic,c_critic,s_priorcritic,n_priorcritic

    def train_ZXZ(self,X,c):

        # Create labels for BCE in GAN loss
        realBCE_X = tf.ones((self.batchSize,1), dtype=tf.float32)
        fakeBCE_X = tf.zeros((self.batchSize,1), dtype=tf.float32)

        # Train discriminators
        self.Fx.trainable = False
        self.Gz.trainable = False
        self.Dx.trainable = True

        # Train nCritic times the discriminators
        for _ in range(self.nCritic):

            # Sample factorial prior S
            s_prior = tf.random.normal(mean=0.0,stddev=1.0,
                shape=[self.batchSize,self.latentSdim],
                dtype=tf.float32)

            # Sample factorial prior N
            n_prior = tf.random.normal(mean=0.0,stddev=1.0,
                shape=[self.batchSize,self.latentNdim],
                dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:

                # Decode factorial prior
                X_fake = self.Gz((s_prior,c,n_prior),training=True)

                # Discriminate real and fake X
                X_fakecritic = self.Dx(X_fake,training=True)
                X_critic = self.Dx(X,training=True)

                # Compute the discriminator loss GAN loss (penalized)
                AdvDlossX = self.AdvDlossDx(fakeBCE_X,X_fakecritic)*self.PenAdvXloss
                AdvDlossX += self.AdvDlossDx(realBCE_X,X_critic)*self.PenAdvXloss

            # Compute the discriminator gradient
            gradDx = tape.gradient(AdvDlossX,self.Dx.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            # Update the weights of the discriminator using the discriminator optimizer
            self.DxOpt.apply_gradients(zip(gradDx,self.Dx.trainable_variables))

        
        # Train generators
        self.Fx.trainable = True
        self.Gz.trainable = True
        self.Dx.trainable = False
        
        # Train nGenerator times the generators
        for _ in range(self.nGenerator):

            # Sample factorial prior S
            s_prior = tf.random.normal(mean=0.0,stddev=1.0,
                shape=[self.batchSize,self.latentSdim],
                dtype=tf.float32)
            
            # Sample factorial prior N
            n_prior = tf.random.normal(mean=0.0,stddev=1.0,
                shape=[self.batchSize,self.latentNdim],
                dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:

                # Decode factorial prior
                X_fake = self.Gz((s_prior,c,n_prior),training=True)

                # Discriminate real and fake X
                X_fakecritic = self.Dx(X_fake,training=True)

                # Compute adversarial loos (penalized)
                AdvGlossX = self.AdvGlossDx(realBCE_X,X_fakecritic)*self.PenAdvXloss
                
                # Encode fake signals
                [μs_rec,σs_rec,s_rec,c_rec,_] = self.Fx(X_fake,training=True)

                #Q_cont_distribution = tfp.distributions.MultivariateNormalDiag(loc=μs_rec, scale_diag=σs_rec)
                #RecGlossS = -tf.reduce_mean(Q_cont_distribution.log_prob(recS))
                RecGlossS = self.RecSloss(s_prior,μs_rec,σs_rec)*self.PenRecSloss
                RecGlossC = self.RecCloss(c,c_rec)*self.PenRecCloss #+ self.RecCloss(c,c_fake)*self.PenRecCloss

                # Compute InfoGAN Q loos
                Qloss = RecGlossS + RecGlossC

            # Get the gradients w.r.t the generator loss
            gradGz = tape.gradient(AdvGlossX,(self.Gz.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Get the gradients w.r.t the generator loss
            gradFx, gradGz = tape.gradient(Qloss,
                (self.Fx.trainable_variables,self.Gz.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Update the weights of the generator using the generator optimizer
            self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))

            # Update the weights of the generator using the generator optimizer
            self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))
            self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))

        return AdvDlossX,AdvGlossX,RecGlossS,RecGlossC,Qloss,X_fakecritic,X_critic

    def train_step(self, XC):

        X, c, mag, di = XC

        self.batchSize = tf.shape(X)[0]

        #------------------------------------------------
        #           Construct Computational Graph
        #               for the Discriminator
        #------------------------------------------------
        #       
        for _ in range(self.nRepXRep):  
            ZXZout = self.train_ZXZ(X,c)
        for _ in range(self.nXRepX):
            XZXout = self.train_XZX(X,c)

        (AdvDlossX,AdvGlossX,RecGlossS,RecGlossC,Qloss,X_fakecritic,X_critic) = ZXZout

        (RecGlossX,AdvDloss,AdvDlossC,AdvDlossS,AdvDlossN,AdvGloss,AdvGlossC,AdvGlossS,AdvGlossN,\
            c_fakecritic,s_fakecritic,n_fakecritic,c_critic,s_priorcritic,n_priorcritic) = XZXout     
      
        # Compute our own metrics
        AdvDLoss_tracker.update_state(AdvDloss)
        AdvGLoss_tracker.update_state(AdvGloss)
        AdvDlossX_tracker.update_state(AdvDlossX)
        AdvDlossC_tracker.update_state(AdvDlossC)
        AdvDlossS_tracker.update_state(AdvDlossS)
        AdvDlossN_tracker.update_state(AdvDlossN)

        AdvGlossX_tracker.update_state(AdvGlossX)
        AdvGlossC_tracker.update_state(AdvGlossC)
        AdvGlossS_tracker.update_state(AdvGlossS)
        AdvGlossN_tracker.update_state(AdvGlossN)

        RecGlossX_tracker.update_state(RecGlossX)
        RecGlossC_tracker.update_state(RecGlossC)
        RecGlossS_tracker.update_state(RecGlossS)

        Qloss_tracker.update_state(Qloss)

        return {"AdvDLoss": AdvDLoss_tracker.result(),"AdvGLoss": AdvGLoss_tracker.result(),"AdvDlossX": AdvDlossX_tracker.result(),
            "AdvDlossC": AdvDlossC_tracker.result(), "AdvDlossS": AdvDlossS_tracker.result(),"AdvDlossN": AdvDlossN_tracker.result(),
            "AdvGlossX": AdvGlossX_tracker.result(),"AdvGlossC": AdvGlossC_tracker.result(),"AdvGlossS": AdvGlossS_tracker.result(),
            "AdvGlossN": AdvGlossN_tracker.result(),"RecGlossX": RecGlossX_tracker.result(),"RecGlossC": RecGlossC_tracker.result(),
            "RecGlossS": RecGlossS_tracker.result(), "Qloss": Qloss_tracker.result(),
            "fakeX":tf.math.reduce_mean(X_fakecritic),"X":tf.math.reduce_mean(X_critic),
            "c_fake":tf.math.reduce_mean(c_fakecritic),"c":tf.math.reduce_mean(c_critic),"n_fake":tf.math.reduce_mean(n_fakecritic),
            "n_prior":tf.math.reduce_mean(n_priorcritic),"s_fake":tf.math.reduce_mean(s_fakecritic),"s_prior":tf.math.reduce_mean(s_priorcritic)}


    def call(self, X):
        [_,_,s_fake,c_fake,n_fake] = self.Fx(X)
        X_rec = self.Gz((s_fake,c_fake,n_fake))
        return X_rec, c_fake, s_fake, n_fake

    def plot(self,X,c):
        [_,_,s_fake,c_fake,n_fake] = self.Fx(X,training=False)
        s_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],self.latentSdim])
        n_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],self.latentNdim])
        fakeX = self.Gz((s_prior,c,n_prior),training=False)
        X_rec = self.Gz((s_fake,c_fake,n_fake),training=False)
        return X_rec, c_fake, s_fake, n_fake, fakeX

    def label_predictor(self, X, c):
        [_,_,s_fake,c_fake,n_fake] = self.Fx(X)
        s_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[s_fake.shape[0],self.latentSdim])
        n_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[n_fake.shape[0],self.latentNdim])
        fakeX = self.Gz((s_prior,c,n_prior),training=False)
        [_,_,_,c_rec,_] = self.Fx(fakeX,training=False)
        return c_fake, c_rec
    
    def distribution(self,X,c):
        [μs,σs,s_fake,c_fake,n_fake] = self.Fx(X,training=False)
        s_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],self.latentSdim])
        n_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],self.latentNdim])
        fakeX = self.Gz((s_prior,c,n_prior),training=False)
        [μs_rec,σs_rec,s_rec,c_rec,n_rec] = self.Fx(fakeX,training=False)
        return s_prior, n_prior, s_fake, n_fake, s_rec, n_rec, μs, σs, μs_rec, σs_rec

    def generate(self, X, c_fake_new):
        [_,_,s_fake,c_fake,n_fake] = self.Fx(X)
        X_rec_new = self.Gz((s_fake,c_fake_new,n_fake),training=False)
        return X_rec_new

    def BuildFx(self):
        """
            Conv1D Fx structure
        """
        # To build this model using the functional API

        # Input layer
        X = Input(shape=self.Xshape,name="X")

        # Initial CNN layer
        layer = -1
        h = Conv1D(self.nZfirst, 
                self.kernel,1,padding="same",
                data_format="channels_last",name="FxCNN0")(X)
        h = BatchNormalization(momentum=0.95)(h)
        h = LeakyReLU(alpha=0.1,name="FxA0")(h)
        h = Dropout(0.2,name="FxDO0")(h)

        # Common encoder CNN layers
        for layer in range(self.nAElayers):
            h = Conv1D(self.nZfirst*self.stride**(layer+1),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
            h = BatchNormalization(momentum=0.95)(h)
            h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
            h = Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)

        # Last common CNN layer (no stride, same channels) before branching
        layer = self.nAElayers
        h = Conv1D(self.nZchannels,
            self.kernel,1,padding="same",
            data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
        h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
        h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
        z = Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)
        # z ---> Zshape = (Zsize,nZchannels)

        layer = 0
        # variable s
        # s-average
        Zmu = Conv1D(self.nZchannels*self.Sstride**(layer+1),
            self.Skernel,self.Sstride,padding="same",
            data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(z)
        Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(Zmu)
        Zmu = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(Zmu)
        Zmu = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(Zmu)

        # s-log std
        Zsigma = Conv1D(self.nZchannels*self.Sstride**(layer+1),
            self.Skernel,self.Sstride,padding="same",
            data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(z)
        Zsigma = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zsigma)
        Zsigma = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zsigma)
        Zsigma = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zsigma)

        # variable c
        Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(z)
        Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
        Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
        #Zc = tfa.layers.InstanceNormalization()(Zc)
        Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)

        # variable n
        Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(z)
        Zn = BatchNormalization(momentum=0.95)(Zn)
        Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
        #Zn = tfa.layers.InstanceNormalization()(Zn)
        Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)

        # variable s
        for layer in range(1,self.nSlayers):
            # s-average
            Zmu = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(Zmu)
            Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(Zmu)
            Zmu = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(Zmu)
            Zmu = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(Zmu)

            # s-log std
            Zsigma = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(Zsigma)
            Zsigma = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zsigma)
            Zsigma = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zsigma)
            Zsigma = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zsigma)

        # variable c
        for layer in range(1,self.nClayers):
            Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(Zc)
            Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
            Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
            #Zc = tfa.layers.InstanceNormalization()(Zc)
            Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)

        # variable n
        for layer in range(1,self.nNlayers):
            Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(Zn)
            Zn = BatchNormalization(momentum=0.95)(Zn)
            Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
            #Zn = tfa.layers.InstanceNormalization()(Zn)
            Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)

        # variable s
        Zmu = Flatten(name="FxFLmuS{:>d}".format(layer+1))(Zmu)
        Zmu = Dense(1024)(Zmu)
        Zmu = BatchNormalization(momentum=0.95)(Zmu)
        Zmu = LeakyReLU(alpha=0.1)(Zmu)
        Zmu = Dense(self.latentSdim,name="FxFWmuS")(Zmu)
        Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS")(Zmu)
        Zmu = LeakyReLU(alpha=0.1)(Zmu)

        # s-sigma
        Zsigma = Flatten(name="FxFLlvS{:>d}".format(layer+1))(Zsigma)
        Zsigma = Dense(1024)(Zsigma)
        Zsigma = BatchNormalization(momentum=0.95)(Zsigma)
        Zsigma = LeakyReLU(alpha=0.1)(Zsigma)
        Zsigma = Dense(self.latentSdim,name="FxFWlvS")(Zsigma)
        Zsigma = BatchNormalization(momentum=0.95,axis=-1,name="FxBNlvS")(Zsigma)
        Zsigma = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+2))(Zsigma)     
        Zsigma = tf.math.sigmoid(Zsigma)

        # variable c
        layer = self.nClayers
        Zc = Flatten(name="FxFLC{:>d}".format(layer+1))(Zc)
        Zc = Dense(1024)(Zc)
        Zc = BatchNormalization(momentum=0.95,axis=-1)(Zc)
        Zc = LeakyReLU(alpha=0.1)(Zc)
        Zc = Dense(self.latentCdim,name="FxFWC")(Zc)
        Zc = BatchNormalization(momentum=0.95,axis=-1)(Zc)
        #Zc = tfa.layers.InstanceNormalization()(Zc)

        # variable n
        layer = self.nNlayers
        Zn = Flatten(name="FxFLN{:>d}".format(layer+1))(Zn)
        Zn = Dense(1024)(Zn)
        Zn = BatchNormalization(momentum=0.95)(Zn)
        Zn = LeakyReLU(alpha=0.1)(Zn)
        Zn = Dense(self.latentNdim,name="FxFWN")(Zn)

        # variable s
        s = SamplingFxNormSfromSigma()([Zmu,Zsigma])

        # variable c
        #c = Dense(self.latentCdim,activation=tf.keras.activations.softmax)(Zc)
        c = tf.keras.layers.Softmax()(Zc)

        # variable n
        n = BatchNormalization(momentum=0.95)(Zn)
        #n = tfa.layers.InstanceNormalization()(Zn)

        Fx = keras.Model(X,[Zmu,Zsigma,s,c,n],name="Fx")

        return Fx

    def BuildGz(self):
        """
            Conv1D Gz structure
            https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

        """
    
        s = Input(shape=(self.latentSdim,),name="s")
        c = Input(shape=(self.latentCdim,),name="c")
        n = Input(shape=(self.latentNdim,),name="n")

        layer = 0
        # variable s
        Zs = tfa.layers.SpectralNormalization(Dense(self.Ssize*self.nSchannels,
                                            name="GzFWS0"))(s)
        Zs = BatchNormalization(momentum=0.95)(Zs)
        #Zs = LeakyReLU(alpha=0.1)(Zs)
        Zs = ReLU()(Zs)
        Zs = Reshape((self.Ssize,self.nSchannels))(Zs)

        for layer in range(1,self.nSlayers):
            Zs = tfa.layers.SpectralNormalization(Conv1DTranspose(int(self.nSchannels*self.Sstride**(-layer)),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last"))(Zs)
            Zs = BatchNormalization(momentum=0.95)(Zs)
            Zs = ReLU()(Zs)
            #Zs = LeakyReLU(alpha=0.1)(Zs)
            #Zs = Dropout(0.2,name="GzDOS{:>d}".format(layer))(Zs)
        Zs = tfa.layers.SpectralNormalization(Conv1DTranspose(int(self.nSchannels*self.Sstride**(-self.nSlayers)),
            self.Skernel,self.Sstride,padding="same",
            data_format="channels_last"))(Zs)
        Zs = BatchNormalization(momentum=0.95,name="GzBNS{:>d}".format(self.nSlayers))(Zs)
        Zs = ReLU()(Zs)
        #Zs = LeakyReLU(alpha=0.1)(Zs)
        #Zs = Dropout(0.2)(Zs)
        GzS = keras.Model(s,Zs)


        # variable c
        Zc = tfa.layers.SpectralNormalization(Dense(self.Csize*self.nCchannels))(c)
        Zc = BatchNormalization(momentum=0.95)(Zc)
        Zc = ReLU()(Zc)
        #Zc = LeakyReLU(alpha=0.1,)(Zc)
        Zc = Reshape((self.Csize,self.nCchannels))(Zc)
        for layer in range(1,self.nClayers):
            Zc = tfa.layers.SpectralNormalization(Conv1DTranspose(int(self.nCchannels*self.Cstride**(-layer)),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last"))(Zc)
            #Zc = LeakyReLU(alpha=0.1)(Zc)
            Zc = BatchNormalization(momentum=0.95)(Zc)
            Zc = ReLU()(Zc)
            #Zc = Dropout(0.2)(Zc)
        Zc = tfa.layers.SpectralNormalization(Conv1DTranspose(int(self.nCchannels*self.Cstride**(-self.nClayers)),
            self.Ckernel,self.Cstride,padding="same",
            data_format="channels_last"))(Zc)
        Zc = BatchNormalization(momentum=0.95)(Zc)
        Zc = ReLU()(Zc)
        #Zc = LeakyReLU(alpha=0.1)(Zc)
        #Zc = Dropout(0.2)(Zc)
        GzC = keras.Model(c,Zc)

        # variable n
        Zn = tfa.layers.SpectralNormalization(Dense(self.Nsize*self.nNchannels))(n)
        Zn = BatchNormalization(momentum=0.95)(Zn)
        Zn = ReLU()(Zn)
        #Zn = LeakyReLU(alpha=0.1)(Zn)
        Zn = Reshape((self.Nsize,self.nNchannels))(Zn)
        for layer in range(1,self.nNlayers):
            Zn = tfa.layers.SpectralNormalization(Conv1DTranspose(int(self.nNchannels*self.Nstride**(-layer)),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last"))(Zn)
            Zn = BatchNormalization(momentum=0.95)(Zn)
            Zn = ReLU()(Zn)
            #Zn = LeakyReLU(alpha=0.1)(Zn)
            #Zn = Dropout(0.2)(Zn)
        Zn = tfa.layers.SpectralNormalization(Conv1DTranspose(int(self.nNchannels*self.Nstride**(-self.nNlayers)),
            self.Nkernel,self.Nstride,padding="same",
            data_format="channels_last"))(Zn)
        Zn = BatchNormalization(momentum=0.95)(Zn)
        Zn = ReLU()(Zn)
        #Zn = LeakyReLU(alpha=0.1)(Zn)
        #Zn = Dropout(0.2)(Zn)
        GzN = keras.Model(n,Zn)

        Gz = concatenate([GzS.output,GzC.output,GzN.output])
        Gz = tfa.layers.SpectralNormalization(Conv1DTranspose(self.nZchannels,
                self.kernel,1,padding="same",
                data_format="channels_last"))(Gz)
        Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)
        #Gz = LeakyReLU(alpha=0.1)(Gz)
        Gz = ReLU()(Gz)

        for layer in range(self.nAElayers-1):
            Gz = tfa.layers.SpectralNormalization(Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                self.kernel,self.stride,padding="same",use_bias=False))(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)
            Gz = ReLU()(Gz)
            #Gz = LeakyReLU(alpha=0.1)(Gz)

        layer = self.nAElayers-1
        Gz = tfa.layers.SpectralNormalization(Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                self.kernel,self.stride,padding="same",use_bias=False))(Gz)
        Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)
        Gz = ReLU()(Gz)
        #Gz = LeakyReLU(alpha=0.1,name="GzA{:>d}".format(layer+1))(Gz)

        layer = self.nAElayers
        X = tfa.layers.SpectralNormalization(Conv1DTranspose(self.nXchannels,self.kernel,1,
            padding="same",activation='tanh',use_bias=False))(Gz)

        Gz = keras.Model(inputs=[GzS.input,GzC.input,GzN.input],outputs=X,name="Gz")
        return Gz

    def BuildDx(self):
        """
            Conv1D discriminator structure
        """
        layer = 0
        X = Input(shape=self.Xshape,name="X")
        h = tfa.layers.SpectralNormalization(Conv1D(self.Xsize*self.stride**(-(layer+1)),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="DxCNN0"))(X)
        #h = LayerNormalization(axis=[1,2])(h) #temp
        h = LeakyReLU(alpha=0.1,name="DxA0")(h)
        h = Dropout(0.25)(h)
        
        for layer in range(1,self.nDlayers):
            h = tfa.layers.SpectralNormalization(Conv1D(self.Xsize*self.stride**(-(layer+1)),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="DxCNN{:>d}".format(layer)))(h)
            h = LayerNormalization(axis=[1,2],name="DxLN{:>d}".format(layer))(h) #temp
            h = LeakyReLU(alpha=0.1,name="DxA{:>d}".format(layer))(h)
            h = Dropout(0.25)(h)
            
        layer = self.nDlayers    
        h = Flatten(name="DxFL{:>d}".format(layer))(h)
        h = tfa.layers.SpectralNormalization(Dense(1024))(h)
        h = LayerNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(0.25)(h)
        Px = tfa.layers.SpectralNormalization(Dense(1,activation='sigmoid'))(h)
        Dx = keras.Model(X,Px,name="Dx")
        return Dx


    def BuildDc(self):
        """
            Dense discriminator structure
        """
        c = Input(shape=(self.latentCdim,))
        h = Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(c)
        #h = BatchNormalization(momentum=0.95)(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(0.25)(h)
        h = Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(h)
        #h = BatchNormalization(momentum=0.95)(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(0.25)(h)
        Pc = Dense(1,activation=tf.keras.activations.linear,
                                kernel_constraint=ClipConstraint(self.clipValue))(h)
        Dc = keras.Model(c,Pc,name="Dc")
        return Dc


    def BuildDn(self):
        """
            Dense discriminator structure
        """
        n = Input(shape=(self.latentNdim,))
        h = Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(n) 
        #h = BatchNormalization(momentum=0.95)(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(0.25)(h)
        h = Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(h)
        #h = BatchNormalization(momentum=0.95)(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(0.25)(h)
        Pn = Dense(1,activation=tf.keras.activations.linear,
                                kernel_constraint=ClipConstraint(self.clipValue))(h)
        Dn = keras.Model(n,Pn,name="Dn")
        return Dn

    def BuildDs(self):
        """
            Dense discriminator structure
        """
        s = Input(shape=(self.latentSdim,))
        h = Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(s) 
        #h = BatchNormalization(momentum=0.95)(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(0.25)(h)
        h = Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(h)
        #h = BatchNormalization(momentum=0.95)(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(0.25)(h)
        Ps = Dense(1,activation=tf.keras.activations.linear,
                                kernel_constraint=ClipConstraint(self.clipValue))(h)
        Ds = keras.Model(s,Ps,name="Ds")
        return Ds

    
    def DumpModels(self):
        self.Fx.save("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/01_06/Fx",save_format="tf")
        self.Gz.save("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/01_06/Gz",save_format="tf")
        self.Dx.save("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/01_06/Dx",save_format="tf")
        self.Ds.save("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/01_06/Ds",save_format="tf")
        self.Dn.save("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/01_06/Dn",save_format="tf")
        self.Dc.save("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/01_06/Dc",save_format="tf")
        return

def Main(DeviceName):

    options = ParseOptions()

    if not options['cuda']:
        DeviceName = "/cpu:0"

    with tf.device(DeviceName):
        optimizers = {}
        optimizers['DxOpt'] = Adam(learning_rate=0.002, beta_1=0.5, beta_2=0.9999)
        optimizers['DcOpt'] = RMSprop(learning_rate=0.002)#Adam(learning_rate=0.002, beta_1=0.5, beta_2=0.9999)
        optimizers['DsOpt'] = RMSprop(learning_rate=0.002)#Adam(learning_rate=0.002, beta_1=0.5, beta_2=0.9999)
        optimizers['DnOpt'] = RMSprop(learning_rate=0.002)#Adam(learning_rate=0.002, beta_1=0.5, beta_2=0.9999)
        optimizers['FxOpt'] = Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9999)
        optimizers['GzOpt'] = Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9999)

        losses = {}
        losses['AdvDlossDz'] = WassersteinLoss #WassersteinDiscriminatorLoss
        losses['AdvGlossDz'] = WassersteinLoss #WassersteinDiscriminatorLoss
        losses['AdvDlossDx'] = tf.keras.losses.BinaryCrossentropy()
        losses['AdvGlossDx'] = tf.keras.losses.BinaryCrossentropy()
        losses['RecSloss'] = GaussianNLL
        losses['RecXloss'] = tf.keras.losses.MeanAbsoluteError()
        losses['RecCloss'] = tf.keras.losses.CategoricalCrossentropy()


        losses['PenAdvXloss'] = 1.
        losses['PenAdvCloss'] = 1.
        losses['PenAdvSloss'] = 1.
        losses['PenAdvNloss'] = 1.
        losses['PenRecXloss'] = 1.
        losses['PenRecCloss'] = 1.
        losses['PenRecSloss'] = 1.


        # Instantiate the RepGAN model.
        GiorgiaGAN = RepGAN(options)

        # Compile the RepGAN model.
        GiorgiaGAN.compile(optimizers,losses) #run_eagerly=True

        GiorgiaGAN.build(input_shape=(options['batchSize'],options['Xsize'],
                                options['nXchannels']))

        GiorgiaGAN.compute_output_shape(input_shape=(options['batchSize'],options['Xsize'],
                                options['nXchannels']))


        if options['CreateData']:
            # Create the dataset
            Xtrn,  Xvld, _ = mdof.CreateData(**options)
        else:
            # Load the dataset
            Xtrn, Xvld, _ = mdof.LoadData(**options)

        #validation_data=Xvld
        history = GiorgiaGAN.fit(Xtrn,epochs=options["epochs"],
            callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}.ckpt", save_freq='epoch',period=500)]) #CustomLearningRateScheduler(schedule), NewCallback(p,epochs)

        GiorgiaGAN.DumpModels()

        PlotLoss(history) # Plot loss
        
        

        
if __name__ == '__main__':
    DeviceName = tf.test.gpu_device_name()
    Main(DeviceName)

