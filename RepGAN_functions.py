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

import tensorflow as tf
import numpy as np

import argparse
import math as mt

from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
import tensorflow_addons as tfa
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Layer
from tensorflow.keras.layers import Lambda, Concatenate, concatenate, ZeroPadding1D, Add
from tensorflow.keras.layers import LeakyReLU, ReLU, Softmax, Activation, Average, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.constraints import Constraint, min_max_norm

import tensorflow_probability as tfp

from RepGAN_losses import GaussianNLL,WassersteinLoss

class SamplingFxS(Layer):
    """Uses (z_mean, z_std) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_std = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + z_std * epsilon

class SamplingFxNormSfromLogVariance(Layer):

    def call(self, inputs):
        z_mean, z_logvar = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean),mean=0.0,stddev=1.0)
        z = z_mean + tf.math.multiply(tf.exp(z_logvar * .5),epsilon)

        return z

class SamplingFxNormSfromSigma(Layer):

    def call(self, inputs):
        z_mean, z_std = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
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
        clip_value = clip_value
    # clip model weights to hypercube
    def __call__(self, weights):
        if clip_value is not None:
            return tf.keras.backend.clip(weights, -clip_value, clip_value)
        else:             
            return tf.keras.backend.clip(weights, None, None)

    # get the config
    def get_config(self):
        return {'clip_value': clip_value}


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    idx = np.random.randint(low=0, high=dataset.shape[0], size=n_samples)
    X = tf.gather(dataset, idx)

    return X

class RandomWeightedAverage(Layer):
    """Provides a (random) weighted average between real and generated signal samples"""
    def _merge_function(self,inputs,**kwargs):
        alpha = tf.random_uniform((32,1,1,1))
        return (alpha*inputs[0])+((1.0-alpha)*inputs[1])

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
        predX = Dx(intX,training=True)

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
        predX = Ds(intX,training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    GradX = gp_tape.gradient(predX, [intX])[0]
    # 3. Calculate the norm of the gradients.
    NormGradX = tf.sqrt(tf.reduce_sum(tf.square(GradX), axis=[1]))
    gp = tf.reduce_mean((NormGradX - 1.0) ** 2)
    return gp

def train_XZX(X,c):
    
    # Create labels for BCE in GAN loss
    realBCE_C = tf.ones((batchSize,1), dtype=tf.float32)
    fakeBCE_C = -tf.ones((batchSize,1), dtype=tf.float32)
    realBCE_S = tf.ones((batchSize,1), dtype=tf.float32)
    fakeBCE_S = -tf.ones((batchSize,1), dtype=tf.float32)
    realBCE_N = tf.ones((batchSize,1), dtype=tf.float32)
    fakeBCE_N = -tf.ones((batchSize,1), dtype=tf.float32)

    # Train generators
    Fx.trainable = True
    Gz.trainable = True
    Dx.trainable = False
    Dc.trainable = False
    Ds.trainable = False
    Dn.trainable = False

    # Train nGenerator times the generators
    for _ in range(nGenerator):

        with tf.GradientTape(persistent=True) as tape:

            # Encode real signals
            [μs_fake,Σs_fake,c_fake,n_fake] = Fx(X,training=True)

            # variable s
            s_fake = SamplingFxNormSfromLogVariance()([μs_fake,Σs_fake])  

            # Reconstruct real signals
            X_rec = Gz((s_fake,c_fake,n_fake),training=True)

            #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_rec, labels=X)
            #logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
            logpz = GaussianNLL(s_fake, tf.zeros((batchSize,1), dtype=tf.float32), tf.zeros((batchSize,1), dtype=tf.float32),mod='logvar')
            logqz_x = GaussianNLL(s_fake, μs_fake, Σs_fake,mod='logvar')

            # Compute reconstruction loss
            #RecGlossX = RecXloss(X,X_rec)*PenRecXloss
            RecGlossX = (RecXloss(X,X_rec) -tf.reduce_mean(logpz - logqz_x))*PenRecXloss #-tf.reduce_mean(logpx_z + logpz - logqz_x)
    
        # Get the gradients w.r.t the generator loss
        gradFx, gradGz = tape.gradient(RecGlossX,
        (Fx.trainable_variables,Gz.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Update the weights of the generator using the generator optimizer
        FxOpt.apply_gradients(zip(gradFx,Fx.trainable_variables))
        GzOpt.apply_gradients(zip(gradGz,Gz.trainable_variables))

    # Train discriminators
    Fx.trainable = False
    Gz.trainable = False
    Dx.trainable = False
    Dc.trainable = True
    Ds.trainable = True
    Dn.trainable = True

    # Train nCritic times the discriminators
    for _ in range(nCritic):
        
        # Sample factorial prior S
        s_prior = tf.random.normal(mean=0.0,stddev=1.0,
            shape=[batchSize,latentSdim],
            dtype=tf.float32)

        # Sample factorial prior N
        n_prior = tf.random.normal(mean=0.0,stddev=1.0,
            shape=[batchSize,latentNdim],
            dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:

            # Encode real signals X
            [μs_fake,Σs_fake,c_fake,n_fake] = Fx(X,training=True)

            s_fake = SamplingFxNormSfromLogVariance()([μs_fake,Σs_fake])

            # Discriminates real and fake S
            s_fakecritic = Ds(s_fake,training=True)
            s_priorcritic = Ds(s_prior,training=True)

            # Discriminates real and fake N
            n_fakecritic = Dn(n_fake,training=True)
            n_priorcritic = Dn(n_prior,training=True)

            # Discriminates real and fake C
            c_fakecritic = Dc(c_fake,training=True)
            c_critic = Dc(c,training=True)

            # Compute XZX adversarial loss (JS(s),JS(n),JS(c))
            AdvDlossC  = AdvDlossDz(realBCE_C,c_critic)*PenAdvCloss
            AdvDlossC += AdvDlossDz(fakeBCE_C,c_fakecritic)*PenAdvCloss
            AdvDlossS  = AdvDlossDz(realBCE_S,s_priorcritic)*PenAdvSloss
            AdvDlossS += AdvDlossDz(fakeBCE_S,s_fakecritic)*PenAdvSloss
            AdvDlossN  = AdvDlossDz(realBCE_N,n_priorcritic)*PenAdvNloss
            AdvDlossN += AdvDlossDz(fakeBCE_N,n_fakecritic)*PenAdvNloss
            
            AdvDloss = 0.5*(AdvDlossC + AdvDlossS + AdvDlossN)

        # Compute the discriminator gradient
        gradDc, gradDs, gradDn = tape.gradient(AdvDloss,
                (Dc.trainable_variables,Ds.trainable_variables, 
                Dn.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Update discriminators' weights
        DcOpt.apply_gradients(zip(gradDc,Dc.trainable_variables))
        DsOpt.apply_gradients(zip(gradDs,Ds.trainable_variables))
        DnOpt.apply_gradients(zip(gradDn,Dn.trainable_variables))
    

    Fx.trainable = True
    Gz.trainable = False
    Dx.trainable = False
    Dc.trainable = False
    Ds.trainable = False
    Dn.trainable = False

    with tf.GradientTape(persistent=True) as tape:

        # Encode real signals
        μs_fake,Σs_fake,c_fake,n_fake = Fx(X,training=True)

        s_fake = SamplingFxNormSfromLogVariance()([μs_fake,Σs_fake])

        # Discriminate fake latent space
        s_fakecritic = Ds(s_fake,training=True)
        c_fakecritic = Dc(c_fake,training=True)
        n_fakecritic = Dn(n_fake,training=True)

        # Compute adversarial loss for generator
        AdvGlossC = AdvGlossDz(realBCE_C,c_fakecritic)*PenAdvCloss
        AdvGlossS = AdvGlossDz(realBCE_S,s_fakecritic)*PenAdvSloss
        AdvGlossN = AdvGlossDz(realBCE_N,n_fakecritic)*PenAdvNloss

        # Compute total generator loss
        AdvGloss = AdvGlossC + AdvGlossS + AdvGlossN
        
    # Get the gradients w.r.t the generator loss
    gradFx = tape.gradient(AdvGloss,(Fx.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

    # Update the weights of the generator using the generator optimizer
    FxOpt.apply_gradients(zip(gradFx,Fx.trainable_variables))

    return RecGlossX,AdvDloss,AdvDlossC,AdvDlossS,AdvDlossN,AdvGloss,AdvGlossC,AdvGlossS,AdvGlossN,\
            c_fakecritic,s_fakecritic,n_fakecritic,c_critic,s_priorcritic,n_priorcritic

def train_ZXZ(self,X,c):

    # Create labels for BCE in GAN loss
    realBCE_X = tf.ones((batchSize,1), dtype=tf.float32)
    fakeBCE_X = tf.zeros((batchSize,1), dtype=tf.float32)

    # Train discriminators
    Fx.trainable = False
    Gz.trainable = False
    Dx.trainable = True

    # Train nCritic times the discriminators
    for _ in range(nCritic):

        # Sample factorial prior S
        s_prior = tf.random.normal(mean=0.0,stddev=1.0,
            shape=[batchSize,latentSdim],
            dtype=tf.float32)

        # Sample factorial prior N
        n_prior = tf.random.normal(mean=0.0,stddev=1.0,
            shape=[batchSize,latentNdim],
            dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:

            # Decode factorial prior
            X_fake = Gz((s_prior,c,n_prior),training=True)

            # Discriminate real and fake X
            X_fakecritic = Dx(X_fake,training=True)
            X_critic = Dx(X,training=True)

            # Compute the discriminator loss GAN loss (penalized)
            AdvDlossX = AdvDlossDx(fakeBCE_X,X_fakecritic)*PenAdvXloss
            AdvDlossX += AdvDlossDx(realBCE_X,X_critic)*PenAdvXloss

        # Compute the discriminator gradient
        gradDx = tape.gradient(AdvDlossX,Dx.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        # Update the weights of the discriminator using the discriminator optimizer
        DxOpt.apply_gradients(zip(gradDx,Dx.trainable_variables))

    
    # Train generators
    Fx.trainable = True
    Gz.trainable = True
    Dx.trainable = False
    
    # Train nGenerator times the generators
    for _ in range(nGenerator):

        # Sample factorial prior S
        s_prior = tf.random.normal(mean=0.0,stddev=1.0,
            shape=[batchSize,latentSdim],
            dtype=tf.float32)
        
        # Sample factorial prior N
        n_prior = tf.random.normal(mean=0.0,stddev=1.0,
            shape=[batchSize,latentNdim],
            dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:

            # Decode factorial prior
            X_fake = Gz((s_prior,c,n_prior),training=True)

            # Discriminate real and fake X
            X_fakecritic = Dx(X_fake,training=True)

            # Compute adversarial loos (penalized)
            AdvGlossX = AdvGlossDx(realBCE_X,X_fakecritic)*PenAdvXloss
            
            # Encode fake signals
            [μs_rec,Σs_rec,c_rec,_] = Fx(X_fake,training=True)

            s_rec = SamplingFxNormSfromLogVariance()([μs_rec,Σs_rec])

            #Q_cont_distribution = tfp.distributions.MultivariateNormalDiag(loc=μs_rec, scale_diag=tf.exp(Σs_rec))
            #RecGlossS = -tf.reduce_mean(Q_cont_distribution.log_prob(s_prior))
            logpz = GaussianNLL(s_rec, tf.zeros((batchSize,1), dtype=tf.float32), tf.zeros((batchSize,1), dtype=tf.float32),mod='logvar')
            logqz_x = GaussianNLL(s_rec, μs_rec, Σs_rec,mod='logvar')
            RecGlossS = -tf.reduce_mean(logpz - logqz_x)*PenRecSloss
            #RecGlossS = RecSloss(s_prior,μs_rec,σs_rec)*PenRecSloss
            RecGlossC = RecCloss(c,c_rec)*PenRecCloss #+ RecCloss(c,c_fake)*PenRecCloss

            # Compute InfoGAN Q loos
            Qloss = RecGlossS + RecGlossC

        # Get the gradients w.r.t the generator loss
        gradGz = tape.gradient(AdvGlossX,(Gz.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Get the gradients w.r.t the generator loss
        gradFx, gradGz = tape.gradient(Qloss,
            (Fx.trainable_variables,Gz.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Update the weights of the generator using the generator optimizer
        GzOpt.apply_gradients(zip(gradGz,Gz.trainable_variables))

        # Update the weights of the generator using the generator optimizer
        FxOpt.apply_gradients(zip(gradFx,Fx.trainable_variables))
        GzOpt.apply_gradients(zip(gradGz,Gz.trainable_variables))

    return AdvDlossX,AdvGlossX,RecGlossS,RecGlossC,Qloss,X_fakecritic,X_critic


def plot(self,X,c):
    [μs_fake,Σs_fake,c_fake,n_fake] = Fx(X,training=False)
    s_fake = SamplingFxNormSfromLogVariance()([μs_fake,Σs_fake])
    s_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],latentSdim])
    n_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],latentNdim])
    fakeX = Gz((s_prior,c,n_prior),training=False)
    X_rec = Gz((s_fake,c_fake,n_fake),training=False)
    return X_rec, c_fake, s_fake, n_fake, fakeX

def label_predictor(self, X, c):
    [μs_fake,Σs_fake,c_fake,n_fake] = Fx(X)
    s_fake = SamplingFxNormSfromLogVariance()([μs_fake,Σs_fake])
    s_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[s_fake.shape[0],latentSdim])
    n_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[n_fake.shape[0],latentNdim])
    fakeX = Gz((s_prior,c,n_prior),training=False)
    [_,_,_,c_rec,_] = Fx(fakeX,training=False)
    return c_fake, c_rec

def distribution(self,X,c):
    [μs_fake,Σs_fake,c_fake,n_fake] = Fx(X,training=False)
    s_fake = SamplingFxNormSfromLogVariance()([μs_fake,Σs_fake])
    s_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],latentSdim])
    n_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],latentNdim])
    fakeX = Gz((s_prior,c,n_prior),training=False)
    [μs_rec,σs_rec,s_rec,c_rec,n_rec] = Fx(fakeX,training=False)
    return s_prior, n_prior, s_fake, n_fake, s_rec, n_rec, μs_fake, Σs_fake, μs_rec, σs_rec

def generate(self, X, c_fake_new):
    [μs_fake,Σs_fake,c_fake,n_fake] = Fx(X)
    s_fake = SamplingFxNormSfromLogVariance()([μs_fake,Σs_fake])
    X_rec_new = Gz((s_fake,c_fake_new,n_fake),training=False)
    return X_rec_new

def DumpModels(self,resultFolder):
    Fx.save("{:>s}/Fx",save_format="tf")
    Gz.save("{:>s}/Gz",save_format="tf")
    Dx.save("{:>s}/Dx",save_format="tf")
    Ds.save("{:>s}/Ds",save_format="tf")
    Dn.save("{:>s}/Dn",save_format="tf")
    Dc.save("{:>s}/Dc",save_format="tf")
    return

