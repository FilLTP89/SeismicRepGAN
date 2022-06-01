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

def GaussianNLL(x,μ,Σ,raxis=None,mod='var'):
    """
        Gaussian negative loglikelihood loss function
    """
    n_dims = int(x.shape[1])
    if not raxis:
        raxis = [i for i in range(1,len(x.shape))]
    log2pi = 0.5*n_dims*tf.math.log(2.*np.pi)
    if 'var' in mod:
        Σ = tf.log(Σ)

    mse = -0.5*tf.square(x-μ)*tf.exp(-Σ)
    traceΣ = tf.reduce_sum(Σ,axis=raxis)

    NLL = tf.reduce_sum(mse,axis=raxis)+traceΣ+log2pi

    return tf.reduce_mean(NLL)

def GANLoss(y_true, y_predict):
    # General GAN Loss (for real and fake) with labels: {0,1}. Sigmoid output from D
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = bce_loss(tf.ones_like(y_true), y_true)
    fake_loss = bce_loss(tf.zeros_like(y_predict), y_predict)
    return real_loss + fake_loss

def WassersteinDiscriminatorLoss(y_true, y_fake):
    real_loss = tf.reduce_mean(y_true)
    fake_loss = tf.reduce_mean(y_fake)
    return fake_loss - real_loss

def WassersteinGeneratorLoss(y_fake):
    return -tf.reduce_mean(y_fake)

def WassersteinLoss(y_true, y_predict):
    # General WGAN Loss (for real and fake) with labels: {-1,1}. Linear output from D
    # Adapted to work with multiple output discriminator (e.g.: PatchGAN Discriminator)
    # get axis to reduce (reduce for output sample) [b,a,b,c,..] -> [b,1]
    raxis = [i for i in range(1,len(y_predict.shape))]
    # reduce to average for global batch (reduce for output sample) [b,1] -> [b,1]
    return y_true*tf.reduce_mean(y_predict,axis=raxis)

def MutualInfoLoss(c, c_given_x):
    """The mutual information metric we aim to minimize"""
    eps = 1e-8
    conditional_entropy = -tf.keras.backend.mean(tf.keras.backend.sum(tf.math.log(c_given_x+eps)*c,axis=1))
    entropy = -tf.keras.backend.mean(tf.keras.backend.sum(tf.math.log(c+eps)*c,axis=1))

    return conditional_entropy - entropy

# Info Loss for Q (GANPatch adapted)
def InfoLoss(y_true, y_predict):
    return tf.keras.losses.CategoricalCrossentropy(y_true, y_predict)