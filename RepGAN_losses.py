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
from tensorflow.keras.optimizers import Adam, RMSprop

ε = 1e-8

def GaussianNLL(x,μ,logσ2,raxis=None):
    """
        Gaussian negative loglikelihood loss function
    """
    n_dims = int(x.shape[1])

    if not raxis:
        raxis = [i for i in range(1,len(x.shape))]

    log2pi = -0.5*n_dims*tf.math.log(2.*np.pi)

    # if 'var' in mod:
    #   Σ = tf.math.log(Σ+ε)

    σ2 = tf.exp(logσ2)

    # mse = -0.5*tf.square(x-μ)*tf.exp(-Σ)
    #traceΣ = -tf.reduce_sum(Σ,axis=raxis)
    mse = -0.5*tf.square(x-μ)/(tf.math.maximum(σ2,ε))
    traceΣ = -0.5*tf.reduce_sum(tf.math.log(tf.math.maximum(σ2,ε)),axis=raxis)
    NLL = tf.reduce_sum(mse,axis=raxis)+traceΣ+log2pi

    # mse = -0.5*tf.reduce_sum(tf.keras.backend.square((x-μ))/sigma,axis=raxis) 
    # sigma_trace = -0.5*tf.reduce_sum(tf.math.log(sigma), axis=raxis)
    # NLL = mse+sigma_trace+log2pi

    return tf.reduce_mean(-NLL)

def GANDiscriminatorLoss(X, Gz):
    # General GAN Loss (for real and fake) with labels: {0,1}. Sigmoid output from D
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = bce_loss(tf.ones_like(X), X)
    fake_loss = bce_loss(tf.zeros_like(Gz), Gz)
    return real_loss + fake_loss

def GANGeneratorLoss(logitsDGz):
    """Generator GAN Loss for generator from logits"""
    return -tf.reduce_mean(logitsDGz)

def HingeDGANLoss(logitsDX, logitsDGz):
    real_loss = tf.reduce_mean(tf.nn.relu(1. - logitsDX))
    fake_loss = tf.reduce_mean(tf.nn.relu(1. + logitsDGz))
    return real_loss + fake_loss

def WassersteinDiscriminatorLoss(X, Gz):
    """Compute standard Wasserstein loss (complete)"""
    real_loss = tf.reduce_mean(X)
    fake_loss = tf.reduce_mean(Gz)
    return fake_loss - real_loss

def WassersteinGeneratorLoss(Gz):
    """Compute standard Wasserstein loss (generator only)"""
    return -tf.reduce_mean(Gz)

def WassersteinLoss(s, Gz):
    """General WGAN Loss (for real and fake) with labels s:={-1,1}. 
    Logit output from D
    Adapted to work with multiple output discriminator (e.g.: PatchGAN Discriminator)
    get axis to reduce (reduce for output sample) [b,a,b,c,..] -> [b,1]"""
    raxis = [i for i in range(1,len(Gz.shape))]
    # reduce to average for global batch (reduce for output sample) [b,1] -> [b,1]
    return s*tf.reduce_mean(Gz,axis=raxis)


def GradientPenalty(X, Gz, D):
    """Compute the gradient penalty of discriminator D"""
    # Get the interpolated image
    α = tf.random.normal([X.shape[0], 1, 1], 0.0, 1.0)
    δ = Gz - X
    δ = Gz - X
    XδX = X + α*δ

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(XδX)
        # 1. Get the discriminator output for this interpolated image.
        DXδX = D(XδX, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    GradD = gp_tape.gradient(DXδX, [XδX])[0]
    # 3. Calculate the norm of the gradients.
    NormGradD = tf.sqrt(tf.reduce_sum(tf.square(GradD), axis=[1]))
    GPloss = tf.reduce_mean((NormGradD - 1.0) ** 2)
    return GPloss

def WassersteinLossGP(X, Gz, D, λ=1.0):
    "Wasserstrain loss with Gradient Penalty"
    Ls=WassersteinLoss(X,Gz)
    GP=GradientPenalty(X, Gz, D)
    return Ls+λ*GP

def MutualInfoLoss(c, c_given_x,raxis=1):
    """The mutual information metric we aim to minimize"""
    H_CgivenX = -tf.reduce_mean(tf.reduce_mean(tf.math.log(c_given_x+ε)*c,axis=raxis))
    H_C = -tf.reduce_mean(tf.reduce_mean(tf.math.log(c+ε)*c,axis=raxis))
    return H_CgivenX - H_C

# Info Loss for Q (GANPatch adapted)
def InfoLoss(X, Gz):
    return tf.keras.losses.CategoricalCrossentropy(X, Gz)

def getOptimizers(**kwargs):
    getOptimizers.__globals__.update(kwargs)
    optimizers = {}
    optimizers['DxOpt'] = Adam(learning_rate=DxLR, beta_1=0.5, beta_2=0.9999)
    if 'WGAN' in DzTrainType:
        optimizers['DcOpt'] = RMSprop(learning_rate=DcLR)
        optimizers['DsOpt'] = RMSprop(learning_rate=DsLR)
        optimizers['DnOpt'] = RMSprop(learning_rate=DnLR)
    else:
        optimizers['DcOpt'] = Adam(learning_rate=DcLR, beta_1=0.5, beta_2=0.9999)
        optimizers['DsOpt'] = Adam(learning_rate=DsLR, beta_1=0.5, beta_2=0.9999)
        optimizers['DnOpt'] = Adam(learning_rate=DnLR, beta_1=0.5, beta_2=0.9999) 
    optimizers['FxOpt'] = Adam(learning_rate=FxLR, beta_1=0.5, beta_2=0.9999)
    optimizers['GzOpt'] = Adam(learning_rate=GzLR, beta_1=0.5, beta_2=0.9999)
    return optimizers

def getLosses(**kwargs):
    getLosses.__globals__.update(kwargs)
    losses = {}
    
    losses['PenAdvXloss'] = 1.
    losses['PenAdvCloss'] = 1.
    losses['PenAdvSloss'] = 1.
    losses['PenAdvNloss'] = 1.
    losses['PenRecXloss'] = 10.
    losses['PenRecCloss'] = 1.
    losses['PenRecSloss'] = 1.
    
    if DzTrainType.upper() == "WGAN":
        losses['AdvDlossDz'] = WassersteinLoss 
        losses['AdvGlossDz'] = WassersteinLoss 
    elif DzTrainType.upper() == "WGANGP":
        losses['AdvDlossDz'] = WassersteinLoss
        losses['AdvGlossDz'] = WassersteinLoss
    elif DzTrainType.upper() == "GAN":
        losses['AdvDlossDz'] = tf.keras.losses.BinaryCrossentropy()
        losses['AdvGlossDz'] = tf.keras.losses.BinaryCrossentropy()
    elif DzTrainType.upper() == "HINGE":
        losses['AdvDlossDz'] = HingeDGANLoss
        losses['AdvGlossDz'] = GANGeneratorLoss

    if DxTrainType.upper() == "WGAN":
        losses['AdvDlossDx'] = WassersteinLoss
        losses['AdvGlossDx'] = WassersteinLoss
    elif DxTrainType.upper() == "WGANGP":
        losses['AdvDlossDx'] = WassersteinLoss
        losses['AdvGlossDx'] = WassersteinLoss
    elif DxTrainType.upper() == 'GAN':
        losses['AdvDlossDx'] = tf.keras.losses.BinaryCrossentropy()
        losses['AdvGlossDx'] = tf.keras.losses.BinaryCrossentropy()
    elif DxTrainType.upper() == "HINGE":
        raise Exception("Hinge loss not implemented for Dx")

    losses['RecSloss'] = GaussianNLL
    losses['RecXloss'] = tf.keras.losses.MeanSquaredError()
    losses['RecCloss'] = MutualInfoLoss
    losses['FakeCloss'] = tf.keras.losses.CategoricalCrossentropy()

    return losses