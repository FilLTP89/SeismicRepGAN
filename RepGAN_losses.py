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
import tensorflow.keras.losses as kl
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop

ε = 1e-8

class GaussianNLL(kl.Loss):
    """
        Gaussian negative loglikelihood loss function
    """
    def __init__(self, mod='var', raxis=None, λ=1.0):
        super(GaussianNLL,self).__init__()

        self.mod = mod
        self.raxis = raxis
        self.λ = λ
    
    @tf.function
    def call(self, x, μlogΣ):
                
        μ, logΣ = tf.split(μlogΣ,num_or_size_splits=2, axis=1)

        n_dims = int(x.shape[1])

        if not self.raxis:
            raxis = [i for i in range(1,len(x.shape))]
        else:
            raxis = self.raxis

        log2pi = 0.5*n_dims*tf.math.log(2.*np.pi)

        mse = tf.reduce_sum(0.5*tf.math.square(x-μ)*tf.math.exp(-logΣ),axis=raxis)
        traceΣ = tf.reduce_sum(tf.math.exp(logΣ), axis=raxis)
        NLL = mse+traceΣ+log2pi

        return self.λ*tf.reduce_mean(NLL)

class GANDiscriminatorLoss(kl.Loss):
    """
         General GAN Loss (for real and fake) with labels: {0,1}. 
         Logit output from D
    """
    def __init__(self, raxis=1, λ=1.0):
        super(GANDiscriminatorLoss,self).__init__()

        self.raxis = raxis
        self.λ = λ
    
    @tf.function
    def call(self,DX,DGz):
        
        if not self.raxis:
            raxis = [i for i in range(1, len(DGz.shape))]
        else:
            raxis = self.raxis
            
        real_loss = kl.BinaryCrossentropy(from_logits=True, axis=raxis)(tf.ones_like(DX), DX)
        fake_loss = kl.BinaryCrossentropy(from_logits=True, axis=raxis)(tf.zeros_like(DGz), DGz)
        
        return self.λ*(real_loss + fake_loss)

class GANGeneratorLoss(kl.Loss):
    """
         General GAN Loss (for real and fake) with labels: {0,1}.
         Logit output from D
    """
    def __init__(self, raxis=1, λ=1.0):
        super(GANGeneratorLoss,self).__init__()

        self.raxis = raxis
        self.λ = λ
        
    @tf.function
    def call(self, DX, DGz):

        if not self.raxis:
            raxis = [i for i in range(1, len(DGz.shape))]
        else:
            raxis = self.raxis
        
        real_loss = kl.BinaryCrossentropy(from_logits=True, axis=raxis)(tf.ones_like(DGz), DGz)
        return self.λ*real_loss

class HingeGANDiscriminatorLoss(kl.Loss):
    """
         Hinge GAN Loss (for real and fake) with labels: {0,1}.
         Logit output from D
    """
    def __init__(self, raxis=1, λ=1.0):
        super(HingeGANDiscriminatorLoss,self).__init__()

        self.raxis = raxis
        self.λ = λ

    @tf.function
    def call(self,logitsDX, logitsDGz):
        if not self.raxis:
            raxis = [i for i in range(1, len(DGz.shape))]
        else:
            raxis = self.raxis

        real_loss = tf.reduce_mean(tf.nn.relu(1. - logitsDX),axis=raxis)
        fake_loss = tf.reduce_mean(tf.nn.relu(1. + logitsDGz),axis=raxis)
        return self.λ*(real_loss + fake_loss)

class WGANDiscriminatorLoss(kl.Loss):
    """
        Compute standard WGAN loss (complete)
    """
    def __init__(self, raxis=1, λ=1.0):
        super(WGANDiscriminatorLoss,self).__init__()

        self.raxis = raxis
        self.λ = λ
        
    @tf.function
    def call(self, DX, DGz):
        
        if not self.raxis:
            raxis = [i for i in range(1, len(DGz.shape))]
        else:
            raxis = self.raxis
        
        EDX  = tf.reduce_mean(DX,axis=raxis)
        EDGz = tf.reduce_mean(DGz,axis=raxis)
        return self.λ*(EDGz-EDX)


class WGANGeneratorLoss(kl.Loss):
    """
        Compute standard WGAN loss (generator only)
    """

    def __init__(self, raxis=1, λ=1.0):
        super(WGANGeneratorLoss,self).__init__()

        self.raxis = raxis
        self.λ = λ

    @tf.function
    def call(self, DX, DGz):

        if not self.raxis:
            raxis = [i for i in range(1, len(DGz.shape))]
        else:
            raxis = self.raxis

        EDGz = tf.reduce_mean(DGz, axis=raxis)
        return -self.λ*EDGz

@tf.function
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


@tf.function
def WGANGPDiscriminatorLoss(DX, DGz, D, λ=1.0, λGP=1.0):
    "Wasserstrain loss with Gradient Penalty"
    Ls = WGANDiscriminatorLoss(DX, DGz)
    GP = GradientPenalty(DX, DGz, D)
    return λ*Ls+λGP*GP


@tf.function
def WGANGPGeneratorLoss(DX, DGz, λ=1.0):
    "Wasserstrain loss with Gradient Penalty"
    Ls = WGANGeneratorLoss(DX, DGz)
    return λ*Ls

class MutualInfoLoss(kl.Loss):
    """
        Mutual Information loss (InfoGAN)
    """
    def __init__(self, raxis=1, λ=1.0):
        super(MutualInfoLoss,self).__init__()

        self.raxis = raxis
        self.λ = λ
    
    @tf.function
    def call(self, c, c_given_x):
        
        if not self.raxis:
            raxis = [i for i in range(1, len(c_given_x.shape))]
        else:
            raxis = self.raxis
        
        # H_CgivenX = kl.
        H_CgivenX = -tf.reduce_mean(tf.reduce_mean(tf.math.log(c_given_x+ε)*c,axis=raxis))
        H_C = -tf.reduce_mean(tf.reduce_mean(tf.math.log(c+ε)*c,axis=raxis))
        
        return self.λ*(H_CgivenX - H_C)


class InfoLoss(kl.Loss):
    """
        Categorical cross entropy loss as Information loss (InfoGAN)
    """
    def __init__(from_logits=False, λ=1.0):
        super(InfoLoss,self).__init__()

        self.from_logits = from_logits
        self.λ = λ
        
    @tf.function
    def call(self, DX, DGz):
        return self.λ*kl.CategoricalCrossentropy(from_logits=self.from_logits)(DX, DGz)

def getOptimizers(**kwargs):
    getOptimizers.__globals__.update(kwargs)
    optimizers = {}
    
    if DxTrainType.upper() == "WGAN" or DxTrainType.upper() == "WGANSN":
        optimizers['DxOpt'] = RMSprop(learning_rate=DxLR)
    else:
        optimizers['DxOpt'] = Adam(learning_rate=DxLR, beta_1=0.5, beta_2=0.9999)
        
    if DcTrainType.upper() == "WGAN" or DcTrainType.upper() == "WGANSN":
        optimizers['DcOpt'] = RMSprop(learning_rate=DcLR)        
    else:
        optimizers['DcOpt'] = Adam(learning_rate=DcLR, beta_1=0.5, beta_2=0.9999)
        
    if DsTrainType.upper() == "WGAN" or DsTrainType.upper() == "WGANSN":    
        optimizers['DsOpt'] = RMSprop(learning_rate=DsLR)
    else:
        optimizers['DsOpt'] = Adam(learning_rate=DsLR, beta_1=0.5, beta_2=0.9999)
        
    if DnTrainType.upper() == "WGAN" or DnTrainType.upper() == "WGANSN":
        optimizers['DnOpt'] = RMSprop(learning_rate=DnLR)
    else:
        optimizers['DnOpt'] = Adam(learning_rate=DnLR, beta_1=0.5, beta_2=0.9999)

        
    optimizers['FxOpt'] = Adam(learning_rate=FxLR, beta_1=0.5, beta_2=0.9999)
    optimizers['GzOpt'] = Adam(learning_rate=GzLR, beta_1=0.5, beta_2=0.9999)
    return optimizers

def getLosses(**kwargs):
    getLosses.__globals__.update(kwargs)
    losses = {}
    
    if DcTrainType.upper() == "WGAN":
        losses['AdvDlossDc'] = WGANDiscriminatorLoss(λ=PenAdvNloss)
        losses['AdvGlossDc'] = WGANGeneratorLoss(λ=PenAdvNloss)
    elif DcTrainType.upper() == "WGANGP":
        losses['AdvDlossDc'] = WGANGPDiscriminatorLoss(λ=PenAdvNloss)
        losses['AdvGlossDc'] = WGANGPGeneratorLoss(λ=PenAdvNloss)
    elif DcTrainType.upper() == "GAN":
        losses['AdvDlossDc'] = GANDiscriminatorLoss(λ=PenAdvNloss)
        losses['AdvGlossDc'] = GANGeneratorLoss(λ=PenAdvNloss)
    elif DcTrainType.upper() == "HINGE":
        losses['AdvDlossDc'] = HingeGANDiscriminatorLoss(λ=PenAdvNloss)
        losses['AdvGlossDc'] = GANGeneratorLoss(λ=PenAdvNloss)

    if DsTrainType.upper() == "WGAN":
        losses['AdvDlossDs'] = WGANDiscriminatorLoss(λ=PenAdvSloss)
        losses['AdvGlossDs'] = WGANGeneratorLoss(λ=PenAdvSloss)
    elif DsTrainType.upper() == "WGANGP":
        losses['AdvDlossDs'] = WGANGPDiscriminatorLoss(λ=PenAdvSloss)
        losses['AdvGlossDs'] = WGANGPGeneratorLoss(λ=PenAdvSloss)
    elif DsTrainType.upper() == "GAN":
        losses['AdvDlossDs'] = GANDiscriminatorLoss(λ=PenAdvSloss)
        losses['AdvGlossDs'] = GANGeneratorLoss(λ=PenAdvSloss)
    elif DsTrainType.upper() == "HINGE":
        losses['AdvDlossDs'] = HingeGANDiscriminatorLoss(λ=PenAdvSloss)
        losses['AdvGlossDs'] = GANGeneratorLoss(λ=PenAdvSloss)
        
    if DnTrainType.upper() == "WGAN":
        losses['AdvDlossDn'] = WGANDiscriminatorLoss(λ=PenAdvNloss)
        losses['AdvGlossDn'] = WGANGeneratorLoss(λ=PenAdvNloss)
    elif DnTrainType.upper() == "WGANGP":
        losses['AdvDlossDn'] = WGANGPDiscriminatorLoss(λ=PenAdvNloss)
        losses['AdvGlossDn'] = WGANGPGeneratorLoss(λ=PenAdvNloss)
    elif DnTrainType.upper() == "GAN":
        losses['AdvDlossDn'] = GANDiscriminatorLoss(λ=PenAdvNloss)
        losses['AdvGlossDn'] = GANGeneratorLoss(λ=PenAdvNloss)
    elif DnTrainType.upper() == "HINGE":
        losses['AdvDlossDn'] = HingeGANDiscriminatorLoss(λ=PenAdvNloss)
        losses['AdvGlossDn'] = GANGeneratorLoss(λ=PenAdvNloss)

    if DxTrainType.upper() == "WGAN":
        losses['AdvDlossDx'] = WGANDiscriminatorLoss(λ=PenAdvXloss)
        losses['AdvGlossDx'] = WGANGeneratorLoss(λ=PenAdvXloss)
    elif DxTrainType.upper() == "WGANGP":
        losses['AdvDlossDx'] = WGANGPDiscriminatorLoss(λ=PenAdvXloss)
        losses['AdvGlossDx'] = WGANGPGeneratorLoss(λ=PenAdvXloss)
    elif DxTrainType.upper() == 'GAN':
        losses['AdvDlossDx'] = GANDiscriminatorLoss(λ=PenAdvXloss)
        losses['AdvGlossDx'] = GANGeneratorLoss(λ=PenAdvXloss)
    elif DxTrainType.upper() == "HINGE":
        raise Exception("Hinge loss not implemented for Dx")

    losses['RecSloss']  = GaussianNLL(λ=PenRecSloss)
    losses['RecXloss']  = kl.MeanSquaredError()
    losses['RecCloss']  = MutualInfoLoss(λ=PenRecCloss)
    losses['FakeCloss'] = kl.CategoricalCrossentropy()

    return losses