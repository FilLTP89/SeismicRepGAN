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
import tensorflow as tf
import tensorflow.keras.losses as kl
import tensorflow.keras.callbacks as kc
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop

ε = 1e-8

class GaussianNLL(kl.Loss):
    """
        Gaussian negative loglikelihood loss function
    """
    def __init__(self, mod='var', raxis=None, λ=1.0, name="GaussianNLL"):
        super(GaussianNLL,self).__init__(name=name)

        self.mod = mod
        self.raxis = raxis
        self.λ = λ
    
    #@tf.function
    def call(self, x, μlogΣ):
                
        μ, logΣ = tf.split(μlogΣ,num_or_size_splits=2, axis=1)

        n_dims = int(x.shape[1])

        if not self.raxis:
            raxis = [i for i in range(1,len(x.shape))]
        else:
            raxis = self.raxis

        nlog2π = 0.5*n_dims*tf.math.log(2.*np.pi)

        mse = tf.reduce_sum(0.5*tf.math.square(x-μ)*tf.math.exp(-logΣ),axis=raxis)
        TrlogΣ = tf.reduce_sum(logΣ, axis=raxis)
        NLL = mse+TrlogΣ+nlog2π

        return self.λ*tf.reduce_mean(NLL)

class GANDiscriminatorLoss(kl.Loss):
    """
         General GAN Loss (for real and fake) with labels: {0,1}. 
         Logit output from D
    """
    def __init__(self, from_logits=True, raxis=1, λ=1.0, name="GANDiscriminatorLoss"):
        super(GANDiscriminatorLoss,self).__init__(name=name)

        self.from_logits = from_logits
        self.raxis = raxis
        self.λ = λ
    
    #@tf.function
    def call(self,DX,DGz):
        
        if not self.raxis:
            raxis = [i for i in range(1, len(DGz.shape))]
        else:
            raxis = self.raxis
            
        real_loss = kl.BinaryCrossentropy(from_logits=self.from_logits,
                                          axis=raxis)(tf.ones_like(DX), DX)
        fake_loss = kl.BinaryCrossentropy(from_logits=self.from_logits,
                                          axis=raxis)(tf.zeros_like(DGz), DGz)
        
        return self.λ*(real_loss + fake_loss)

class GANGeneratorLoss(kl.Loss):
    """
         General GAN Loss (for real and fake) with labels: {0,1}.
         Logit output from D
    """
    def __init__(self, from_logits=True, raxis=1, λ=1.0, name="GANGeneratorLoss"):
        super(GANGeneratorLoss,self).__init__(name=name)

        self.from_logits = from_logits
        self.raxis = raxis
        self.λ = λ
        
    #@tf.function
    def call(self, DX, DGz):

        if not self.raxis:
            raxis = [i for i in range(1, len(DGz.shape))]
        else:
            raxis = self.raxis
        
        real_loss = kl.BinaryCrossentropy(from_logits=self.from_logits, 
                                          axis=raxis)(tf.ones_like(DGz), DGz)
        return self.λ*real_loss

class HingeGANDiscriminatorLoss(kl.Loss):
    """
         Hinge GAN Loss (for real and fake) with labels: {0,1}.
         Logit output from D
    """
    def __init__(self, raxis=1, λ=1.0, name="HingeGANDiscriminatorLoss"):
        super(HingeGANDiscriminatorLoss,self).__init__(name=name)

        self.raxis = raxis
        self.λ = λ

    #@tf.function
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
    def __init__(self, raxis=1, λ=1.0, name="WGANDiscriminatorLoss"):
        super(WGANDiscriminatorLoss,self).__init__(name=name)

        self.raxis = raxis
        self.λ = λ
        
    # #@tf.function
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

    def __init__(self, raxis=1, λ=1.0, name="WGANGeneratorLoss"):
        super(WGANGeneratorLoss,self).__init__(name=name)

        self.raxis = raxis
        self.λ = λ

    #@tf.function
    def call(self, DX, DGz):

        if not self.raxis:
            raxis = [i for i in range(1, len(DGz.shape))]
        else:
            raxis = self.raxis

        EDGz = tf.reduce_mean(DGz, axis=raxis)
        return -self.λ*EDGz

class GradientPenalty(kl.Loss):
    """
        Compute the gradient penalty of discriminator D
    """

    def __init__(self, raxis=1, λ=1.0, kLip=1.0, name="GradientPenalty"):
        super(GradientPenalty, self).__init__(name=name)

        self.raxis = raxis
        self.λ = λ
        self.kLip = kLip
    
    def random_deviation(X, Gz):
        # Get the interpolated image
        α = tf.random.normal(shape=[X.shape[0], 1, 1], mean=0.0, stddev=1.0)
        δ = Gz - X
        XδX = X + α*δ
        return XδX

    #@tf.function
    def call(self, X, Gz):
        
        if not self.raxis:
            raxis = [i for i in range(1, len(GradD.shape))]
        else:
            raxis = self.raxis
        
        # 1. Compute random deviation
        XδX = self.random_deviation(X, X_fake)
        with tf.GradientTape() as gp_tape:
            # 1. Random deviation
            gp_tape.watch(XδX)
            # 2. Get the discriminator output for this interpolated image.
            DXδX = self.Dx(XδX, training=True)
            # 3. Calculate the gradients w.r.t to this interpolated image.
            gradDx_x = tape.gradient(DXδX, [XδX])[0]
    
        # Calculate the norm of the gradients.
        NormGradD = tf.math.sqrt(tf.math.reduce_sum(
            tf.math.square(gradDx_x), axis=raxis))
        GPloss = tf.math.reduce_mean((NormGradD - self.kLip) ** 2)
        return self.λ*GPloss

class MutualInfoLoss(kl.Loss):
    """
        Mutual Information loss (InfoGAN)
    """
    def __init__(self, raxis=1, λ=1.0, name="MutualInfoLoss"):
        super(MutualInfoLoss,self).__init__(name=name)

        self.raxis = raxis
        self.λ = λ
    
    #@tf.function
    def call(self, c, c_given_x):
        
        if not self.raxis:
            raxis = [i for i in range(1, len(c_given_x.shape))]
        else:
            raxis = self.raxis
        
        H_CgivenX = -tf.reduce_mean(tf.reduce_mean(tf.math.log(c_given_x+ε)*c,axis=raxis))
        H_C = -tf.reduce_mean(tf.reduce_mean(tf.math.log(c+ε)*c,axis=raxis))
        
        return self.λ*(H_CgivenX - H_C)


class InfoLoss(kl.Loss):
    """
        Categorical cross entropy loss as Information loss (InfoGAN)
    """
    def __init__(self, from_logits=True, raxis=1, λ=1.0, name="InfoLoss"):
        super(InfoLoss,self).__init__(name=name)

        self.from_logits = from_logits
        self.raxis = raxis
        self.λ = λ
        
    #@tf.function
    def call(self, c, QcX):

        if not self.raxis:
            raxis = [i for i in range(1, len(QcX.shape))]
        else:
            raxis = self.raxis

        return self.λ*kl.CategoricalCrossentropy(from_logits=self.from_logits,axis=raxis)(c, QcX)

def getCallbacks(**kwargs):
    getCallbacks.__globals__.update(kwargs)
    
    cb=[tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir,"ckpt-{epoch}.ckpt"),
                     save_freq=checkpoint_step, save_weights_only=True)]
    return cb

def getOptimizers(**kwargs):
    getOptimizers.__globals__.update(kwargs)
    optimizers = {}
    
    if DxTrainType.upper() == "WGAN":
        optimizers['DxOpt'] = RMSprop(learning_rate=DxLR)
    elif DxTrainType.upper() == "WGANGP":
        optimizers['DxOpt'] = Adam(learning_rate=DxLR, beta_1=0.0, beta_2=0.9)
    elif DxTrainType.upper() == "GAN":
        optimizers['DxOpt'] = Adam(learning_rate=DxLR, beta_1=0.5, beta_2=0.9999)

    # if DcTrainType.upper() == "WGAN":
    #     optimizers['DcOpt'] = RMSprop(learning_rate=DcLR)        
    # elif DcTrainType.upper() == "WGANGP":
    #     optimizers['DcOpt'] = Adam(learning_rate=DcLR, beta_1=0.0, beta_2=0.9)
    # elif DcTrainType.upper() == "GAN":
    #     optimizers['DcOpt'] = Adam(learning_rate=DcLR, beta_1=0.5, beta_2=0.9999)

    # if DsTrainType.upper() == "WGAN":
    #     optimizers['DsOpt'] = RMSprop(learning_rate=DsLR)
    # elif DsTrainType.upper() == "WGANGP":
    #     optimizers['DsOpt'] = Adam(learning_rate=DsLR, beta_1=0.0, beta_2=0.9)
    # elif DsTrainType.upper() == "GAN":
    #     optimizers['DsOpt'] = Adam(learning_rate=DsLR, beta_1=0.5, beta_2=0.9999)
        
    # if DnTrainType.upper() == "WGAN":
    #     optimizers['DnOpt'] = RMSprop(learning_rate=DnLR)
    # elif DnTrainType.upper() == "WGANGP":
    #     optimizers['DnOpt'] = Adam(learning_rate=DnLR, beta_1=0.0, beta_2=0.9)
    # elif DnTrainType.upper() == "GAN":
    #     optimizers['DnOpt'] = Adam(learning_rate=DnLR, beta_1=0.5, beta_2=0.9999)
        
    if DzTrainType.upper() == "WGAN":
        optimizers['DzOpt'] = RMSprop(learning_rate=DzLR)
    elif DzTrainType.upper() == "WGANGP":
        optimizers['DzOpt'] = Adam(learning_rate=DzLR, beta_1=0.0, beta_2=0.9)
    elif DzTrainType.upper() == "GAN":
        optimizers['DzOpt'] = Adam(learning_rate=DzLR, beta_1=0.5, beta_2=0.9999)

    # if DxTrainType.upper() == "WGAN":
    #     optimizers['FxOpt'] = Adam(learning_rate=FxLR, beta_1=0.5, beta_2=0.9999)
    #     optimizers['GzOpt'] = Adam(learning_rate=GzLR, beta_1=0.5, beta_2=0.9999)
    # elif DxTrainType.upper() == "WGANGP":
    #     optimizers['FxOpt'] = Adam(learning_rate=FxLR, beta_1=0.0, beta_2=0.9)
    #     optimizers['GzOpt'] = Adam(learning_rate=GzLR, beta_1=0.0, beta_2=0.9)
    # elif DxTrainType.upper() == "GAN":
    #     optimizers['FxOpt'] = Adam(learning_rate=FxLR, beta_1=0.5, beta_2=0.9999)
    #     optimizers['GzOpt'] = Adam(learning_rate=GzLR, beta_1=0.5, beta_2=0.9999)
    
    return optimizers

def getLosses(**kwargs):
    getLosses.__globals__.update(kwargs)
    losses = {}
    
    # if DcTrainType.upper() == "WGAN" or "WGANGP":
    #     losses['AdvDlossC'] = WGANDiscriminatorLoss(λ=PenAdvCloss)
    #     losses['AdvGlossC'] = WGANGeneratorLoss(λ=PenAdvCloss)
    # elif DcTrainType.upper() == "GAN":
    #     losses['AdvDlossC'] = GANDiscriminatorLoss(λ=PenAdvCloss)
    #     losses['AdvGlossC'] = GANGeneratorLoss(λ=PenAdvCloss)
    # elif DcTrainType.upper() == "HINGE":
    #     losses['AdvDlossC'] = HingeGANDiscriminatorLoss(λ=PenAdvCloss)
    #     losses['AdvGlossC'] = GANGeneratorLoss(λ=PenAdvCloss)

    # if DsTrainType.upper() == "WGAN" or "WGANGP":
    #     losses['AdvDlossS'] = WGANDiscriminatorLoss(λ=PenAdvSloss)
    #     losses['AdvGlossS'] = WGANGeneratorLoss(λ=PenAdvSloss)
    # elif DsTrainType.upper() == "GAN":
    #     losses['AdvDlossS'] = GANDiscriminatorLoss(λ=PenAdvSloss)
    #     losses['AdvGlossS'] = GANGeneratorLoss(λ=PenAdvSloss)
    # elif DsTrainType.upper() == "HINGE":
    #     losses['AdvDlossS'] = HingeGANDiscriminatorLoss(λ=PenAdvSloss)
    #     losses['AdvGlossS'] = GANGeneratorLoss(λ=PenAdvSloss)
        
    # if DnTrainType.upper() == "WGAN" or "WGANGP":
    #     losses['AdvDlossN'] = WGANDiscriminatorLoss(λ=PenAdvNloss)
    #     losses['AdvGlossN'] = WGANGeneratorLoss(λ=PenAdvNloss)
    # elif DnTrainType.upper() == "GAN":
    #     losses['AdvDlossN'] = GANDiscriminatorLoss(λ=PenAdvNloss)
    #     losses['AdvGlossN'] = GANGeneratorLoss(λ=PenAdvNloss)
    # elif DnTrainType.upper() == "HINGE":
    #     losses['AdvDlossN'] = HingeGANDiscriminatorLoss(λ=PenAdvNloss)
    #     losses['AdvGlossN'] = GANGeneratorLoss(λ=PenAdvNloss)
    if DzTrainType.upper() == "WGAN" or "WGANGP":
        losses['AdvDlossZ'] = WGANDiscriminatorLoss(λ=PenAdvZloss)
        losses['AdvGlossZ'] = WGANGeneratorLoss(λ=PenAdvZloss)
    elif DzTrainType.upper() == "GAN":
        losses['AdvDlossZ'] = GANDiscriminatorLoss(λ=PenAdvZloss)
        losses['AdvGlossZ'] = GANGeneratorLoss(λ=PenAdvZloss)
    elif DzTrainType.upper() == "HINGE":
        losses['AdvDlossZ'] = HingeGANDiscriminatorLoss(λ=PenAdvZloss)
        losses['AdvGlossZ'] = GANGeneratorLoss(λ=PenAdvZloss)
        
    if DxzTrainType.upper() == "WGAN" or "WGANGP":
        losses['AdvDlossXz'] = WGANDiscriminatorLoss(λ=PenAdvZloss)
        losses['AdvGlossXz'] = WGANGeneratorLoss(λ=PenAdvZloss)
    elif DxzTrainType.upper() == "GAN":
        losses['AdvDlossXz'] = GANDiscriminatorLoss(λ=PenAdvZloss)
        losses['AdvGlossXz'] = GANGeneratorLoss(λ=PenAdvZloss)
    elif DxzTrainType.upper() == "HINGE":
        losses['AdvDlossXz'] = HingeGANDiscriminatorLoss(λ=PenAdvZloss)
        losses['AdvGlossXz'] = GANGeneratorLoss(λ=PenAdvZloss)
        
    # if DxTrainType.upper() == "WGAN" or "WGANGP":
    #     losses['AdvDlossX'] = WGANDiscriminatorLoss(λ=PenAdvXloss)
    #     losses['AdvGlossX'] = WGANGeneratorLoss(λ=PenAdvXloss)
    # elif DxTrainType.upper() == 'GAN':
    #     losses['AdvDlossX'] = GANDiscriminatorLoss(λ=PenAdvXloss)
    #     losses['AdvGlossX'] = GANGeneratorLoss(λ=PenAdvXloss)
    # elif DxTrainType.upper() == "HINGE":
    #     raise Exception("Hinge loss not implemented for Dx")

    # if DcTrainType.upper() == "WGANGP":
    #     losses["PenDcLoss"] = GradientPenalty(λ=PenGPCloss)
    # if DsTrainType.upper() == "WGANGP":
    #     losses["PenDsLoss"] = GradientPenalty(λ=PenGPSloss)
    # if DnTrainType.upper() == "WGANGP":
    #     losses["PenDnLoss"] = GradientPenalty(λ=PenGPNloss)
    # if DxTrainType.upper() == "WGANGP":
    #     losses["PenDxLoss"] = GradientPenalty(λ=PenGPXloss)
    if DzTrainType.upper() == "WGANGP":
            losses["PenDzLoss"] = GradientPenalty(λ=PenGPZloss)
    
    # losses['RecSloss']  = GaussianNLL(λ=PenRecSloss)
    losses['RecXloss']  = kl.MeanSquaredError()
    # losses['RecCloss']  = InfoLoss(λ=PenRecCloss)
    # losses['FakeCloss'] = kl.CategoricalCrossentropy(from_logits=True)

    return losses