# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
__author__ = "Filippo Gatti Giorgia Colombera"
__copyright__ = "Copyright 2021, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__Maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import math as mt

import tensorflow as tf
#import tensorflow_probability as tfp
#tf.config.run_functions_eagerly(True)
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Layer
from tensorflow.keras.layers import Lambda, Concatenate, concatenate, ZeroPadding1D
from tensorflow.keras.layers import LeakyReLU, ReLU, Softmax
from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.constraints import Constraint, min_max_norm
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import timeit
from numpy.random import randn
from numpy.random import randint
from tensorflow.python.eager import context
import kerastuner as kt
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from numpy.linalg import norm
import MDOFload as mdof
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import GridSearchCV

from tensorflow.python.util.tf_export import tf_export
from copy import deepcopy
from plot_tools import *

π = tf.constant(np.pi)
log2π = tf.constant(-0.5*np.log(2*π))

AdvDLoss_tracker  = keras.metrics.Mean(name="loss")
AdvDlossX_tracker = keras.metrics.Mean(name="loss")
AdvDlossC_tracker = keras.metrics.Mean(name="loss")
AdvDlossS_tracker = keras.metrics.Mean(name="loss")
AdvDlossN_tracker = keras.metrics.Mean(name="loss")
AdvDlossPenGradX_tracker = keras.metrics.Mean(name="loss")
AdvDlossPenGradS_tracker = keras.metrics.Mean(name="loss")
AdvGLoss_tracker = keras.metrics.Mean(name="loss")
AdvGlossX_tracker = keras.metrics.Mean(name="loss")
AdvGlossC_tracker = keras.metrics.Mean(name="loss")
AdvGlossS_tracker = keras.metrics.Mean(name="loss")
AdvGlossN_tracker = keras.metrics.Mean(name="loss")
RecGlossX_tracker = keras.metrics.Mean(name="loss")
RecGlossC_tracker = keras.metrics.Mean(name="loss")
RecGlossS_tracker = keras.metrics.Mean(name="loss")

checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def make_or_restore_model():
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()

def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=2000,help='Number of epochs')
    parser.add_argument("--Xsize",type=int,default=2048,help='Data space size')
    parser.add_argument("--nX",type=int,default=200,help='Number of signals')
    parser.add_argument("--nXchannels",type=int,default=2,help="Number of data channels")
    parser.add_argument("--nAElayers",type=int,default=3,help='Number of AE CNN layers')
    parser.add_argument("--nDlayers",type=int,default=10,help='Number of D CNN layers')
    parser.add_argument("--kernel",type=int,default=3,help='CNN kernel size')
    parser.add_argument("--stride",type=int,default=2,help='CNN stride')
    parser.add_argument("--nZfirst",type=int,default=8,help="Initial number of channels")
    parser.add_argument("--branching",type=str,default='conv',help='conv or dens')
    parser.add_argument("--latentSdim",type=int,default=2,help="Latent space s dimension")
    parser.add_argument("--latentCdim",type=int,default=2,help="Number of classes")
    parser.add_argument("--latentNdim",type=int,default=20,help="Latent space n dimension")
    parser.add_argument("--nSlayers",type=int,default=3,help='Number of S-branch CNN layers')
    parser.add_argument("--nClayers",type=int,default=3,help='Number of C-branch CNN layers')
    parser.add_argument("--nNlayers",type=int,default=3,help='Number of N-branch CNN layers')
    parser.add_argument("--Skernel",type=int,default=7,help='CNN kernel of S-branch branch')
    parser.add_argument("--Ckernel",type=int,default=7,help='CNN kernel of C-branch branch')
    parser.add_argument("--Nkernel",type=int,default=7,help='CNN kernel of N-branch branch')
    parser.add_argument("--Sstride",type=int,default=4,help='CNN stride of S-branch branch')
    parser.add_argument("--Cstride",type=int,default=4,help='CNN stride of C-branch branch')
    parser.add_argument("--Nstride",type=int,default=4,help='CNN stride of N-branch branch')
    parser.add_argument("--Ssampling",type=str,default='normal',help='Sampling distribution for s')
    parser.add_argument("--Nsampling",type=str,default='normal',help='Sampling distribution for n')
    parser.add_argument("--batchSize",type=int,default=25,help='input batch size')    
    parser.add_argument("--nCritic",type=int,default=5,help='number of discriminator training steps')
    parser.add_argument("--clipValue",type=float,default=0.01,help='clip weight for WGAN')
    parser.add_argument("--dataroot_1",type=str,default="/gpfs/workdir/invsem07/stead_1_1U",help="Data root folder - Undamaged")
    parser.add_argument("--dataroot_2",type=str,default="/gpfs/workdir/invsem07/stead_1_1D",help="Data root folder - Damaged") 
    parser.add_argument("--idChannels",type=int,nargs='+',default=[1,39],help="Channel 1")
    parser.add_argument("--nParams",type=str,default=2,help="Number of parameters")
    parser.add_argument("--case",type=str,default="train_model",help="case")
    parser.add_argument("--avu",type=str,nargs='+',default="U",help="case avu")
    parser.add_argument("--pb",type=str,default="DC",help="case pb")#DC
    parser.add_argument("--CreateData",action='store_true',default=True,help='Create data flag')
    parser.add_argument("--cuda",action='store_true',default=False,help='Use cuda powered GPU')
    parser.add_argument('--dtm',type=float,default=0.04,help='time-step [s]')
    options = parser.parse_args().__dict__

    options['batchXshape'] = (options['batchSize'],options['Xsize'],options['nXchannels'])
    options['Xshape'] = (options['Xsize'],options['nXchannels'])

    options['latentZdim'] = options['latentSdim']+options['latentNdim']+options['latentCdim']

    options['Zsize'] = options['Xsize']//(options['stride']**options['nAElayers'])
    options['nZchannels'] = options['nZfirst']*(options['stride']**options['nAElayers']) # options['latentZdim']//options['Zsize']
    options['nZshape'] = (options['Zsize'],options['nZchannels'])

    options['nSchannels'] = options['nZchannels']*options['Sstride']**options['nSlayers']
    options['nCchannels'] = options['nZchannels']*options['Cstride']**options['nClayers']
    options['nNchannels'] = options['nZchannels']*options['Nstride']**options['nNlayers']
    options['Ssize'] = int(options['Zsize']*options['Sstride']**(-options['nSlayers']))
    options['Csize'] = int(options['Zsize']*options['Cstride']**(-options['nClayers']))
    options['Nsize'] = int(options['Zsize']*options['Nstride']**(-options['nNlayers']))

    options['nDlayers'] = min(options['nDlayers'],int(mt.log(options['Xsize'],options['stride'])))

    return options


class SamplingFxNormSfromVariance(Layer):
    def call(self, inputs):
        μ_z, σ_z2 = inputs
        σ_z = tf.math.sqrt(σ_z2)
        ε = tf.random.normal(shape=tf.shape(μ_z),mean=0.0,stddev=1.0)
        z = μ_z + tf.math.multiply(σ_z,ε)
        return z

# batch = tf.shape(μ_z)[0]
# dim = tf.shape(μ_z)[1]
# ε = tf.keras.backend.random_normal(shape=(batch, dim))

class SamplingFxLogNormSfromVariance(Layer):
    def call(self, inputs):
        λ_z, ζ_z2 = inputs
        ζ_z = tf.math.sqrt(ζ_z2)
        ε = tf.random.normal(shape=tf.shape(λ_z),mean=0.0,stddev=1.0)
        logz = λ_z + tf.math.multiply(ζ_z,ε)
        return tf.math.exp(logz)

class SamplingFxNormSfromLogVariance(Layer):
    def call(self, inputs):
        μ_z, logσ_z2 = inputs
        ε = tf.random.normal(shape=tf.shape(μ_z),mean=0.0,stddev=1.0)
        z = μ_z + tf.math.multiply(tf.math.exp(logσ_z2*0.5),ε)
        return z

class SamplingFxLogNormSfromLogVariance(Layer):
    def call(self, inputs):
        λ_z, logζ_z2 = inputs
        ε = tf.random.normal(shape=tf.shape(λ_z),mean=0.0,stddev=1.0)
        logz = λ_z + tf.math.multiply(tf.math.exp(logσ_z2*0.5),ε)
        return tf.math.exp(logz)

# Clip model weights to a given hypercube
class ClipConstraint(Constraint):

    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)
    def get_config(self):
        return {'clip_value': self.clip_value}

def WassersteinDiscriminatorLoss(y_true, y_fake):
    """
        Waisserstein Loss for Generator
        Ls = D(G(X))-D(X)
    """
    real_loss = tf.reduce_mean(y_true)
    fake_loss = tf.reduce_mean(y_fake)
    return fake_loss - real_loss

def WassersteinGeneratorLoss(y_fake):
    """
        Waisserstein Loss for Generator
        Ls = -D(G(X))
    """
    return -tf.reduce_mean(y_fake)

def GaussianNLLfromNorm(y,Fx):
    """
        Gaussian negative loglikelihood loss function for pred~N(0,I)
    """
    n = int(int(Fx.shape[-1])/2)
    μ,σ2 = Fx[:,0:n],Fx[:,n:]
    σ = tf.math.sqrt(σ2)
    ε2 = -0.5*tf.math.reduce_sum(tf.math.square((y-μ)/σ),axis=-1)
    Trσ = tf.math.reduce_sum(tf.math.log(σ),axis=-1)
    # ε2 = -0.5*tf.keras.backend.sum(tf.keras.backend.square(,axis=-1)
    # Trσ = -tf.keras.backend.sum(tf.keras.backend.log(σ),axis=-1)
    log_likelihood = ε2+Trσ+log2π*n

    return tf.math.reduce_mean(-log_likelihood)


def GaussianNLLfromLogVariance(y,Fx):
    """
        Gaussian negative loglikelihood loss function for logpred~N(0,I)
    """
    n = int(int(Fx.shape[-1])/2)
    μ,logσ2 = Fx[:, 0:n],Fx[:, n:]
    σ = tf.math.exp(0.5*logσ2)
    ε2 = -0.5*tf.math.reduce_sum(tf.math.square((y-μ)/σ),axis=-1)
    Trσ = tf.math.reduce_sum(tf.math.log(σ),axis=-1)
    # ε2 = -0.5*tf.keras.backend.sum(tf.keras.backend.square((true-μ)/σ),axis=-1)
    # Trσ = -tf.keras.backend.sum(tf.keras.backend.log(σ),axis=-1)
    log_likelihood = ε2+Trσ+log2π*n

    return tf.math.reduce_mean(-log_likelihood)

def KLDivergenceFromLogVariance(Fx):
    n = int(int(Fx.shape[-1])/2)
    μ_z, logσ_z2 = Fx[:,0:n],Fx[:,n:]
    DKL = -0.5*(1.0+logσ_z2-tf.math.square(μ_z)-tf.math.exp(logσ_z2))
    DKL = tf.math.reduce_sum(DKL,axis=-1)
    return tf.math.reduce_mean(DKL)

def MutualInfoLoss(C,CgivenX):
    ε = 1e-8
    # SCgivenX = -tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.log(CgivenX+ε)*C,axis=1))
    # SC = -tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.log(C+ε)*C,axis=1))
    # return SCgivenX + SC
    SC = -tf.math.reduce_mean(tf.math.reduce_sum(tf.math.multiply(tf.math.log(C+ε),C),axis=-1))
    SCgivenX = -tf.math.reduce_mean(tf.math.reduce_sum(tf.math.multiply(tf.math.log(CgivenX+ε),C),axis=-1))
    return SCgivenX + SC

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

        self.Fx, self.Qs, self.Qc = self.BuildFx()
        self.Gz = self.BuildGz()

        tf.keras.utils.plot_model(self.Fx,to_file="Fx.png",
            show_shapes=True,show_layer_names=True)
        tf.keras.utils.plot_model(self.Qs,to_file="Qs.png",
            show_shapes=True,show_layer_names=True)
        tf.keras.utils.plot_model(self.Qc,to_file="Qc.png",
            show_shapes=True,show_layer_names=True)
        tf.keras.utils.plot_model(self.Gz,to_file="Gz.png",
            show_shapes=True,show_layer_names=True)
        tf.keras.utils.plot_model(self.Ds,to_file="Ds.png",
            show_shapes=True,show_layer_names=True)
        tf.keras.utils.plot_model(self.Dn,to_file="Dn.png",
            show_shapes=True,show_layer_names=True)
        tf.keras.utils.plot_model(self.Dc,to_file="Dc.png",
            show_shapes=True,show_layer_names=True)
        tf.keras.utils.plot_model(self.Dx,to_file="Dx.png",
            show_shapes=True,show_layer_names=True)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'size': self.size})
        return config    

    @property
    def metrics(self):
        return [AdvDLoss_tracker,
            AdvGLoss_tracker,
            AdvDlossX_tracker,
            AdvDlossC_tracker,
            AdvDlossS_tracker,
            AdvDlossN_tracker,
            AdvDlossPenGradX_tracker,
            AdvDlossPenGradS_tracker,
            AdvGlossX_tracker,
            AdvGlossC_tracker,
            AdvGlossS_tracker,
            AdvGlossN_tracker,
            RecGlossX_tracker,
            RecGlossC_tracker,
            RecGlossS_tracker]

    def compile(self,optimizers,losses):
        super(RepGAN, self).compile()
        self.__dict__.update(optimizers)
        self.__dict__.update(losses)

    def GradientPenaltyX(self,batchSize,realX,fakeX):
        """Compute the Discriminator gradient for penalty

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        α = tf.random.normal([batchSize,1,1],0.0,1.0)
        δX = fakeX - realX
        intX = realX + α*δX

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(intX)
            # 1. Get the discriminator output for this interpolated image.
            predX = self.Dx(intX,training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        gradDx = gp_tape.gradient(predX,[intX])[0]
        # 3. Calculate the norm of the gradients.
        NormgradDx = tf.math.sqrt(tf.reduce_sum(tf.math.square(gradDx),axis=-1))
        λNormgradDx = tf.reduce_mean((NormgradDx - 1.0) ** 2)
        return λNormgradDx


    def GradientPenaltyS(self,batchSize,realS,fakeS):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        α = tf.random.normal([batchSize,1],0.0,1.0)
        δs = fakeS - realS
        intS = realS + α*δs

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(intS)
            # 1. Get the discriminator output for this interpolated image.
            predS = self.Ds(intS,training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        gradDs = gp_tape.gradient(predS,[intS])[0]
        # 3. Calculate the norm of the gradients.
        NormgradDs = tf.math.sqrt(tf.reduce_sum(tf.math.square(GradDs),axis=-1))
        λNormgradDs = tf.reduce_mean((NormgradDs-1.0)**2)
        return λNormgradDs

    def SamplingNoise(self,mean=0.0,stddev=1.0,latentDim=128,distribution='normal'):
        if distribution=='normal':
            return tf.random.normal(mean=0.0,stddev=1.0,shape=[self.batchSize,latentDim])
        elif distribution=='lognormal':
            return tf.random.lognormal(mean=0.0,stddev=1.0,shape=[self.batchSize,latentDim])
        elif distribution=='uniform':
            return tf.random.lognormal(mean=0.0,stddev=1.0,shape=[self.batchSize,latentDim])
    
    def train_step(self, realXC):

        # Upwrap data batch (X,C)
        realX, realC = realXC
        self.batchSize = tf.shape(realX)[0]

        # # Generate S and N from Normal distributions
        realS = self.SamplingNoise(mean=0.0,stddev=1.0,
                latentDim=self.latentSdim,distribution=self.Ssampling)
        realN = self.SamplingNoise(mean=0.0,stddev=1.0,
                latentDim=self.latentNdim,distribution=self.Nsampling)

        # Adversarial ground truths for X
        critic_X = self.Dx(realX)
        realBCE_X = tf.ones_like(critic_X)
        fakeBCE_X = tf.zeros_like(critic_X)

        # Adversarial ground truths for C
        critic_C = self.Dc(realC)
        realBCE_C = tf.ones_like(critic_C)
        fakeBCE_C = tf.zeros_like(critic_C)

        # # Adversarial ground truths for S
        critic_S = self.Ds(realS)
        realBCE_S = tf.ones_like(critic_S)
        fakeBCE_S = tf.zeros_like(critic_S)

        # # Adversarial ground truths for N
        critic_N = self.Dn(realN)
        realBCE_N = tf.ones_like(critic_N)
        fakeBCE_N = tf.zeros_like(critic_N)




        """
            Train Discriminators
        """

        # Freeze generators' layers while training critics
        self.Fx.trainable = False
        self.Gz.trainable = False
        self.Qc.trainable = False
        self.Qs.trainable = False
        self.Dx.trainable = True
        self.Dc.trainable = True
        self.Ds.trainable = True
        self.Dn.trainable = True

        # Train discriminators nCritic times
        for _ in range(self.nCritic):

            # Generate S and N from Normal distributions
            
            realS = self.SamplingNoise(mean=0.0,stddev=1.0,
                latentDim=self.latentSdim,distribution=self.Ssampling)
            realN = self.SamplingNoise(mean=0.0,stddev=1.0,
                latentDim=self.latentNdim,distribution=self.Nsampling)

            # # Adversarial ground truths for S
            # critic_S = self.Ds(realS)
            # realBCE_S = tf.ones_like(critic_S)
            # fakeBCE_S = tf.zeros_like(critic_S)

            # # Adversarial ground truths for N
            # critic_N = self.Dn(realN)
            # realBCE_N = tf.ones_like(critic_N)
            # fakeBCE_N = tf.zeros_like(critic_N)

            with tf.GradientTape(persistent=True) as tape:

                # Generate fake latent space (S,C,N) from real signals 
                # encoded (s,c,n) = Fx(X)
                [fakeS,fakeC,fakeN] = self.Fx(realX) 

                # Generate fake signals X from real latent code
                # X = Gz(s,c,n)
                fakeX = self.Gz((realS,realC,realN)) 

                # Discriminator determines validity of the real and fake X
                fakeXcritic = self.Dx(fakeX,training=True)
                realXcritic = self.Dx(realX,training=True)

                # Discriminator determines validity of the real and fake C
                fakeCcritic = self.Dc(fakeC,training=True)
                realCcritic = self.Dc(realC,training=True)

                # Discriminator determines validity of the real and fake N
                fakeNcritic = self.Dn(fakeN,training=True)
                realNcritic = self.Dn(realN,training=True)

                # Discriminator determines validity of the real and fake S
                fakeScritic = self.Ds(fakeS,training=True)
                realScritic = self.Ds(realS,training=True)

                # Calculate the discriminator loss using the fake and real logits
                AdvDlossX  = self.AdvDlossGAN(realBCE_X,realXcritic)*self.PenAdvXloss
                AdvDlossX += self.AdvDlossGAN(fakeBCE_X,fakeXcritic)*self.PenAdvXloss
                #AdvDlossX = self.AdvDlossWGAN(realXcritic,fakeXcritic)*self.PenAdvXloss
                #AdvDlossX = -tf.reduce_mean(tf.log(realXcritic+1e-8) + tf.log(1 - fakeXcritic+1e-8))*self.PenAdvXloss
                #AdvDlossC = self.AdvDlossWGAN(realCcritic,fakeCcritic)*self.PenAdvCloss
                AdvDlossC  = self.AdvDlossGAN(realBCE_C,realCcritic)*self.PenAdvCloss
                AdvDlossC += self.AdvDlossGAN(fakeBCE_C,fakeCcritic)*self.PenAdvCloss
                #AdvDlossS = self.AdvDlossWGAN(realScritic,fakeScritic)*self.PenAdvSloss
                AdvDlossS  = self.AdvDlossGAN(realBCE_S,realScritic)*self.PenAdvSloss
                AdvDlossS += self.AdvDlossGAN(fakeBCE_S,fakeScritic)*self.PenAdvSloss
                AdvDlossN  = self.AdvDlossGAN(realBCE_N,realNcritic)*self.PenAdvNloss
                AdvDlossN += self.AdvDlossGAN(fakeBCE_N,fakeNcritic)*self.PenAdvNloss
                #AdvDlossN = self.AdvDlossWGAN(realNcritic,fakeNcritic)*self.PenAdvNloss
                #AdvDlossPenGradX = self.GradientPenaltyX(self.batchSize,realX,fakeX)*self.PenGradX
                #AdvDlossPenGradS = self.GradientPenaltyS(self.batchSize,realS,fakeS)*self.PenGradS

                AdvDloss = AdvDlossX + AdvDlossC + AdvDlossS + AdvDlossN #+ AdvDlossPenGradS #+AdvDlossPenGradX

            # Get the gradients w.r.t the discriminator loss
            gradDx, gradDc, gradDs, gradDn = tape.gradient(AdvDloss,(self.Dx.trainable_variables, self.Dc.trainable_variables,
                self.Ds.trainable_variables, self.Dn.trainable_variables))

            # Update the weights of the discriminator using the discriminator optimizer
            self.DxOpt.apply_gradients(zip(gradDx,self.Dx.trainable_variables))
            self.DcOpt.apply_gradients(zip(gradDc,self.Dc.trainable_variables))
            self.DsOpt.apply_gradients(zip(gradDs,self.Ds.trainable_variables))
            self.DnOpt.apply_gradients(zip(gradDn,self.Dn.trainable_variables))



        """
            Train Generators
        """

        # Freeze critics' layers while training generators
        self.Fx.trainable = True
        self.Gz.trainable = True
        self.Qc.trainable = True
        self.Qs.trainable = True
        self.Dx.trainable = False
        self.Dc.trainable = False
        self.Ds.trainable = False
        self.Dn.trainable = False

        realS = self.SamplingNoise(mean=0.0,stddev=1.0,
                latentDim=self.latentSdim,distribution=self.Ssampling)
        realN = self.SamplingNoise(mean=0.0,stddev=1.0,
                latentDim=self.latentNdim,distribution=self.Nsampling)

        with tf.GradientTape(persistent=True) as tape:
            # Generate fake latent code from real signal
            [fakeS,fakeC,fakeN] = self.Fx(realX,training=True) # encoded z = Fx(X)

            fakeScritic = self.Ds(fakeS)
            fakeCcritic = self.Dc(fakeC)
            fakeNcritic = self.Dn(fakeN)

            fakeX = self.Gz((realS,realC,realN),training=True)

            fakeXcritic = self.Dx(fakeX)

            # Reconstruction
            recX = self.Gz((fakeS,fakeC,fakeN),training=True)
            recS = self.Qs(fakeX,training=True)
            recC = self.Qc(fakeX,training=True)

            # Adversarial ground truths
            AdvGlossX = self.AdvGlossGAN(realBCE_X,fakeXcritic)*self.PenAdvXloss
            #AdvGlossX = self.AdvGlossWGAN(fakeXcritic)*self.PenAdvXloss
            #AdvGlossX = - tf.reduce_mean(tf.log(fakeXcritic+1e-8))
            #AdvGlossC = self.AdvGlossWGAN(fakeCcritic)*self.PenAdvCloss
            AdvGlossC = self.AdvGlossGAN(realBCE_C,fakeCcritic)*self.PenAdvCloss
            #AdvGlossS = self.AdvGlossWGAN(fakeScritic)*self.PenAdvSloss
            #AdvGlossN = self.AdvGlossWGAN(fakeNcritic)*self.PenAdvNloss
            AdvGlossS = self.AdvGlossGAN(realBCE_S,fakeScritic)*self.PenAdvSloss
            AdvGlossN = self.AdvGlossGAN(realBCE_N,fakeNcritic)*self.PenAdvNloss
            RecGlossX = self.RecXloss(realX,recX)*self.PenRecXloss
            RecGlossS = self.RecSloss(realS,recS)*self.PenRecSloss
            #RecGlossS = -tf.reduce_mean(recS.log_prob(realS))
            RecGlossC = self.RecCloss(realC,recC)*self.PenRecCloss
            
            AdvGloss = AdvGlossX + AdvGlossC + AdvGlossS + AdvGlossN + RecGlossX + RecGlossC + RecGlossS

        # Get the gradients w.r.t the generator loss
        gradFx, gradGz, gradQs, gradQc = tape.gradient(AdvGloss,
            (self.Fx.trainable_variables,self.Gz.trainable_variables,
             self.Qs.trainable_variables,self.Qc.trainable_variables))

        # Update the weights of the generator using the generator optimizer
        self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))
        self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))
        self.QsOpt.apply_gradients(zip(gradQs,self.Qs.trainable_variables))
        self.QcOpt.apply_gradients(zip(gradQc,self.Qc.trainable_variables))

        # Compute our own metrics
        AdvDLoss_tracker.update_state(AdvDloss)
        AdvGLoss_tracker.update_state(AdvGloss)
        AdvDlossX_tracker.update_state(AdvDlossX)
        AdvDlossC_tracker.update_state(AdvDlossC)
        AdvDlossS_tracker.update_state(AdvDlossS)
        AdvDlossN_tracker.update_state(AdvDlossN)
        #AdvDlossPenGradX_tracker.update_state(AdvDlossPenGradX)
        #AdvDlossPenGradS_tracker.update_state(AdvDlossPenGradS)

        AdvGlossX_tracker.update_state(AdvGlossX)
        AdvGlossC_tracker.update_state(AdvGlossC)
        AdvGlossS_tracker.update_state(AdvGlossS)
        AdvGlossN_tracker.update_state(AdvGlossN)

        RecGlossX_tracker.update_state(RecGlossX)
        RecGlossC_tracker.update_state(RecGlossC)
        RecGlossS_tracker.update_state(RecGlossS)

        return {"AdvDlossX": AdvDlossX_tracker.result(),"AdvDlossC": AdvDlossC_tracker.result(),"AdvDlossS": AdvDlossS_tracker.result(),
            "AdvDlossN": AdvDlossN_tracker.result(),"AdvGlossX": AdvGlossX_tracker.result(),"AdvGlossC": AdvGlossC_tracker.result(),
            "AdvGlossS": AdvGlossS_tracker.result(),"AdvGlossN": AdvGlossN_tracker.result(),"RecGlossX": RecGlossX_tracker.result(), 
            "RecGlossC": RecGlossC_tracker.result(), "RecGlossS": RecGlossS_tracker.result(),"fakeX":tf.math.reduce_mean(fakeXcritic),"realX":tf.math.reduce_mean(realXcritic),
            "fakeC":tf.math.reduce_mean(fakeCcritic),"realC":tf.math.reduce_mean(realCcritic),"fakeN":tf.math.reduce_mean(fakeNcritic),"realN":tf.math.reduce_mean(realNcritic),"fakeS":tf.math.reduce_mean(fakeScritic),"realS":tf.math.reduce_mean(realScritic)}
        #"AdvDlossPenGradS":AdvDlossPenGradS_tracker.result()
        #return {"AdvDloss": AdvDLoss_tracker.result(),"AdvGloss": AdvGLoss_tracker.result(), "AdvDlossX": AdvDlossX_tracker.result(),
        #    "AdvDlossC": AdvDlossC_tracker.result(),"AdvDlossS": AdvDlossS_tracker.result(),"AdvDlossN": AdvDlossN_tracker.result(),
        #    "RecGlossX": RecGlossX_tracker.result(), "RecGlossC": RecGlossC_tracker.result(), "RecGlossS": RecGlossS_tracker.result()}


    def call(self, X):
        [fakeS,fakeC,fakeN] = self.Fx(X)
        fakeX = self.Gz((fakeS,fakeC,fakeN))
        fakeN_res = self.SamplingNoise(mean=0.0,stddev=1.0,
                latentDim=self.latentNdim,distribution=self.Nsampling)
        fakeX_res = self.Gz((fakeS,fakeC,fakeN_res))
        return fakeX, fakeC, fakeS, fakeN, fakeX_res

    def generate(self, X, fakeC_new):
        [fakeS,fakeC,fakeN] = self.Fx(X)
        fakeX = self.Gz((fakeS,fakeC_new,fakeN))
        return fakeX

    def BuildFx(self):
        """
            Conv1D Fx structure
        """
        # To build this model using the functional API

        # Input layer
        X = Input(shape=self.Xshape,name="X")

        # Initial CNN layer
        layer = -1
        # h = Conv1D(self.nZfirst*self.nSchannels*self.stride**(-self.nAElayers+layer+1),
        h = Conv1D(self.nZfirst, #*self.nSchannels*self.stride**(-self.nAElayers+layer+1),
                self.kernel,1,padding="same", # self.kernel,self.stride,padding="same",
                data_format="channels_last",name="FxCNN0")(X)
        h = LeakyReLU(alpha=0.1,name="FxA0")(h)
        h = BatchNormalization(momentum=0.95,name="FxBN0")(h)
        h = Dropout(0.2,name="FxDO0")(h)

        # Common encoder CNN layers
        for layer in range(self.nAElayers):
            # h = Conv1D(self.nZfirst*self.nSchannels*self.stride**(-self.nAElayers+layer+1),
            h = Conv1D(self.nZfirst*self.stride**(layer+1),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
            h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
            h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
            h = Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)

        # Last common CNN layer (no stride, same channels) before branching
        layer = self.nAElayers
        # h = Conv1D(self.nZfirst*self.nSchannels*self.stride**(-self.nAElayers+layer+1),
        h = Conv1D(self.nZchannels,
            self.kernel,1,padding="same",
            data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
        h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
        h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
        z = Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)
        # z ---> Zshape = (Zsize,nZchannels)

        layer = 0
        if 'dense' in self.branching:
            # Flatten and branch
            h = Flatten(name="FxFL{:>d}".format(layer+1))(z)
            h = Dense(self.latentZdim,name="FxFW{:>d}".format(layer+1))(h)
            h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
            h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)

            # variable s
            # s-average
            h = Dense(self.latentSdim,name="FxFWmuS")(zf)
            μ_s = BatchNormalization(momentum=0.95)(h)

            # s-log std
            h = Dense(self.latentSdim,name="FxFWlvS")(zf)
            logσ_s2 = BatchNormalization(momentum=0.95)(h)

            # variable c
            h = Dense(self.latentCdim,name="FxFWC")(zf)
            Zc = BatchNormalization(momentum=0.95,name="FxBNC")(h)

            # variable n
            Zn = Dense(self.latentNdim,name="FxFWN")(zf)

        elif 'conv' in self.branching:
            # variable s
            # s-average
            μ_s = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(z)
            μ_s = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(μ_s)
            μ_s = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(μ_s)
            μ_s = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(μ_s)

            # s-log std
            logσ_s2 = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(z)
            logσ_s2 = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(logσ_s2)
            logσ_s2 = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(logσ_s2)
            logσ_s2 = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(logσ_s2)

            # variable c
            Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(z)
            Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
            Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
            Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)

            # variable n
            Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
                    self.Nkernel,self.Nstride,padding="same",
                    data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(z)
            Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
            Zn = BatchNormalization(momentum=0.95,name="FxBNN{:>d}".format(layer+1))(Zn)
            Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)

            # variable s
            for layer in range(1,self.nSlayers):
                # s-average
                μ_s = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(μ_s)
                μ_s = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(μ_s)
                μ_s = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(μ_s)
                μ_s = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(μ_s)

                # s-log std
                logσ_s2 = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(logσ_s2)
                logσ_s2 = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(logσ_s2)
                logσ_s2 = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(logσ_s2)
                logσ_s2 = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(logσ_s2)

            # c
            for layer in range(1,self.nClayers):
                Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(Zc)
                Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
                Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
                Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)

            # n
            for layer in range(1,self.nNlayers):
                Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
                    self.Nkernel,self.Nstride,padding="same",
                    data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(Zn)
                Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
                Zn = BatchNormalization(momentum=0.95,name="FxBNN{:>d}".format(layer+1))(Zn)
                Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)

            # s-average
            μ_s = Flatten(name="FxFLmuS{:>d}".format(layer+1))(μ_s)
            μ_s = Dense(self.latentSdim,name="FxFWmuS")(μ_s)
            μ_s = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+2))(μ_s)
            μ_s = BatchNormalization(momentum=0.95,name="FxBNmuS")(μ_s)

            # s-log variance
            logσ_s2 = Flatten(name="FxFLlvS{:>d}".format(layer+1))(logσ_s2)
            logσ_s2 = Dense(self.latentSdim,name="FxFWlvS")(logσ_s2)
            logσ_s2 = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+2))(logσ_s2)
            logσ_s2 = BatchNormalization(momentum=0.95,name="FxBNlvS")(logσ_s2)

            # c
            layer = self.nClayers
            Zc = Flatten(name="FxFLC{:>d}".format(layer+1))(Zc)
            Zc = Dense(self.latentCdim,name="FxFWC")(Zc)
            Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+2))(Zc)
            Zc = BatchNormalization(momentum=0.95,name="FxBNC")(Zc)

            # n
            layer = self.nNlayers
            Zn = Flatten(name="FxFLN{:>d}".format(layer+1))(Zn)
            Zn = Dense(self.latentNdim,name="FxFWN{:>d}".format(layer+2))(Zn)
            Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)

        # variable s
        s = SamplingFxNormSfromLogVariance()([μ_s,logσ_s2])
        QsX = Concatenate(axis=-1)([μ_s,logσ_s2])

        # variable c
        c = Softmax(name="FxAC")(Zc)
        #c = BatchNormalization(momentum=0.95,name="FxBNC2")(c)
        QcX = Softmax(name="QcAC")(Zc)
        #QcX = BatchNormalization(momentum=0.95,name="QcBNC")(QcX)

        # variable n
        n = BatchNormalization(momentum=0.95,name="FxBNN")(Zn)

        Fx = keras.Model(X,[s,c,n],name="Fx")
        Qs = keras.Model(X,QsX,name="Qs")
        Qc = keras.Model(X,QcX,name="Qc")

        return Fx,Qs,Qc

    def BuildGz(self):
        """
            Conv1D Gz structure
            https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
        """
        s = Input(shape=(self.latentSdim,),name="s")
        c = Input(shape=(self.latentCdim,),name="c")
        n = Input(shape=(self.latentNdim,),name="n")

        layer = 0
        if 'dense' in self.branching:
            # variable s
            Zs = Dense(self.latentSdim,name="GzFWS")(s)
            Zs = BatchNormalization(momentum=0.95,name="GzBNS")(Zs)
            GzS = keras.Model(s,Zs)

            # variable c
            Zc = Dense(self.latentCdim,name="GzFWC")(c)
            Zc = BatchNormalization(momentum=0.95,name="GzBNC")(Zc)
            GzC = keras.Model(c,Zc)

            # variable n
            Zn = Dense(self.latentNdim,name="GzFWN")(n)
            Zn = BatchNormalization(momentum=0.95,name="GzBNN")(Zn)
            GzN = keras.Model(n,Zn)

            Gz = concatenate([GzS.output,GzC.output,GzN.output])
            Gz = Reshape((self.Zsize,self.nZchannels))(z)
            Gz = LeakyReLU(alpha=0.1,name="GzA0".format(layer+1))(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)

        elif 'conv' in self.branching:
            # variable s
            Zs = Dense(self.Ssize*self.nSchannels,name="GzFWS0")(s)
            Zs = BatchNormalization(name="GzBNS0")(Zs)
            Zs = Reshape((self.Ssize,self.nSchannels))(Zs)

            for layer in range(1,self.nSlayers):
                Zs = Conv1DTranspose(int(self.nSchannels*self.Sstride**(-layer)),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last",name="GzCNNS{:>d}".format(layer))(Zs)
                Zs = LeakyReLU(alpha=0.1,name="GzAS{:>d}".format(layer))(Zs)
                Zs = BatchNormalization(momentum=0.95,name="GzBNS{:>d}".format(layer))(Zs)
                Zs = Dropout(0.2,name="GzDOS{:>d}".format(layer))(Zs)
            Zs = Conv1DTranspose(int(self.nSchannels*self.Sstride**(-self.nSlayers)),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="GzCNNS{:>d}".format(self.nSlayers))(Zs)
            Zs = LeakyReLU(alpha=0.1,name="GzAS{:>d}".format(self.nSlayers))(Zs)
            Zs = BatchNormalization(momentum=0.95,name="GzBNS{:>d}".format(self.nSlayers))(Zs)
            Zs = Dropout(0.2,name="GzDOS{:>d}".format(self.nSlayers))(Zs)
            GzS = keras.Model(s,Zs)

            # variable c
            Zc = Dense(self.Csize*self.nCchannels,name="GzFWC0")(c)
            Zc = BatchNormalization(name="GzBNC0")(Zc)
            Zc = Reshape((self.Csize,self.nCchannels))(Zc)
            for layer in range(1,self.nClayers):
                Zc = Conv1DTranspose(int(self.nCchannels*self.Cstride**(-layer)),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last",name="GzCNNC{:>d}".format(layer))(Zc)
                Zc = LeakyReLU(alpha=0.1,name="GzAC{:>d}".format(layer))(Zc)
                Zc = BatchNormalization(momentum=0.95,name="GzBNC{:>d}".format(layer))(Zc)
                Zc = Dropout(0.2,name="GzDOC{:>d}".format(layer))(Zc)
            Zc = Conv1DTranspose(int(self.nCchannels*self.Cstride**(-self.nClayers)),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last",name="GzCNNC{:>d}".format(self.nClayers))(Zc)
            Zc = LeakyReLU(alpha=0.1,name="GzAC{:>d}".format(self.nClayers))(Zc)
            Zc = BatchNormalization(momentum=0.95,name="GzBNC{:>d}".format(self.nClayers))(Zc)
            Zc = Dropout(0.2,name="GzDOC{:>d}".format(self.nClayers))(Zc)
            GzC = keras.Model(c,Zc)

            # variable n
            Zn = Dense(self.Nsize*self.nNchannels,name="GzFWN0")(n)
            Zn = BatchNormalization(name="GzBNN0")(Zn)
            Zn = Reshape((self.Nsize,self.nNchannels))(Zn)
            for layer in range(1,self.nNlayers):
                Zn = Conv1DTranspose(int(self.nNchannels*self.Nstride**(-layer)),
                    self.Nkernel,self.Nstride,padding="same",
                    data_format="channels_last",name="GzCNNN{:>d}".format(layer))(Zn)
                Zn = LeakyReLU(alpha=0.1,name="GzAN{:>d}".format(layer))(Zn)
                Zn = BatchNormalization(momentum=0.95,name="GzBNN{:>d}".format(layer))(Zn)
                Zn = Dropout(0.2,name="GzDON{:>d}".format(layer))(Zn)
            Zn = Conv1DTranspose(int(self.nNchannels*self.Nstride**(-self.nNlayers)),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last",name="GzCNNN{:>d}".format(self.nNlayers))(Zn)
            Zn = LeakyReLU(alpha=0.1,name="GzAN{:>d}".format(self.nNlayers))(Zn)
            Zn = BatchNormalization(momentum=0.95,name="GzBNN{:>d}".format(self.nNlayers))(Zn)
            Zn = Dropout(0.2,name="GzDON{:>d}".format(self.nNlayers))(Zn)
            GzN = keras.Model(n,Zn)

            Gz = concatenate([GzS.output,GzC.output,GzN.output])
            Gz = Conv1DTranspose(self.nZchannels,
                    self.kernel,1,padding="same",
                    data_format="channels_last",name="GzCNN0")(Gz)
            Gz = LeakyReLU(alpha=0.1,name="GzA0".format(layer+1))(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95,name="GzBN0")(Gz)

        for layer in range(self.nAElayers):
            Gz = Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                self.kernel,self.stride,padding="same",use_bias=False,
                name="GzCNN{:>d}".format(layer+1))(Gz)
            Gz = LeakyReLU(alpha=0.1,name="GzA{:>d}".format(layer+1))(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95,name="GzBN{:>d}".format(layer+1))(Gz)

        layer = self.nAElayers
        #X = Conv1DTranspose(self.nXchannels,self.kernel,1,
        #    padding="same",use_bias=False,name="GzCNN{:>d}".format(layer+1))(Gz)
        X = Conv1DTranspose(self.nXchannels,self.kernel,1,
           padding="same",activation='tanh',use_bias=False,name="GzCNN{:>d}".format(layer+1))(Gz) #activation='tanh'
        Gz = keras.Model(inputs=[GzS.input,GzC.input,GzN.input],outputs=X,name="Gz")
        return Gz

    def BuildDx(self):
        """
            Conv1D discriminator structure
        """
        layer = 0
        X = Input(shape=self.Xshape,name="X")
        h = Conv1D(self.Xsize*self.stride**(-(layer+1)),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="DxCNN0")(X)
        h = LeakyReLU(alpha=0.1,name="DxA0")(h)
        h = Dropout(0.25,name="DxDO0")(h)
        for layer in range(1,self.nDlayers):
            h = Conv1D(self.Xsize*self.stride**(-(layer+1)),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="DxCNN{:>d}".format(layer))(h)
            h = LeakyReLU(alpha=0.1,name="DxA{:>d}".format(layer))(h)
            h = BatchNormalization(momentum=0.95,name="DxBN{:>d}".format(layer))(h)
            h = Dropout(0.25,name="DxDO{:>d}".format(layer))(h)
        layer = self.nDlayers    
        h = Flatten(name="DxFL{:>d}".format(layer))(h)
        Px = Dense(1,activation=tf.keras.activations.sigmoid)(h)
        Dx = keras.Model(X,Px,name="Dx")
        return Dx


    def BuildDc(self):
        """
            Dense discriminator structure
        """
        c = Input(shape=(self.latentCdim,))
        h = Dense(3000)(c)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(0.25)(h)
        h = Dense(3000)(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = BatchNormalization(momentum=0.95,)(h)
        h = Dropout(0.25)(h)
        Pc = Dense(1,activation=tf.keras.activations.sigmoid)(h)
        Dc = keras.Model(c,Pc,name="Dc")
        return Dc

    def BuildDn(self):
        """
            Dense discriminator structure
        """
        n = Input(shape=(self.latentNdim,))
        h = Dense(3000)(n)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(0.25)(h)
        h = Dense(3000)(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = BatchNormalization(momentum=0.95,)(h)
        h = Dropout(0.25)(h) 
        Pn = Dense(1,activation=tf.keras.activations.sigmoid)(h)
        Dn = keras.Model(n,Pn,name="Dn")
        return Dn

    def BuildDs(self):
        """
            Dense discriminator structure
        """
        s = Input(shape=(self.latentSdim,))
        h = Dense(3000)(s)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(0.25)(h)
        h = Dense(3000)(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = BatchNormalization(momentum=0.95,)(h)
        h = Dropout(0.25)(h)
        #Ps = Dense(1,activation=tf.keras.activations.sigmoid)(h)
        Ps = Dense(1,activation=tf.keras.activations.sigmoid)(h)
        Ds = keras.Model(s,Ps,name="Ds")
        return Ds

    def DumpModels(self):
        self.Fx.save("Fx.h5")
        self.Qs.save("Qs.h5")
        self.Qc.save("Qc.h5")
        self.Gz.save("Gz.h5")
        self.Dx.save("Dx.h5")
        self.Ds.save("Ds.h5")
        self.Dn.save("Dn.h5")
        self.Dc.save("Dc.h5")
        return

def Main(DeviceName):

    options = ParseOptions()

    if not options['cuda']:
        DeviceName = "/cpu:0"

    with tf.device(DeviceName):
        optimizers = {}
        optimizers['DxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['DcOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)#RMSprop(learning_rate=0.00005)
        optimizers['DsOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)#RMSprop(learning_rate=0.00005)
        optimizers['DnOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)#RMSprop(learning_rate=0.00005)
        optimizers['FxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['GzOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['QsOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['QcOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)

        losses = {}
        losses['AdvDlossWGAN'] = WassersteinDiscriminatorLoss
        losses['AdvGlossWGAN'] = WassersteinGeneratorLoss
        losses['AdvDlossGAN'] = tf.keras.losses.BinaryCrossentropy()
        losses['AdvGlossGAN'] = tf.keras.losses.BinaryCrossentropy()
        losses['RecSloss'] = GaussianNLLfromLogVariance
        losses['RecXloss'] = tf.keras.losses.MeanAbsoluteError()
        losses['RecCloss'] = MutualInfoLoss
        losses['PenAdvXloss'] = 1.
        losses['PenAdvCloss'] = 1.
        losses['PenAdvSloss'] = 1.
        losses['PenAdvNloss'] = 1.
        losses['PenRecXloss'] = 1.
        losses['PenRecCloss'] = 1.
        losses['PenRecSloss'] = 1.
        losses['PenGradX'] = 10.
        losses['PenGradS'] = 10.

    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # # Open a strategy scope.
    # with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.  

        # Instantiate the RepGAN model.
        GiorgiaGAN = RepGAN(options)

        # Compile the RepGAN model.
        GiorgiaGAN.compile(optimizers,losses) #run_eagerly=True

        if options['CreateData']:
            # Create the dataset
            Xtrn,  Xvld, _ = mdof.CreateData(**options)
        else:
            # Load the dataset
            Xtrn, Xvld, _ = mdof.LoadData(**options)
            # (Xtrn,Ctrn), (Xvld,Cvld), _ = mdof.LoadNumpyData(**options)

        callbacks = [keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}", 
            save_freq='epoch',period=500)] #keras.callbacks.EarlyStopping(patience=10)

        history = GiorgiaGAN.fit(Xtrn,epochs=options["epochs"],validation_data=Xvld,
            callbacks=callbacks)

        GiorgiaGAN.DumpModels()

        PlotLoss(history) # Plot loss

        Xtrn_u,  Xvld_u, _ = mdof.LoadUndamaged(**options)

        Xtrn_d,  Xvld_d, _ = mdof.LoadDamaged(**options)

        PlotReconstructedTHs(GiorgiaGAN,Xvld,Xvld_u,Xvld_d) # Plot reconstructed time-histories

        PlotTHSGoFs(GiorgiaGAN,Xvld) # Plot reconstructed time-histories

        ViolinPlot(GiorgiaGAN,Xvld) # Violin plot

        PlotBatchGoFs(GiorgiaGAN,Xtrn,Xvld) # Plot GoFs on a batch

        #PlotBatchGoFs_new(GiorgiaGAN,Xvld_u,Xvld_d) # Plot GoFs on a batch (after the change of C)

        #PlotClassificationMetrics(GiorgiaGAN,Xvld) # Plot classification metrics

        SwarmPlot(GiorgiaGAN,Xvld) # Swarm plot

if __name__ == '__main__':
    DeviceName = tf.test.gpu_device_name()
    Main(DeviceName)

# # We run each op once to warm up; see: https://stackoverflow.com/a/45067900
# cpu()
# gpu()

# # Run the op several times.
# print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
#       '(batch x height x width x channel). Sum of ten runs.')
# print('CPU (s):')
# cpu_time = timeit.timeit('cpu()', number=10, setup="from __Main__ import cpu")
# print(cpu_time)
# print('GPU (s):')
# gpu_time = timeit.timeit('gpu()', number=10, setup="from __Main__ import gpu")
# print(gpu_time)
# print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))
