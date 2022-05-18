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

import argparse
import math as mt

import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Layer
from tensorflow.keras.layers import Lambda, Concatenate, concatenate, ZeroPadding1D, Add
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

import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.python.util.tf_export import tf_export
from copy import deepcopy
from plot_tools_ultimo import *
import subprocess

#from tensorflow.python.framework import ops
#import keras.backend as K

from numpy.lib.type_check import imag
import sklearn
from PIL import Image
import io
from obspy.signal.tf_misfit import plot_tf_gofs, eg, pg
import numpy as np

from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import CustomJS, Slider, ColumnDataSource
from bokeh.io import curdoc
import bokeh
from bokeh.models import Text, Label
import panel
panel.extension()

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


#gpu_devices = tf.config.experimental.list_physical_devices('GPU')


checkpoint_dir = "./checkpoint_ultimo/14_04"
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

def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=2000,help='Number of epochs')
    parser.add_argument("--Xsize",type=int,default=2048,help='Data space size')
    parser.add_argument("--nX",type=int,default=4000,help='Number of signals')
    parser.add_argument("--nXchannels",type=int,default=4,help="Number of data channels")
    parser.add_argument("--nAElayers",type=int,default=3,help='Number of AE CNN layers')
    parser.add_argument("--nDlayers",type=int,default=10,help='Number of D CNN layers')
    parser.add_argument("--kernel",type=int,default=3,help='CNN kernel size')
    parser.add_argument("--stride",type=int,default=2,help='CNN stride')
    parser.add_argument("--nZfirst",type=int,default=8,help="Initial number of channels")
    parser.add_argument("--branching",type=str,default='conv',help='conv or dens')
    parser.add_argument("--latentSdim",type=int,default=2,help="Latent space s dimension")
    parser.add_argument("--latentCdim",type=int,default=2,help="Number of classes")
    parser.add_argument("--latentNdim",type=int,default=2,help="Latent space n dimension")
    parser.add_argument("--nSlayers",type=int,default=3,help='Number of S-branch CNN layers')
    parser.add_argument("--nClayers",type=int,default=3,help='Number of C-branch CNN layers')
    parser.add_argument("--nNlayers",type=int,default=3,help='Number of N-branch CNN layers')
    parser.add_argument("--Skernel",type=int,default=3,help='CNN kernel of S-branch branch')
    parser.add_argument("--Ckernel",type=int,default=3,help='CNN kernel of C-branch branch')
    parser.add_argument("--Nkernel",type=int,default=3,help='CNN kernel of N-branch branch')
    parser.add_argument("--Sstride",type=int,default=2,help='CNN stride of S-branch branch')
    parser.add_argument("--Cstride",type=int,default=2,help='CNN stride of C-branch branch')
    parser.add_argument("--Nstride",type=int,default=2,help='CNN stride of N-branch branch')
    parser.add_argument("--batchSize",type=int,default=50,help='input batch size')    
    parser.add_argument("--nCritic",type=int,default=1,help='number of discriminator training steps')
    parser.add_argument("--nGenerator",type=int,default=5,help='number of generator training steps')
    parser.add_argument("--clipValue",type=float,default=0.01,help='clip weight for WGAN')
    parser.add_argument("--dataroot", nargs="+", default=["./PortiqueElasPlas_N_2000",
                        "./PortiqueElasPlas_E_2000"],help="Data root folder") 
    # parser.add_argument("--dataroot", nargs="+", default=["/gpfs/workdir/invsem07/stead_1_9U","/gpfs/workdir/invsem07/stead_1_9D",
    #                     "/gpfs/workdir/invsem07/stead_1_10D"],help="Data root folder") 
    parser.add_argument("--idChannels",type=int,nargs='+',default=[1,2,3,4],help="Channel 1")
    parser.add_argument("--nParams",type=str,default=2,help="Number of parameters")
    parser.add_argument("--case",type=str,default="train_model",help="case")
    parser.add_argument("--avu",type=str,nargs='+',default="U",help="case avu")
    parser.add_argument("--pb",type=str,default="DC",help="case pb")#DC
    parser.add_argument("--CreateData",action='store_true',default=False,help='Create data flag')
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

    # assert options['nSchannels'] >= 1
    # assert options['Ssize'] >= options['Zsize']//(options['stride']**options['nSlayers'])

    return options


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
        z = z_mean + tf.multiply(z_std,tf.random.normal([1]))
        return z

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
    # clip model weights to hypercube
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

def WassersteinDiscriminatorLoss(y_true, y_fake):
    real_loss = tf.reduce_mean(y_true)
    fake_loss = tf.reduce_mean(y_fake)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def WassersteinGeneratorLoss(y_fake):
    return -tf.reduce_mean(y_fake)


def GaussianNLL(true,mu,sigma):
    """
     Gaussian negative loglikelihood loss function
        true=s
        pred=Qs(Gz(s,c,n))
    """
    n_dims = int(true.shape[1])
    #mu = pred[:, 0:n_dims]
    #sigma = pred[:, n_dims:]
    mse = -0.5*tf.keras.backend.sum(tf.keras.backend.square((true-mu)/sigma),axis=1)
    sigma_trace = -tf.keras.backend.sum(tf.keras.backend.log(sigma), axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood = mse+sigma_trace+log2pi

    return tf.keras.backend.mean(-log_likelihood)

def MutualInfoLoss(c, c_given_x):
    """The mutual information metric we aim to minimize"""
    eps = 1e-8
    conditional_entropy = -tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.log(c_given_x+eps)*c,axis=1))
    entropy = -tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.log(c+eps)*c,axis=1))

    return conditional_entropy - entropy

# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    idx = np.random.randint(low=0, high=dataset.shape[0], size=n_samples)
    #X[tf.arange(size_batch), rand_idx] = dataset
    X = tf.gather(dataset, idx)
    #idx = np.random.randint(0, size=(dataset.shape[0], n_samples))
    # select images
    #X = dataset[idx]

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

        self.Fx, self.h1, self.h0 = self.BuildFx() 
        self.Q, self.h2, self.h3 = self.BuildQ() 
        self.Gz = self.BuildGz()
        self.Gq = self.BuildGq()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config    

    @property
    def metrics(self):
        return [AdvDLoss_tracker,AdvGLoss_tracker,AdvDlossX_tracker,AdvDlossC_tracker,AdvDlossS_tracker,
            AdvDlossN_tracker,AdvGlossX_tracker,AdvGlossC_tracker,AdvGlossS_tracker,AdvGlossN_tracker,
            RecGlossX_tracker,RecGlossC_tracker,RecGlossS_tracker]

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
        GradDx = gp_tape.gradient(predX, [intX])[0]
        # 3. Calculate the norm of the gradients.
        NormGradX = tf.sqrt(tf.reduce_sum(tf.square(GradDx), axis=[1]))
        gp = tf.reduce_mean((NormGradX - 1.0) ** 2)
        return gp

    def GradientPenaltyS(self,batchSize,realX,fakeX):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batchSize, 1], 0.0, 1.0)
        diffX = fakeX - realX
        intX = realX + alpha * diffX

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


    def train_step(self, realXC):

        realX, realC = realXC

        self.batchSize = tf.shape(realX)[0]

                
        # Adversarial ground truths
        critic_X = self.Dx(realX)
        realBCE_X = tf.ones_like(critic_X)
        fakeBCE_X = tf.zeros_like(critic_X)
        critic_C = self.Dc(realC)
        realBCE_C = tf.ones_like(critic_C)
        fakeBCE_C = tf.zeros_like(critic_C)
        

        #idx_damaged = tf.math.argmax(realC, axis=1)

        #------------------------------------------------
        #           Construct Computational Graph
        #               for the Discriminator
        #------------------------------------------------

        # Freeze generators' layers while training critics
        self.Fx.trainable = False
        self.Gz.trainable = False
        self.Gq.trainable = True
        self.Q.trainable = False
        self.Dx.trainable = True
        self.Dc.trainable = True
        self.Ds.trainable = True
        self.Dn.trainable = True

        realS = tf.random.normal(mean=0.0,stddev=1.0,shape=[self.batchSize,self.latentSdim])
        realN = tf.random.normal(mean=0.0,stddev=1.0,shape=[self.batchSize,self.latentNdim])


        critic_S = self.Ds(realS)
        realBCE_S = tf.ones_like(critic_S)
        fakeBCE_S = tf.zeros_like(critic_S)
        critic_N = self.Dn(realN)
        realBCE_N = tf.ones_like(critic_N)
        fakeBCE_N = tf.zeros_like(critic_N)

        for _ in range(self.nCritic):

            with tf.GradientTape(persistent=True) as tape:

                # Generate fake signals from real latent code
                fakeX = self.Gq((realX,realS,realC,realN),training=True) # fake X = Gz(Fx(X))
                [_,_,fakeS,fakeC,fakeN] = self.Fx(realX,training=True)

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
                AdvDlossX = -tf.reduce_mean(tf.keras.backend.log(realXcritic+1e-8) + tf.keras.backend.log(1 - fakeXcritic+1e-8))*self.PenAdvXloss
                AdvGlossX = -tf.reduce_mean(tf.keras.backend.log(fakeXcritic+1e-8))*self.PenAdvXloss

                AdvDlossC  = self.AdvDlossGAN(realBCE_C,realCcritic)*self.PenAdvCloss
                AdvDlossC += self.AdvDlossGAN(fakeBCE_C,fakeCcritic)*self.PenAdvCloss
                AdvDlossS  = self.AdvDlossGAN(realBCE_S,realScritic)*self.PenAdvSloss
                AdvDlossS += self.AdvDlossGAN(fakeBCE_S,fakeScritic)*self.PenAdvSloss
                AdvDlossN  = self.AdvDlossGAN(realBCE_N,realNcritic)*self.PenAdvNloss
                AdvDlossN += self.AdvDlossGAN(fakeBCE_N,fakeNcritic)*self.PenAdvNloss

                
                AdvDloss = AdvDlossX + AdvGlossX + AdvDlossC + AdvDlossS + AdvDlossN

            # Get the gradients w.r.t the discriminator loss
            gradGq, gradDx, gradDc, gradDs, gradDn = tape.gradient(AdvDloss,
                (self.Gq.trainable_variables, self.Dx.trainable_variables,
                self.Dc.trainable_variables,self.Ds.trainable_variables, self.Dn.trainable_variables))

            # Update the weights of the discriminator using the discriminator optimizer
            self.GqOpt.apply_gradients(zip(gradGq,self.Gq.trainable_variables))
            self.DxOpt.apply_gradients(zip(gradDx,self.Dx.trainable_variables))
            self.DcOpt.apply_gradients(zip(gradDc,self.Dc.trainable_variables))
            self.DsOpt.apply_gradients(zip(gradDs,self.Ds.trainable_variables))
            self.DnOpt.apply_gradients(zip(gradDn,self.Dn.trainable_variables))

        # Freeze critics' layers while training generators
        self.Fx.trainable = True
        self.Gz.trainable = True
        self.Gq.trainable = True
        self.Q.trainable = True
        self.Dx.trainable = False
        self.Dc.trainable = False
        self.Ds.trainable = False
        self.Dn.trainable = False

        for _ in range(self.nGenerator):

            with tf.GradientTape(persistent=True) as tape:

                # Generate fake latent code from real signal
                [Fakemu,Fakesigma,fakeS,fakeC,fakeN] = self.Fx(realX,training=True)
                fakeX = self.Gq((realX,realS,realC,realN),training=True) # fake X = Gz(Fx(X))
                [Recmu,Recsigma,recS,recC,_] = self.Q(fakeX,training=True)

                recX = self.Gz((realX,fakeS,fakeC,fakeN),training=True)

                fakeScritic = self.Ds(fakeS,training=True)
                fakeCcritic = self.Dc(fakeC,training=True)
                fakeNcritic = self.Dn(fakeN,training=True)

                AdvGlossC = self.AdvGlossGAN(realBCE_C,fakeCcritic)*self.PenAdvCloss
                AdvGlossS = self.AdvGlossGAN(realBCE_S,fakeScritic)*self.PenAdvSloss
                AdvGlossN = self.AdvGlossGAN(realBCE_N,fakeNcritic)*self.PenAdvNloss

                RecGlossX = self.RecXloss(realX,recX)*self.PenRecXloss

                RecGlossS = self.RecSloss(realS,Recmu,Recsigma)*self.PenRecSloss
                RecGlossC = self.RecCloss(realC,recC)*self.PenRecCloss + self.RecCloss(realC,fakeC)*self.PenRecCloss

                
                AdvGloss = RecGlossS + RecGlossC + RecGlossX + AdvGlossC + AdvGlossS + AdvGlossN 


            # Get the gradients w.r.t the generator loss
            gradQ, gradGq, gradFx, gradGz = tape.gradient(AdvGloss,
                (self.Q.trainable_variables,self.Gq.trainable_variables,
                self.Fx.trainable_variables,self.Gz.trainable_variables))

            # Update the weights of the generator using the generator optimizer
            self.QOpt.apply_gradients(zip(gradQ,self.Q.trainable_variables))
            self.GqOpt.apply_gradients(zip(gradGq,self.Gq.trainable_variables))
            self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))
            self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))
            
      

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

        return {"AdvDlossX": AdvDlossX_tracker.result(),"AdvDlossC": AdvDlossC_tracker.result(), "AdvDlossS": AdvDlossS_tracker.result(),
            "AdvDlossN": AdvDlossN_tracker.result(),"AdvGlossX": AdvGlossX_tracker.result(),"AdvGlossC": AdvGlossC_tracker.result(),
            "AdvGlossS": AdvGlossS_tracker.result(),"AdvGlossN": AdvGlossN_tracker.result(),"RecGlossX": RecGlossX_tracker.result(), 
            "RecGlossC": RecGlossC_tracker.result(), "RecGlossS": RecGlossS_tracker.result(),
            "fakeX":tf.math.reduce_mean(fakeXcritic),"realX":tf.math.reduce_mean(realXcritic),
            "fakeC":tf.math.reduce_mean(fakeCcritic),"realC":tf.math.reduce_mean(realCcritic),"fakeN":tf.math.reduce_mean(fakeNcritic),
            "realN":tf.math.reduce_mean(realNcritic),"fakeS":tf.math.reduce_mean(fakeScritic),"realS":tf.math.reduce_mean(realScritic)}


    def call(self, X):
        [_,_,fakeS,fakeC,fakeN] = self.Fx(X)
        recX = self.Gz((X,fakeS,fakeC,fakeN))
        return recX, fakeC, fakeS, fakeN

    def plot(self,realX,realC):
        [_,_,fakeS,fakeC,fakeN] = self.Fx(realX,training=False)
        realS = tf.random.normal(mean=0.0,stddev=1.0,shape=[realX.shape[0],self.latentSdim])
        realN = tf.random.normal(mean=0.0,stddev=1.0,shape=[realX.shape[0],self.latentNdim])
        fakeX = self.Gq((realX,realS,realC,realN),training=False)
        recX = self.Gz((realX,fakeS,fakeC,fakeN),training=False)
        return recX, fakeC, fakeS, fakeN, fakeX

    def label_predictor(self, X, realC):
        [_,_,fakeS,fakeC,fakeN] = self.Fx(X)

        realS = tf.random.normal(mean=0.0,stddev=1.0,shape=[fakeS.shape[0],self.latentSdim])
        realN = tf.random.normal(mean=0.0,stddev=1.0,shape=[fakeN.shape[0],self.latentNdim])
        fakeX = self.Gq((X,realS,realC,realN),training=False)
        [_,_,_,recC,_] = self.Q(fakeX,training=False)
        return fakeC, recC
    
    def distribution(self,realX,realC):
        [_,_,fakeS,fakeC,fakeN] = self.Fx(realX,training=False)
        realS = tf.random.normal(mean=0.0,stddev=1.0,shape=[realX.shape[0],self.latentSdim])
        realN = tf.random.normal(mean=0.0,stddev=1.0,shape=[realX.shape[0],self.latentNdim])
        fakeX = self.Gq((realX,realS,realC,realN),training=False)
        [_,_,recS,recC,recN] = self.Q(fakeX,training=False)
        return realS, realN, fakeS, fakeN, recS, recN

    def cycling(self,realX,realC):
        realS = tf.random.normal(mean=0.0,stddev=1.0,shape=[realX.shape[0],self.latentSdim])
        realN = tf.random.normal(mean=0.0,stddev=1.0,shape=[realX.shape[0],self.latentNdim])
        fakeX = self.Gq((realX,realS,realC,realN),training=False)
        [_,_,recS,recC,recN] = self.Q(fakeX,training=True)

        return realS,realN,recS,recC,recN

    def generate(self, X, fakeC_new):
        [_,_,fakeS,fakeC,fakeN] = self.Fx(X)
        recX_new = self.Gz((X,fakeS,fakeC_new,fakeN),training=False)
        return recX_new

    def switchN(self, realX_u, realX_d):
        [_,_,fakeS_u,fakeC_u,fakeN_u] = self.Fx(realX_u)
        [_,_,fakeS_d,fakeC_d,fakeN_d] = self.Fx(realX_d)
        recX_new = self.Gz((realX_d,fakeS_u,fakeC_d,fakeN_u),training=False)
        return recX_new

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
        h0 = keras.Model(X,h)
        h = LeakyReLU(alpha=0.1,name="FxA0")(h)
        h = BatchNormalization(momentum=0.95)(h)
        h = Dropout(0.2,name="FxDO0")(h)

        # Common encoder CNN layers
        for layer in range(self.nAElayers):
            h = Conv1D(self.nZfirst*self.stride**(layer+1),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
            h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
            h = BatchNormalization(momentum=0.95)(h)
            h = Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)

        # Last common CNN layer (no stride, same channels) before branching
        layer = self.nAElayers
        h = Conv1D(self.nZchannels,
            self.kernel,1,padding="same",
            data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
        h1 = keras.Model(X,h)
        h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
        h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
        z = Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)
        # z ---> Zshape = (Zsize,nZchannels)

        layer = 0
        if 'dense' in self.branching:
            # Flatten and branch
            h = Flatten(name="FxFL{:>d}".format(layer+1))(z)
            h = Dense(self.latentZdim,name="FxFW{:>d}".format(layer+1))(h)
            h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
            zf = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)

            # variable s
            # s-average
            h = Dense(self.latentSdim,name="FxFWmuS")(zf)
            Zmu = BatchNormalization(momentum=0.95)(h)

            # s-log std
            h = Dense(self.latentSdim,name="FxFWlvS")(zf)
            Zlv = BatchNormalization(momentum=0.95)(h)

            # variable c
            h = Dense(self.latentCdim,name="FxFWC")(zf)
            Zc = BatchNormalization(momentum=0.95,name="FxBNC")(h)

            # variable n
            Zn = Dense(self.latentNdim,name="FxFWN")(zf)

        elif 'conv' in self.branching:
            # variable s
            # s-average
            Zmu = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(z)
            Zmu = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(Zmu)
            Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(Zmu)
            Zmu = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(Zmu)

            # s-log std
            Zsigma = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(z)
            Zsigma = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zsigma)
            Zsigma = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zsigma)
            Zsigma = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zsigma)

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
            Zn = BatchNormalization(momentum=0.95)(Zn)
            Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)

            # variable s
            for layer in range(1,self.nSlayers):
                # s-average
                Zmu = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(Zmu)
                Zmu = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(Zmu)
                Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(Zmu)
                Zmu = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(Zmu)

                # s-log std
                Zsigma = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(Zsigma)
                Zsigma = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zsigma)
                Zsigma = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zsigma)
                Zsigma = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zsigma)

            # variable c
            for layer in range(1,self.nClayers):
                Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(Zc)
                Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
                Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
                Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)

            # variable n
            for layer in range(1,self.nNlayers):
                Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
                    self.Nkernel,self.Nstride,padding="same",
                    data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(Zn)
                Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
                Zn = BatchNormalization(momentum=0.95)(Zn)
                Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)

            # variable s
            Zmu = Flatten(name="FxFLmuS{:>d}".format(layer+1))(Zmu)
            Zmu = Dense(self.latentSdim,name="FxFWmuS")(Zmu)
            Zmu = LeakyReLU(alpha=0.1)(Zmu)
            Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS")(Zmu)

            # s-sigma
            Zsigma = Flatten(name="FxFLlvS{:>d}".format(layer+1))(Zsigma)
            Zsigma = Dense(self.latentSdim,name="FxFWlvS")(Zsigma)
            Zsigma = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+2))(Zsigma)
            Zsigma = BatchNormalization(momentum=0.95,axis=-1,name="FxBNlvS")(Zsigma)     
            Zsigma = tf.math.sigmoid(Zsigma)

            # variable c
            layer = self.nClayers
            Zc = Flatten(name="FxFLC{:>d}".format(layer+1))(Zc)
            Zc = Dense(1024)(Zc)
            Zc = LeakyReLU(alpha=0.1)(Zc)

            # variable n
            layer = self.nNlayers
            Zn = Flatten(name="FxFLN{:>d}".format(layer+1))(Zn)
            Zn = Dense(self.latentNdim,name="FxFWN")(Zn)

        # variable s
        s = SamplingFxNormSfromSigma()([Zmu,Zsigma])

        # variable c
        c = Dense(self.latentCdim,activation=tf.keras.activations.softmax)(Zc)

        # variable n
        n = BatchNormalization(momentum=0.95)(Zn)

        Fx = keras.Model(X,[Zmu,Zsigma,s,c,n],name="Fx")

        return Fx,h1,h0

    def BuildQ(self):
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
        h2 = keras.Model(X,h)
        h = LeakyReLU(alpha=0.1,name="FxA0")(h)
        h = BatchNormalization(momentum=0.95)(h)
        h = Dropout(0.2,name="FxDO0")(h)

        # Common encoder CNN layers
        for layer in range(self.nAElayers):
            h = Conv1D(self.nZfirst*self.stride**(layer+1),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
            h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
            h = BatchNormalization(momentum=0.95)(h)
            h = Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)

        # Last common CNN layer (no stride, same channels) before branching
        layer = self.nAElayers
        h = Conv1D(self.nZchannels,
            self.kernel,1,padding="same",
            data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
        h3 = keras.Model(X,h)
        h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
        h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
        z = Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)
        # z ---> Zshape = (Zsize,nZchannels)

        layer = 0
        if 'dense' in self.branching:
            # Flatten and branch
            h = Flatten(name="FxFL{:>d}".format(layer+1))(z)
            h = Dense(self.latentZdim,name="FxFW{:>d}".format(layer+1))(h)
            h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
            zf = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)

            # variable s
            # s-average
            h = Dense(self.latentSdim,name="FxFWmuS")(zf)
            Zmu = BatchNormalization(momentum=0.95)(h)

            # s-log std
            h = Dense(self.latentSdim,name="FxFWlvS")(zf)
            Zlv = BatchNormalization(momentum=0.95)(h)

            # variable c
            h = Dense(self.latentCdim,name="FxFWC")(zf)
            Zc = BatchNormalization(momentum=0.95,name="FxBNC")(h)

            # variable n
            Zn = Dense(self.latentNdim,name="FxFWN")(zf)

        elif 'conv' in self.branching:
            # variable s
            # s-average
            Zmu = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(z)
            Zmu = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(Zmu)
            Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(Zmu)
            Zmu = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(Zmu)

            # s-log std
            Zsigma = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(z)
            Zsigma = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zsigma)
            Zsigma = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zsigma)
            Zsigma = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zsigma)

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
            Zn = BatchNormalization(momentum=0.95)(Zn)
            Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)

            # variable s
            for layer in range(1,self.nSlayers):
                # s-average
                Zmu = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(Zmu)
                Zmu = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(Zmu)
                Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(Zmu)
                Zmu = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(Zmu)

                # s-log std
                Zsigma = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(Zsigma)
                Zsigma = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zsigma)
                Zsigma = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zsigma)
                Zsigma = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zsigma)

            # variable c
            for layer in range(1,self.nClayers):
                Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(Zc)
                Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
                Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
                Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)

            # variable n
            for layer in range(1,self.nNlayers):
                Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
                    self.Nkernel,self.Nstride,padding="same",
                    data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(Zn)
                Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
                Zn = BatchNormalization(momentum=0.95)(Zn)
                Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)

            # variable s
            Zmu = Flatten(name="FxFLmuS{:>d}".format(layer+1))(Zmu)
            Zmu = Dense(self.latentSdim,name="FxFWmuS")(Zmu)
            Zmu = LeakyReLU(alpha=0.1)(Zmu)
            Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS")(Zmu)

            # s-sigma
            Zsigma = Flatten(name="FxFLlvS{:>d}".format(layer+1))(Zsigma)
            Zsigma = Dense(self.latentSdim,name="FxFWlvS")(Zsigma)
            Zsigma = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+2))(Zsigma)
            Zsigma = BatchNormalization(momentum=0.95,axis=-1,name="FxBNlvS")(Zsigma)     
            Zsigma = tf.math.sigmoid(Zsigma)

            # variable c
            layer = self.nClayers
            Zc = Flatten(name="FxFLC{:>d}".format(layer+1))(Zc)
            Zc = Dense(1024)(Zc)
            Zc = LeakyReLU(alpha=0.1)(Zc)

            # variable n
            layer = self.nNlayers
            Zn = Flatten(name="FxFLN{:>d}".format(layer+1))(Zn)
            Zn = Dense(self.latentNdim,name="FxFWN")(Zn)

        # variable s
        s = SamplingFxNormSfromSigma()([Zmu,Zsigma])

        # variable c
        c = Dense(self.latentCdim,activation=tf.keras.activations.softmax)(Zc)

        # variable n
        n = BatchNormalization(momentum=0.95)(Zn)

        Q = keras.Model(X,[Zmu,Zsigma,s,c,n],name="Q")

        return Q,h2,h3

    def BuildGz(self):
        """
            Conv1D Gz structure
            https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

        """
        Xin = Input(shape=self.Xshape,name="Xin")
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

            z = concatenate([GzS.output,GzC.output,GzN.output])
            Gz = Reshape((self.Zsize,self.nZchannels))(z)
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)
            Gz = LeakyReLU(alpha=0.1,name="GzA0".format(layer+1))(Gz)

        elif 'conv' in self.branching:
            # variable s
            Zs = Dense(self.Ssize*self.nSchannels)(s)
            Zs = LeakyReLU(alpha=0.1)(Zs)
            Zs = BatchNormalization(momentum=0.95)(Zs)
            Zs = Reshape((self.Ssize,self.nSchannels))(Zs)

            for layer in range(1,self.nSlayers):
                Zs = Conv1DTranspose(int(self.nSchannels*self.Sstride**(-layer)),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last")(Zs)
                Zs = LeakyReLU(alpha=0.1)(Zs)
                Zs = BatchNormalization(momentum=0.95)(Zs)
                Zs = Dropout(0.2,name="GzDOS{:>d}".format(layer))(Zs)
            Zs = Conv1DTranspose(int(self.nSchannels*self.Sstride**(-self.nSlayers)),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last")(Zs)
            Zs = LeakyReLU(alpha=0.1)(Zs)
            Zs = BatchNormalization(momentum=0.95,name="GzBNS{:>d}".format(self.nSlayers))(Zs)
            Zs = Dropout(0.2)(Zs)
            GzS = keras.Model(s,Zs)


            # variable c
            Zc = Dense(self.Csize*self.nCchannels)(c)
            Zc = LeakyReLU(alpha=0.1,)(Zc)
            Zc = BatchNormalization(momentum=0.95)(Zc)
            Zc = Reshape((self.Csize,self.nCchannels))(Zc)
            for layer in range(1,self.nClayers):
                Zc = Conv1DTranspose(int(self.nCchannels*self.Cstride**(-layer)),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last")(Zc)
                Zc = LeakyReLU(alpha=0.1)(Zc)
                Zc = BatchNormalization(momentum=0.95)(Zc)
                Zc = Dropout(0.2)(Zc)
            Zc = Conv1DTranspose(int(self.nCchannels*self.Cstride**(-self.nClayers)),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last")(Zc)
            Zc = LeakyReLU(alpha=0.1)(Zc)
            Zc = BatchNormalization(momentum=0.95)(Zc)
            Zc = Dropout(0.2)(Zc)
            GzC = keras.Model(c,Zc)

            # variable n
            Zn = Dense(self.Nsize*self.nNchannels)(n)
            Zn = LeakyReLU(alpha=0.1)(Zn)
            Zn = BatchNormalization(momentum=0.95)(Zn)
            Zn = Reshape((self.Nsize,self.nNchannels))(Zn)
            for layer in range(1,self.nNlayers):
                Zn = Conv1DTranspose(int(self.nNchannels*self.Nstride**(-layer)),
                    self.Nkernel,self.Nstride,padding="same",
                    data_format="channels_last")(Zn)
                Zn = LeakyReLU(alpha=0.1)(Zn)
                Zn = BatchNormalization(momentum=0.95)(Zn)
                Zn = Dropout(0.2)(Zn)
            Zn = Conv1DTranspose(int(self.nNchannels*self.Nstride**(-self.nNlayers)),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last")(Zn)
            Zn = LeakyReLU(alpha=0.1)(Zn)
            Zn = BatchNormalization(momentum=0.95)(Zn)
            Zn = Dropout(0.2)(Zn)
            GzN = keras.Model(n,Zn)

            Gz = concatenate([GzS.output,GzC.output,GzN.output])
            Gz = Conv1DTranspose(self.nZchannels,
                    self.kernel,1,padding="same",
                    data_format="channels_last")(Gz)
            Gz1 = self.h1(Xin)
            Gz = Add()([Gz1, Gz])
            Gz = LeakyReLU(alpha=0.1)(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)

        for layer in range(self.nAElayers-1):
            Gz = Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                self.kernel,self.stride,padding="same",use_bias=False)(Gz)
            Gz = LeakyReLU(alpha=0.1)(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)

        layer = self.nAElayers-1
        Gz = Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                self.kernel,self.stride,padding="same",use_bias=False)(Gz)
        Gz0 = self.h0(Xin)
        Gz = Add()([Gz0, Gz])
        Gz = LeakyReLU(alpha=0.1,name="GzA{:>d}".format(layer+1))(Gz)
        Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)

        layer = self.nAElayers
        X = Conv1DTranspose(self.nXchannels,self.kernel,1,
            padding="same",activation='tanh',use_bias=False)(Gz)

        Gz = keras.Model(inputs=[Xin,GzS.input,GzC.input,GzN.input],outputs=X,name="Gz")
        return Gz

    def BuildGq(self):
        """
            Conv1D Gz structure
            https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

        """
        Xin = Input(shape=self.Xshape,name="Xin")
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

            z = concatenate([GzS.output,GzC.output,GzN.output])
            Gz = Reshape((self.Zsize,self.nZchannels))(z)
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)
            Gz = LeakyReLU(alpha=0.1,name="GzA0".format(layer+1))(Gz)

        elif 'conv' in self.branching:
            # variable s
            Zs = Dense(self.Ssize*self.nSchannels)(s)
            Zs = LeakyReLU(alpha=0.1)(Zs)
            Zs = BatchNormalization(momentum=0.95)(Zs)
            Zs = Reshape((self.Ssize,self.nSchannels))(Zs)

            for layer in range(1,self.nSlayers):
                Zs = Conv1DTranspose(int(self.nSchannels*self.Sstride**(-layer)),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last")(Zs)
                Zs = LeakyReLU(alpha=0.1)(Zs)
                Zs = BatchNormalization(momentum=0.95)(Zs)
                Zs = Dropout(0.2,name="GzDOS{:>d}".format(layer))(Zs)
            Zs = Conv1DTranspose(int(self.nSchannels*self.Sstride**(-self.nSlayers)),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last")(Zs)
            Zs = LeakyReLU(alpha=0.1)(Zs)
            Zs = BatchNormalization(momentum=0.95,name="GzBNS{:>d}".format(self.nSlayers))(Zs)
            Zs = Dropout(0.2)(Zs)
            GzS = keras.Model(s,Zs)


            # variable c
            Zc = Dense(self.Csize*self.nCchannels)(c)
            Zc = LeakyReLU(alpha=0.1,)(Zc)
            Zc = BatchNormalization(momentum=0.95)(Zc)
            Zc = Reshape((self.Csize,self.nCchannels))(Zc)
            for layer in range(1,self.nClayers):
                Zc = Conv1DTranspose(int(self.nCchannels*self.Cstride**(-layer)),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last")(Zc)
                Zc = LeakyReLU(alpha=0.1)(Zc)
                Zc = BatchNormalization(momentum=0.95)(Zc)
                Zc = Dropout(0.2)(Zc)
            Zc = Conv1DTranspose(int(self.nCchannels*self.Cstride**(-self.nClayers)),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last")(Zc)
            Zc = LeakyReLU(alpha=0.1)(Zc)
            Zc = BatchNormalization(momentum=0.95)(Zc)
            Zc = Dropout(0.2)(Zc)
            GzC = keras.Model(c,Zc)

            # variable n
            Zn = Dense(self.Nsize*self.nNchannels)(n)
            Zn = LeakyReLU(alpha=0.1)(Zn)
            Zn = BatchNormalization(momentum=0.95)(Zn)
            Zn = Reshape((self.Nsize,self.nNchannels))(Zn)
            for layer in range(1,self.nNlayers):
                Zn = Conv1DTranspose(int(self.nNchannels*self.Nstride**(-layer)),
                    self.Nkernel,self.Nstride,padding="same",
                    data_format="channels_last")(Zn)
                Zn = LeakyReLU(alpha=0.1)(Zn)
                Zn = BatchNormalization(momentum=0.95)(Zn)
                Zn = Dropout(0.2)(Zn)
            Zn = Conv1DTranspose(int(self.nNchannels*self.Nstride**(-self.nNlayers)),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last")(Zn)
            Zn = LeakyReLU(alpha=0.1)(Zn)
            Zn = BatchNormalization(momentum=0.95)(Zn)
            Zn = Dropout(0.2)(Zn)
            GzN = keras.Model(n,Zn)

            Gz = concatenate([GzS.output,GzC.output,GzN.output])
            Gz = Conv1DTranspose(self.nZchannels,
                    self.kernel,1,padding="same",
                    data_format="channels_last")(Gz)
            Gz1 = self.h3(Xin)
            Gz = Add()([Gz1, Gz])
            Gz = LeakyReLU(alpha=0.1)(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)

        for layer in range(self.nAElayers-1):
            Gz = Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                self.kernel,self.stride,padding="same",use_bias=False)(Gz)
            Gz = LeakyReLU(alpha=0.1)(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)

        layer = self.nAElayers-1
        Gz = Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                self.kernel,self.stride,padding="same",use_bias=False)(Gz)
        Gz0 = self.h2(Xin)
        Gz = Add()([Gz0, Gz])
        Gz = LeakyReLU(alpha=0.1,name="GzA{:>d}".format(layer+1))(Gz)
        Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)

        layer = self.nAElayers
        X = Conv1DTranspose(self.nXchannels,self.kernel,1,
            padding="same",activation='tanh',use_bias=False)(Gz)

        Gq = keras.Model(inputs=[Xin,GzS.input,GzC.input,GzN.input],outputs=X,name="Gq")
        return Gq

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
        h = Dense(1024)(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = BatchNormalization(momentum=0.95)(h)
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
        h = BatchNormalization(momentum=0.95)(h)
        h = Dropout(0.25)(h)
        Pc = Dense(1,activation=tf.keras.activations.sigmoid)(h)
        Dc = keras.Model(c,Pc,name="Dc")
        return Dc


    def BuildDn(self):
        """
            Dense discriminator structure
        """
        n = Input(shape=(self.latentNdim,))
        h = Dense(3000)(n) #kernel_constraint=ClipConstraint(self.clipValue)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(0.25)(h)
        h = Dense(3000)(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = BatchNormalization(momentum=0.95)(h)
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
        h = BatchNormalization(momentum=0.95)(h)
        h = Dropout(0.25)(h)
        Ps = Dense(1,activation=tf.keras.activations.sigmoid)(h)
        Ds = keras.Model(s,Ps,name="Ds")
        return Ds

    
    def DumpModels(self):
        self.Fx.save("./checkpoint_ultimo/14_04/Fx",save_format="tf")
        self.Gz.save("./checkpoint_ultimo/14_04/Gz",save_format="tf")
        self.Dx.save("./checkpoint_ultimo/14_04/Dx",save_format="tf")
        self.Ds.save("./checkpoint_ultimo/14_04/Ds",save_format="tf")
        self.Dn.save("./checkpoint_ultimo/14_04/Dn",save_format="tf")
        self.Dc.save("./checkpoint_ultimo/14_04/Dc",save_format="tf")
        self.Q.save("./checkpoint_ultimo/14_04/Q",save_format="tf")
        self.Gq.save("./checkpoint_ultimo/14_04/Gq",save_format="tf")
        self.h0.save("./checkpoint_ultimo/14_04/h0",save_format="tf")
        self.h1.save("./checkpoint_ultimo/14_04/h1",save_format="tf")
        self.h2.save("./checkpoint_ultimo/14_04/h2",save_format="tf")
        self.h3.save("./checkpoint_ultimo/14_04/h3",save_format="tf")
        return

def Main(DeviceName):

    options = ParseOptions()

    if not options['cuda']:
        DeviceName = "/cpu:0"

    with tf.device(DeviceName):
        optimizers = {}
        optimizers['DxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['DcOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['DsOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['DnOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['FxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['QOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['GzOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['GqOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)

        losses = {}
        losses['AdvDlossWGAN'] = WassersteinDiscriminatorLoss
        losses['AdvGlossWGAN'] = WassersteinGeneratorLoss
        losses['AdvDlossGAN'] = tf.keras.losses.BinaryCrossentropy()
        losses['AdvGlossGAN'] = tf.keras.losses.BinaryCrossentropy()
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

        #GiorgiaGAN.build(input_shape=(options['batchSize'], options['Xsize'], options['nXchannels']))

        # GiorgiaGAN.save("GiorgiaGAN_ultimo")

        #GiorgiaGAN.save_weights("ckpt")

        PlotLoss(history) # Plot loss

        # PlotReconstructedTHs(GiorgiaGAN,Xvld) # Plot reconstructed time-histories

        # PlotTHSGoFs(GiorgiaGAN,Xvld) # Plot reconstructed time-histories

        # PlotClassificationMetrics(GiorgiaGAN,Xvld) # Plot classification metrics

        # Xtrn = {}
        # Xvld = {}
        # for i in range(options['latentCdim']):
        #     Xtrn['Xtrn_%d' % i], Xvld['Xvld_%d' % i], _  = mdof.Load_Un_Damaged(i,**options)

        # for i in range(options['latentCdim']):
        #     PlotBatchGoFs(GiorgiaGAN,Xtrn['Xtrn_%d' % i],Xvld['Xvld_%d' % i],i)

        # for i in range(1,options['latentCdim']):
        #     PlotSwitchedTHs(GiorgiaGAN,Xvld['Xvld_%d' % 0],Xvld['Xvld_%d' % i],i) # Plot switched time-histories
        #subprocess.run("panel serve plot_panel.py --show", shell=True)
        #PlotBokeh(GiorgiaGAN,Xvld,**options)
        
        

        
if __name__ == '__main__':
    DeviceName = tf.test.gpu_device_name()
    Main(DeviceName)

