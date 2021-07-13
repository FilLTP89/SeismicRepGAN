# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
__author__ = "Filippo Gatti Giorgia Colombera"
__copyright__ = "Copyright 2021, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

import sys
import os
from os.path import join as opj
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import numpy as np
import math as mt

import tensorflow as tf
#tf.config.run_functions_eagerly(True)
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Layer
from tensorflow.keras.layers import Lambda, Concatenate, concatenate, Activation, ZeroPadding1D
from tensorflow.keras.layers import LeakyReLU, ReLU, Softmax
from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.constraints import Constraint, min_max_norm
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# import tensorflow_probability.distributions as tfd
import timeit

import scipy
from scipy import signal

import seaborn as sn
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import GridSearchCV

import obspy.signal
from obspy.signal.tf_misfit import plot_tf_gofs, eg, pg

import matplotlib.mlab as mlab

from scipy.stats import norm

import itertools

# import wandb
# wandb.init()

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Tahoma']
families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']

import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from numpy.random import randn
from numpy.random import randint

from tensorflow.python.eager import context
import kerastuner as kt
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel

from numpy.linalg import norm
import MDOFload as mdof
import matplotlib.pyplot as plt

from tensorflow.python.util.tf_export import tf_export
from copy import deepcopy

AdvDLoss_tracker = keras.metrics.Mean(name="loss")
AdvDlossX_tracker = keras.metrics.Mean(name="loss")
AdvDlossC_tracker = keras.metrics.Mean(name="loss")
AdvDlossS_tracker = keras.metrics.Mean(name="loss")
AdvDlossN_tracker = keras.metrics.Mean(name="loss")
AdvDlossPenGradX_tracker = keras.metrics.Mean(name="loss")
AdvGLoss_tracker = keras.metrics.Mean(name="loss")
AdvGlossX_tracker = keras.metrics.Mean(name="loss")
AdvGlossC_tracker = keras.metrics.Mean(name="loss")
AdvGlossS_tracker = keras.metrics.Mean(name="loss")
AdvGlossN_tracker = keras.metrics.Mean(name="loss")
RecGlossX_tracker = keras.metrics.Mean(name="loss")
RecGlossC_tracker = keras.metrics.Mean(name="loss")
RecGlossS_tracker = keras.metrics.Mean(name="loss")

#gpu_devices = tf.config.experimental.list_physical_devices('GPU')


checkpoint_dir = "./ckpt"
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
    parser.add_argument("--epochs",type=int,default=1000,help='Number of epochs')
    parser.add_argument("--Xsize",type=int,default=1024,help='Data space size')
    parser.add_argument("--nX",type=int,default=512,help='Number of signals')
    parser.add_argument("--nXchannels",type=int,default=2,help="Number of data channels")
    parser.add_argument("--nAElayers",type=int,default=3,help='Number of AE CNN layers')
    parser.add_argument("--nDlayers",type=int,default=10,help='Number of D CNN layers')
    parser.add_argument("--kernel",type=int,default=3,help='CNN kernel size')
    parser.add_argument("--stride",type=int,default=2,help='CNN stride')
    parser.add_argument("--nZfirst",type=int,default=8,help="Initial number of channels")
    parser.add_argument("--branching",type=str,default='conv',help='conv or dens')
    parser.add_argument("--latentSdim",type=int,default=256,help="Latent space s dimension")
    parser.add_argument("--latentCdim",type=int,default=5,help="Number of classes")
    parser.add_argument("--latentNdim",type=int,default=64,help="Latent space n dimension")
    parser.add_argument("--nSlayers",type=int,default=3,help='Number of S-branch CNN layers')
    parser.add_argument("--nClayers",type=int,default=3,help='Number of C-branch CNN layers')
    parser.add_argument("--nNlayers",type=int,default=3,help='Number of N-branch CNN layers')
    parser.add_argument("--Skernel",type=int,default=7,help='CNN kernel of S-branch branch')
    parser.add_argument("--Ckernel",type=int,default=7,help='CNN kernel of C-branch branch')
    parser.add_argument("--Nkernel",type=int,default=7,help='CNN kernel of N-branch branch')
    parser.add_argument("--Sstride",type=int,default=4,help='CNN stride of S-branch branch')
    parser.add_argument("--Cstride",type=int,default=4,help='CNN stride of C-branch branch')
    parser.add_argument("--Nstride",type=int,default=4,help='CNN stride of N-branch branch')
    parser.add_argument("--batchSize",type=int,default=128,help='input batch size')    
    parser.add_argument("--nCritic",type=int,default=5,help='number of discriminator training steps')
    parser.add_argument("--clipValue",type=float,default=0.01,help='clip weight for WGAN')
    parser.add_argument("--dataroot_1",type=str,default="/gpfs/workdir/invsem07/damaged_1_8P",help="Data root folder - Pirellone") #damaged_1_0
    parser.add_argument("--dataroot_2",type=str,default="/gpfs/workdir/invsem07/damaged_1_3",help="Data root folder - Edificio a taglio") 
    parser.add_argument("--idChannels",type=int,nargs='+',default=[1,39],help="Channel 1") #1,2,3,4
    parser.add_argument("--nParams",type=str,default=2,help="Number of parameters")
    parser.add_argument("--case",type=str,default="train_model",help="case")
    parser.add_argument("--avu",type=str,nargs='+',default="U",help="case avu")
    parser.add_argument("--pb",type=str,default="BC",help="case pb")
    parser.add_argument("--CreateData",action='store_true',default=True,help='Create data flag')
    parser.add_argument("--cuda",action='store_true',default=False,help='Use cuda powered GPU')
    parser.add_argument('--tdof',default='A',help='Signal content (e.g. U, V, A) (mdof database only)') # eventually 'nargs='+' if different types of signals (e.g. displacements, velocities, accelerations etc.) are considered
    parser.add_argument('--wtdof',nargs='+',default=[3],help='Specify the connection between wdof and tdof (mdof database only)')
    parser.add_argument('--dtm',type=float,default=0.001,help='time-step [s]')
    parser.add_argument('--ntm',type=int,default=999,help='Number of time steps')
    parser.add_argument('--batchPercent', type=int,nargs='+', default=[0.8,0.1,0.1], help='train/test/validation %')
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
    def _merge_function(self, inputs, **kwargs):
        alpha = tf.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class SamplingFxS(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        # z_mean, z_log_var = inputs
        z_mean, z_std = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_mean + z_std * epsilon

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
    sigma = pred[:, n_dims:]
    # logsigma = pred[:, n_dims:]
    mse = -0.5*tf.sum(tf.square((true-mu)/sigma),axis=1)
    # mse = -0.5*tf.sum(tf.square((true-mu)/tf.exp(logsigma)),axis=1)

    sigma_trace = -tf.sum(tf.log(sigma), axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood = mse+sigma_trace+log2pi

    return tf.mean(-log_likelihood)

def MutualInfoLoss(c, c_given_x):
    """The mutual information metric we aim to minimize"""
    eps = 1e-8
    conditional_entropy = tf.mean(-tf.sum(tf.log(c_given_x+eps)*c,axis=1))
    entropy = tf.mean(-tf.sum(tf.log(c+eps)*c,axis=1))

    return conditional_entropy + entropy

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
        self.Dx = self.build_Dx()
        self.Dc = self.build_Dc()
        self.Ds = self.build_Ds()
        self.Dn = self.build_Dn()
        """
            Build Fx/Gz (generators)
        """

        self.Fx, self.Qs, self.Qc  = self.build_Fx()
        self.Gz = self.build_Gz()
        self.Fx.save("Fx.h5")
        self.Qs.save("Qs.h5")
        self.Qc.save("Qc.h5")
        self.Gz.save("Gz.h5")
        self.Ds.save("Ds.h5")
        self.Dn.save("Dn.h5")
        self.Dc.save("Dc.h5")

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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config    

    @property
    def metrics(self):
        return [AdvDLoss_tracker,AdvGLoss_tracker,AdvDlossX_tracker,AdvDlossC_tracker,AdvDlossS_tracker,AdvDlossN_tracker,
            RecGlossX_tracker,RecGlossC_tracker,RecGlossS_tracker]

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
        # Adversarial ground truths
        realXcritic = self.Dx(realX)
        realBCE = tf.ones_like(realXcritic)
        fakeBCE = tf.zeros_like(realXcritic)
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

            with tf.GradientTape(persistent=True) as tape:

                realS = tf.random.normal(mean=0.0,stddev=0.5,shape=[self.batchSize,self.latentSdim])
                realN = tf.random.normal(mean=0.0,stddev=0.3,shape=[self.batchSize,self.latentNdim])

                # Generate fake latent code from real signals
                [fakeS,fakeC,fakeN] = self.Fx(realX) # encoded z = Fx(X)
                fakeN = tf.clip_by_value(fakeN,-1.0,1.0)

                # Generate fake signals from real latent code
                fakeX = self.Gz((realS,realC,realN)) # fake X = Gz(Fx(X))

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
                AdvDlossX  = self.AdvDlossGAN(realBCE,realXcritic)*self.PenAdvXloss
                AdvDlossX += self.AdvDlossGAN(fakeBCE,fakeXcritic)*self.PenAdvXloss
                #AdvDlossX  = self.AdvDlossWGAN(realXcritic,fakeXcritic)*self.PenAdvXloss
                AdvDlossC = self.AdvDlossWGAN(realCcritic,fakeCcritic)*self.PenAdvCloss
                AdvDlossS = self.AdvDlossWGAN(realScritic,fakeScritic)*self.PenAdvSloss
                #AdvDlossN  = self.AdvDlossGAN(realBCE,realNcritic)*self.PenAdvNloss
                #AdvDlossN += self.AdvDlossGAN(fakeBCE,fakeNcritic)*self.PenAdvNloss
                AdvDlossN = self.AdvDlossWGAN(realNcritic,fakeNcritic)*self.PenAdvNloss
                #AdvDlossPenGradX = self.GradientPenaltyX(self.batchSize,realX,fakeX)*self.PenGradX

                AdvDloss = AdvDlossX + AdvDlossC + AdvDlossS + AdvDlossN #AdvDlossPenGradX

            # Get the gradients w.r.t the discriminator loss
            gradDx, gradDc, gradDs, gradDn = tape.gradient(AdvDloss,
                (self.Dx.trainable_variables, self.Dc.trainable_variables,
                self.Ds.trainable_variables, self.Dn.trainable_variables))
            # Update the weights of the discriminator using the discriminator optimizer
            self.DxOpt.apply_gradients(zip(gradDx,self.Dx.trainable_variables))
            self.DcOpt.apply_gradients(zip(gradDc,self.Dc.trainable_variables))
            self.DsOpt.apply_gradients(zip(gradDs,self.Ds.trainable_variables))
            self.DnOpt.apply_gradients(zip(gradDn,self.Dn.trainable_variables))

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
            # Generate fake latent code from real signal
            [fakeS,fakeC,fakeN] = self.Fx(realX) # encoded z = Fx(X)
            fakeN = tf.clip_by_value(fakeN,-1,1)

            # Discriminator determines validity of the real and fake S
            fakeScritic = self.Ds(fakeS)

            # Discriminator determines validity of the real and fake C
            fakeCcritic = self.Dc(fakeC)

            # Discriminator determines validity of the real and fake N
            fakeNcritic = self.Dn(fakeN)

            fakeX = self.Gz((realS,realC,realN)) # fake X = Gz(Fx(X))

            # Discriminator determines validity of the real and fake X
            fakeXcritic = self.Dx(fakeX)

            # Reconstruction
            recX = self.Gz((fakeS,fakeC,fakeN))
            recS = self.Qs(fakeX)
            # recSmu,recSsigma = self.Qs(fakeX)
            # self.QsDist = tfd.MultivariateNormalDiag(loc=recSmu,scale_diag=recSsigma)
            recC = self.Qc(fakeX)

            # Adversarial ground truths
            realBCE = tf.ones_like(fakeXcritic)
            AdvGlossX = self.AdvGlossGAN(realBCE,fakeXcritic)*self.PenAdvXloss
            #AdvGlossX = self.AdvGlossWGAN(fakeXcritic)*self.PenAdvXloss
            AdvGlossC = self.AdvGlossWGAN(fakeCcritic)*self.PenAdvCloss
            AdvGlossS = self.AdvGlossWGAN(fakeScritic)*self.PenAdvSloss
            AdvGlossN = self.AdvGlossWGAN(fakeNcritic)*self.PenAdvNloss
            RecGlossX = self.RecXloss(realX,recX)*self.PenRecXloss
            RecGlossS = self.RecSloss(realS,recS)*self.PenRecSloss
            RecGlossC = self.RecCloss(realC,recC)*self.PenRecCloss
            
            AdvGloss = AdvGlossX + AdvGlossC + AdvGlossS + AdvGlossN + RecGlossX + RecGlossC + RecGlossS

        # Get the gradients w.r.t the generator loss
        gradFx, gradGz, gradQs, gradQc = tape.gradient(AdvGloss,
            (self.Fx.trainable_variables,self.Gz.trainable_variables,
             self.Qs.trainable_variables,self.Qc.trainable_variables))

        # Update the weights of the generator using the generator optimizer
        self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))
        self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))
        self.QsOpt.apply_gradients(zip(gradQs,self.Fx.trainable_variables))
        self.QcOpt.apply_gradients(zip(gradQc,self.Gz.trainable_variables))

        # Compute our own metrics
        AdvDLoss_tracker.update_state(AdvDloss)
        AdvGLoss_tracker.update_state(AdvGloss)
        AdvDlossX_tracker.update_state(AdvDlossX)
        AdvDlossC_tracker.update_state(AdvDlossC)
        AdvDlossS_tracker.update_state(AdvDlossS)
        AdvDlossN_tracker.update_state(AdvDlossN)
        RecGlossX_tracker.update_state(RecGlossX)
        RecGlossC_tracker.update_state(RecGlossC)
        RecGlossS_tracker.update_state(RecGlossS)
        #loss_tracker.update_state(loss)
        #mae_metric.update_state(y, y_pred)
        #return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

        return {"AdvDloss": AdvDLoss_tracker.result(),"AdvGloss": AdvGLoss_tracker.result(), "AdvDlossX": AdvDlossX_tracker.result(),
            "AdvDlossC": AdvDlossC_tracker.result(),"AdvDlossS": AdvDlossS_tracker.result(),"AdvDlossN": AdvDlossN_tracker.result(),
            "RecGlossX": RecGlossX_tracker.result(), "RecGlossC": RecGlossC_tracker.result(), "RecGlossS": RecGlossS_tracker.result()}

    def call(self, X):
        [fakeS,fakeC,fakeN] = self.Fx(X)
        fakeX = self.Gz((fakeS,fakeC,fakeN))
        return fakeX, fakeC

    def build_Fx(self):
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
        h = BatchNormalization(momentum=0.95,name="FxBN0")(h)
        h = LeakyReLU(alpha=0.1,name="FxA0")(h)
        h = Dropout(0.2,name="FxDO0")(h)

        # Common encoder CNN layers
        for layer in range(self.nAElayers):
            # h = Conv1D(self.nZfirst*self.nSchannels*self.stride**(-self.nAElayers+layer+1),
            h = Conv1D(self.nZfirst*self.stride**(layer+1),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
            h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
            h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
            h = Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)

        # Last common CNN layer (no stride, same channels) before branching
        layer = self.nAElayers
        # h = Conv1D(self.nZfirst*self.nSchannels*self.stride**(-self.nAElayers+layer+1),
        h = Conv1D(self.nZchannels,
            self.kernel,1,padding="same",
            data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
        h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
        h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
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
            Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(Zmu)
            Zmu = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(Zmu)
            Zmu = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(Zmu)

            # s-log std
            Zlv = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(z)
            Zlv = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zlv)
            Zlv = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zlv)
            Zlv = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zlv)

            # variable c
            Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(z)
            Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
            Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
            Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)

            # variable n
            Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
                    self.Nkernel,self.Nstride,padding="same",
                    data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(z)
            Zn = BatchNormalization(momentum=0.95,name="FxBNN{:>d}".format(layer+1))(Zn)
            Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
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
                Zlv = Conv1D(self.nZchannels*self.Sstride**(layer+1),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(Zlv)
                Zlv = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zlv)
                Zlv = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zlv)
                Zlv = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zlv)

            # variable c
            for layer in range(1,self.nClayers):
                Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(Zc)
                Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
                Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
                Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)

            # variable n
            for layer in range(1,self.nNlayers):
                Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
                    self.Nkernel,self.Nstride,padding="same",
                    data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(Zn)
                Zn = BatchNormalization(momentum=0.95,name="FxBNN{:>d}".format(layer+1))(Zn)
                Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
                Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)

            # variable s
            # layer = self.nSlayers
            # Zmu = Conv1D(self.nZchannels*self.Sstride**(layer+1),
            #     self.Skernel,self.Sstride,padding="same",
            #     data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(Zmu)
            # Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(Zmu)
            # Zmu = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(Zmu)
            # Zmu = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(Zmu)
            Zmu = Flatten(name="FxFLmuS{:>d}".format(layer+1))(Zmu)
            Zmu = Dense(self.latentSdim,name="FxFWmuS")(Zmu)
            Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS")(Zmu)

            # s-log std
            # Zlv = Conv1D(self.nZchannels*self.Sstride**(layer+1),
            #         self.Skernel,self.Sstride,padding="same",
            #     data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(Zlv)
            # Zlv = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zlv)
            # Zlv = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zlv)
            # Zlv = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zlv)
            Zlv = Flatten(name="FxFLlvS{:>d}".format(layer+1))(Zlv)
            Zsigma = Dense(self.latentSdim,activation=tf.keras.activations.sigmoid,
                name="FxFWlvS")(Zlv)
            # Zlv = Dense(self.latentSdim,name="FxFWlvS")(Zlv)
            # Zlv = BatchNormalization(momentum=0.95,name="FxBNlvS")(Zlv)
            # CLIP Zlv
            Zsigma_clip = tf.clip_by_value(Zsigma,-1.0,1.0)

            # variable c
            layer = self.nClayers
            # Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
            #         self.Ckernel,self.Cstride,padding="same",
            #         data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(Zc)
            # Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
            # Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
            # Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)
            Zc = Flatten(name="FxFLC{:>d}".format(layer+1))(Zc)
            Zc = Dense(self.latentCdim,name="FxFWC")(Zc)
            Zc = BatchNormalization(momentum=0.95,name="FxBNC")(Zc)

            # variable n
            layer = self.nNlayers
            # Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
            #     self.Nkernel,self.Nstride,padding="same",
            #     data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(Zn)
            # Zn = BatchNormalization(momentum=0.95,name="FxBNN{:>d}".format(layer+1))(Zn)
            # Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
            # Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)
            Zn = Flatten(name="FxFLN{:>d}".format(layer+1))(Zn)
            Zn = Dense(self.latentNdim,name="FxFWN")(Zn)

        # variable s
        s = SamplingFxS()([Zmu,Zsigma_clip])
        QsX = Concatenate(axis=-1)([Zmu,Zsigma])

        # variable c
        c   = Softmax(name="FxAC")(Zc)
        QcX = Softmax(name="QcAC")(Zc)

        # variable n
        n = BatchNormalization(momentum=0.95,name="FxBNN")(Zn)

        Fx = keras.Model(X,[s,c,n],name="Fx")
        Qs = keras.Model(X,QsX,name="Qs")
        # Qs = keras.Model(X,[Zmu,Zsigma],name="Qs")
        Qc = keras.Model(X,QcX,name="Qc")

        return Fx,Qs,Qc

    def build_Gz(self):
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
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)
            Gz = LeakyReLU(alpha=0.1,name="GzA0".format(layer+1))(Gz)

        elif 'conv' in self.branching:
            # variable s
            Zs = Dense(self.Ssize*self.nSchannels,name="GzFWS0")(s)
            Zs = BatchNormalization(name="GzBNS0")(Zs)
            Zs = Reshape((self.Ssize,self.nSchannels))(Zs)

            for layer in range(1,self.nSlayers):
                Zs = Conv1DTranspose(int(self.nSchannels*self.Sstride**(-layer)),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last",name="GzCNNS{:>d}".format(layer))(Zs)
                Zs = BatchNormalization(momentum=0.95,name="GzBNS{:>d}".format(layer))(Zs)
                Zs = LeakyReLU(alpha=0.1,name="GzAS{:>d}".format(layer))(Zs)
                Zs = Dropout(0.2,name="GzDOS{:>d}".format(layer))(Zs)
            Zs = Conv1DTranspose(int(self.nSchannels*self.Sstride**(-self.nSlayers)),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="GzCNNS{:>d}".format(self.nSlayers))(Zs)
            Zs = BatchNormalization(momentum=0.95,name="GzBNS{:>d}".format(self.nSlayers))(Zs)
            Zs = LeakyReLU(alpha=0.1,name="GzAS{:>d}".format(self.nSlayers))(Zs)
            Zs = Dropout(0.2,name="GzDOS{:>d}".format(self.nSlayers))(Zs)
            GzS = keras.Model(s,Zs)

            # # s average
            # Zmu = Dense(self.Ssize*self.nSchannels,name="GzFW0")(s)
            # Zmu = BatchNormalization(name="GzBNS0")(Zmu)
            # Zmu = Reshape((self.Ssize,self.nSchannels))(Zmu)
            # Gsmu = keras.Model(s,Zmu)

            # # s logsigma
            # Zlv = Dense(self.Ssize*self.nSchannels,name="GzFW0")(s)
            # Zlv = BatchNormalization(name="GzBNS0")(Zlv)
            # Zlv = Reshape((self.Ssize,self.nSchannels))(Zlv)
            # Gslv = keras.Model(s,Zlv)

            # variable c
            Zc = Dense(self.Csize*self.nCchannels,name="GzFWC0")(c)
            Zc = BatchNormalization(name="GzBNC0")(Zc)
            Zc = Reshape((self.Csize,self.nCchannels))(Zc)
            for layer in range(1,self.nClayers):
                Zc = Conv1DTranspose(int(self.nCchannels*self.Cstride**(-layer)),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last",name="GzCNNC{:>d}".format(layer))(Zc)
                Zc = BatchNormalization(momentum=0.95,name="GzBNC{:>d}".format(layer))(Zc)
                Zc = LeakyReLU(alpha=0.1,name="GzAC{:>d}".format(layer))(Zc)
                Zc = Dropout(0.2,name="GzDOC{:>d}".format(layer))(Zc)
            Zc = Conv1DTranspose(int(self.nCchannels*self.Cstride**(-self.nClayers)),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last",name="GzCNNC{:>d}".format(self.nClayers))(Zc)
            Zc = BatchNormalization(momentum=0.95,name="GzBNC{:>d}".format(self.nClayers))(Zc)
            Zc = LeakyReLU(alpha=0.1,name="GzAC{:>d}".format(self.nClayers))(Zc)
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
                Zn = BatchNormalization(momentum=0.95,name="GzBNN{:>d}".format(layer))(Zn)
                Zn = LeakyReLU(alpha=0.1,name="GzAN{:>d}".format(layer))(Zn)
                Zn = Dropout(0.2,name="GzDON{:>d}".format(layer))(Zn)
            Zn = Conv1DTranspose(int(self.nNchannels*self.Nstride**(-self.nNlayers)),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last",name="GzCNNN{:>d}".format(self.nNlayers))(Zn)
            Zn = BatchNormalization(momentum=0.95,name="GzBNN{:>d}".format(self.nNlayers))(Zn)
            Zn = LeakyReLU(alpha=0.1,name="GzAN{:>d}".format(self.nNlayers))(Zn)
            Zn = Dropout(0.2,name="GzDON{:>d}".format(self.nNlayers))(Zn)
            GzN = keras.Model(n,Zn)

            Gz = concatenate([GzS.output,GzC.output,GzN.output])
            Gz = Conv1DTranspose(self.nZchannels,
                    self.kernel,1,padding="same",
                    data_format="channels_last",name="GzCNN0")(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95,name="GzBN0")(Gz)
            Gz = LeakyReLU(alpha=0.1,name="GzA0".format(layer+1))(Gz)

        for layer in range(self.nAElayers):
            Gz = Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                self.kernel,self.stride,padding="same",use_bias=False,
                name="GzCNN{:>d}".format(layer+1))(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95,name="GzBN{:>d}".format(layer+1))(Gz)
            Gz = LeakyReLU(alpha=0.1,name="GzA{:>d}".format(layer+1))(Gz) #Activation('relu')(Gz)

        layer = self.nAElayers
        X = Conv1DTranspose(self.nXchannels,self.kernel,1,
            padding="same",use_bias=False,name="GzCNN{:>d}".format(layer+1))(Gz)

        Gz = keras.Model(inputs=[GzS.input,GzC.input,GzN.input],outputs=X,name="Gz")
        return Gz

    def build_Dx(self):
        """
            Conv1D discriminator structure
        """
        layer = 0
        X = Input(shape=self.Xshape,name="X")
        h = Conv1D(self.Xsize*self.stride**(-(layer+1)),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="DxCNN0")(X)
        #h = Conv1D(self.Xsize*self.stride**(-(layer+1)),
        #        self.kernel,self.stride,padding="same",kernel_constraint=self.ClipD,
        #        data_format="channels_last",name="DxCNN0")(X)
        h = LeakyReLU(alpha=0.1,name="DxA0")(h)

        for layer in range(1,self.nDlayers):
            h = Conv1D(self.Xsize*self.stride**(-(layer+1)),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="DxCNN{:>d}".format(layer))(h)
            #h = Conv1D(self.Xsize*self.stride**(-(layer+1)),
            #    self.kernel,self.stride,padding="same",kernel_constraint=self.ClipD,
            #    data_format="channels_last",name="DxCNN{:>d}".format(layer))(h)
            h = BatchNormalization(momentum=0.95,name="DxBN{:>d}".format(layer))(h)
            h = LeakyReLU(alpha=0.2,name="DxA{:>d}".format(layer))(h)
            h = Dropout(0.25,name="DxDO{:>d}".format(layer))(h)
        layer = self.nDlayers    
        h = Flatten(name="DxFL{:>d}".format(layer))(h)
        Px = Dense(1,activation='sigmoid')(h)
        Dx = keras.Model(X,Px,name="Dx")
        return Dx


    def build_Dc(self):
        """
            Dense discriminator structure
        """
        c = Input(shape=(self.latentCdim,))
        h = Dense(3000,kernel_constraint=self.ClipD)(c)
        h = LeakyReLU()(h)
        h = Dense(3000,kernel_constraint=self.ClipD)(h)
        h = LeakyReLU()(h)
        Pc = Dense(1,activation='linear')(h)
        Dc = keras.Model(c,Pc,name="Dc")
        return Dc

    def build_Dn(self):
        """
            Dense discriminator structure
        """
        n = Input(shape=(self.latentNdim,))
        h = Dense(3000,kernel_constraint=self.ClipD)(n)
        h = LeakyReLU()(h)
        h = Dense(3000,kernel_constraint=self.ClipD)(h)
        h = LeakyReLU()(h) 
        Pn = Dense(1,activation='sigmoid')(h)
        Dn = keras.Model(n,Pn,name="Dn")
        return Dn

    def build_Ds(self):
        """
            Dense discriminator structure
        """
        s = Input(shape=(self.latentSdim,))
        h = Dense(3000,kernel_constraint=self.ClipD)(s)
        h = LeakyReLU()(h)
        h = Dense(3000,kernel_constraint=self.ClipD)(h)
        h = LeakyReLU()(h)
        Ps = Dense(1,activation='linear')(h)
        Ds = keras.Model(s,Ps,name="Ds")
        return Ds

# hyperModel = RepGAN()

# tuner = kt.Hyperband(hyperModel, objective="val_accuracy", max_epochs=30, hyperband_iterations=2)

# tuner.search_space_summary()

# tuner.search(Xtrn,epochs=options["epochs"],validation_data=Xvld,callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],)

# bestModel = tuner.get_best_models(1)[0]

# bestHyperparameters = tuner.get_best_hyperparameters(1)[0]


def main(DeviceName):

    options = ParseOptions()

    if not options['cuda']:
        DeviceName = "/cpu:0"

    with tf.device(DeviceName):
        optimizers = {}
        optimizers['DxOpt'] = RMSprop(learning_rate=0.00005) #Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['DcOpt'] = RMSprop(learning_rate=0.00005)
        optimizers['DsOpt'] = RMSprop(learning_rate=0.00005)
        optimizers['DnOpt'] = RMSprop(learning_rate=0.00005)
        optimizers['FxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['GzOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['QsOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['QcOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)

        losses = {}
        losses['AdvDlossWGAN'] = WassersteinDiscriminatorLoss
        losses['AdvGlossWGAN'] = WassersteinGeneratorLoss
        losses['AdvDlossGAN'] = tf.keras.losses.BinaryCrossentropy()
        losses['AdvGlossGAN'] = tf.keras.losses.BinaryCrossentropy()
        losses['RecSloss'] = GaussianNLL
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

    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # # Open a strategy scope.
    # with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.  

        # Instantiate the RepGAN model.
        GiorgiaGAN = RepGAN(options)

        # Compile the RepGAN model.
        GiorgiaGAN.compile(optimizers,losses)

        if options['CreateData']:
            # Create the dataset
            Xtrn,  Xvld, _ = mdof.CreateData(**options)
        else:
            # Load the dataset
            Xtrn, Xvld, _ = mdof.LoadData(**options)
            # (Xtrn,Ctrn), (Xvld,Cvld), _ = mdof.LoadNumpyData(**options)



        # Callbacks
        #plotter = GANMonitor()

        callbacks = [keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}", 
            save_freq='epoch',period=100)]

        history = GiorgiaGAN.fit(Xtrn,epochs=options["epochs"],validation_data=Xvld,
            callbacks=callbacks)

        realX = np.concatenate([x for x, c in Xvld], axis=0)
        realC = np.concatenate([c for x, c in Xvld], axis=0)
        fakeX,fakeC = GiorgiaGAN.predict(Xvld)
        hfg = plt.figure(figsize=(12,6))
        hax = hfg.add_subplot(111)
        hax.plot(realX[0,:,0], color='black')
        hax.plot(fakeX[0,:,0], color='orange')
        hax.set_title('X reconstruction')
        hax.set_ylabel('X')
        hax.set_xlabel('t')
        hax.legend(['X', 'G(F(X))'], loc='lower right')
        plt.tight_layout()
        plt.savefig('reconstruction.png',bbox_inches = 'tight')
        plt.close()

               

        # Print loss
        hfg = plt.figure(figsize=(12,6))
        hax = hfg.add_subplot(111)
        # hax.plot(history.history['AdvDloss'], color='b')
        # hax.plot(history.history['AdvGloss'], color='g')
        hax.plot(history.history['AdvDlossX'], color='r')
        hax.plot(history.history['AdvDlossC'], color='c')
        hax.plot(history.history['AdvDlossS'], color='m')
        hax.plot(history.history['AdvDlossN'], color='gold')
        #plt.plot(history.history['AdvDlossPenGradX'])
        #plt.plot(history.history['AdvGlossX'])
        #plt.plot(history.history['AdvGlossC'])
        #plt.plot(history.history['AdvGlossS'])
        #plt.plot(history.history['AdvGlossN'])
        hax.plot(history.history['RecGlossX'], color='darkorange')
        hax.plot(history.history['RecGlossC'], color='lime')
        hax.plot(history.history['RecGlossS'], color='grey')
        hax.set_title('model loss')
        hax.set_ylabel('loss')
        hax.set_xlabel('epoch')
        hax.legend(['AdvDloss', 'AdvGloss','AdvDlossX','AdvDlossC','AdvDlossS','AdvDlossN',
            'RecGlossX','RecGlossC','RecGlossS'], loc='lower right')
        plt.tight_layout()
        plt.savefig('loss.png',bbox_inches = 'tight')
        plt.close()


if __name__ == '__main__':
    DeviceName = tf.test.gpu_device_name()
    main(DeviceName)

# # We run each op once to warm up; see: https://stackoverflow.com/a/45067900
# cpu()
# gpu()

# # Run the op several times.
# print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
#       '(batch x height x width x channel). Sum of ten runs.')
# print('CPU (s):')
# cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
# print(cpu_time)
# print('GPU (s):')
# gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
# print(gpu_time)
# print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))
