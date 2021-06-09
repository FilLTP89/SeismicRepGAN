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

import os
from os.path import join as opj
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("Num GPUs:", len(physical_devices))
##tf.config.experimental.disable_mlir_graph_optimization()
#print("Num GPUs Available: ", len(tf.test.gpu_device_name()))
#from tensorflow.python.eager.context import get_config
# tf.compat.v1.disable_eager_execution()
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
#tf.debugging.set_log_device_placement(True)

#
#print(c)
#
#gpus = tf.config.list_physical_devices('GPU')
#if gpus:
#  # Restrict TensorFlow to only use the first GPU
#  try:
#    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#  except RuntimeError as e:
#    # Visible devices must be set before GPUs have been initialized
#    print(e)
# @tf_export('config.experimental.disable_mlir_bridge')
# def disable_mlir_bridge():
#   ##Disables experimental MLIR-Based TensorFlow Compiler Bridge.
#   context.context().enable_mlir_bridge = False
import visualkeras
from PIL import ImageFont

font = ImageFont.truetype("arial.ttf", 40) 

import timeit
import sys
import argparse
import numpy as np
import wandb
wandb.init()

import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import csv

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
from numpy.random import randn
from numpy.random import randint

from tensorflow.python.eager import context
import kerastuner as kt
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel

from numpy.linalg import norm
import MDOFload as mdof
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.python.util.tf_export import tf_export
from copy import deepcopy

AdvDLoss_tracker = keras.metrics.Mean(name="loss")
AdvDlossX_tracker = keras.metrics.Mean(name="loss")
AdvDlossC_tracker = keras.metrics.Mean(name="loss")
AdvDlossS_tracker = keras.metrics.Mean(name="loss")
AdvDlossN_tracker = keras.metrics.Mean(name="loss")
#AdvDlossPenGradX_tracker = keras.metrics.Mean(name="loss")
AdvGLoss_tracker = keras.metrics.Mean(name="loss")
#AdvGlossX_tracker = keras.metrics.Mean(name="loss")
#AdvGlossC_tracker = keras.metrics.Mean(name="loss")
#AdvGlossS_tracker = keras.metrics.Mean(name="loss")
#AdvGlossN_tracker = keras.metrics.Mean(name="loss")
RecGlossX_tracker = keras.metrics.Mean(name="loss")
RecGlossC_tracker = keras.metrics.Mean(name="loss")
RecGlossS_tracker = keras.metrics.Mean(name="loss")

gpu_devices = tf.config.experimental.list_physical_devices('GPU')


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
    parser.add_argument("--epochs",type=int,default=100000,help='Number of epochs')
    parser.add_argument("--kernel",type=int,default=3,help='CNN kernel size')
    parser.add_argument("--stride",type=int,default=2,help='CNN stride')
    parser.add_argument("--nAElayers",type=int,default=5,help='Number of AE CNN layers')
    parser.add_argument("--nDlayers",type=int,default=10,help='Number of D CNN layers')
    parser.add_argument("--Xsize",type=int,default=1024,help='Data space size')
    parser.add_argument("--nX",type=int,default=512,help='Number of signals')
    parser.add_argument("--nXchannels",type=int,default=2,help="Number of data channels")
    parser.add_argument("--latentZdim",type=int,default=1024,help="Latent space dimension")
    parser.add_argument("--batchSize",type=int,default=128,help='input batch size')
    parser.add_argument("--nCritic",type=int,default=5,help='number of discriminator training steps')
    parser.add_argument("--clipValue",type=float,default=0.01,help='clip weight for WGAN')
    parser.add_argument("--dataroot",type=str,default="/gpfs/workdir/invsem07/damaged_1_8P",help="Data root folder")
    parser.add_argument("--idChannels",type=int,nargs='+',default=[21,39],help="Channel 1")
    parser.add_argument("--nParams",type=str,default=2,help="Number of parameters")
    parser.add_argument("--case",type=str,default="train_model",help="case")
    parser.add_argument("--avu",type=str,nargs='+',default="U",help="case avu")
    parser.add_argument("--pb",type=str,default="BC",help="case pb")
       
    parser.add_argument("--CreateData",action='store_true',default=False,help='Create data flag')
    parser.add_argument("--cuda",action='store_true',default=False,help='Use cuda powered GPU')
    options = parser.parse_args().__dict__
    

    options['Xshape'] = (options['Xsize'], options['nXchannels'])
    options['batchXshape'] = (options['batchSize'],options['Xsize'],options['nXchannels'])
    options['latentCidx'] = list(range(512))
    options['latentSidx'] = list(range(512,517))
    options['latentNidx'] = list(range(517,options['latentZdim']))
    options['latentCdim'] = len(options['latentCidx'])
    options['latentSdim'] = len(options['latentSidx'])
    options['latentNdim'] = len(options['latentNidx'])
    options['Zsize'] = options['Xsize']//(options['stride']**options['nAElayers'])
    options['nZchannels'] = options['latentZdim']//options['Zsize']
    options['nSchannels'] = options['latentSdim']//options['Zsize']
    options['Sshape'] = (options['Zsize'], options['nSchannels'])

    return options

class RandomWeightedAverage(Layer):
    """Provides a (random) weighted average between real and generated signal samples"""
    def _merge_function(self, inputs, **kwargs):
        alpha = tf.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class SamplingFxS(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

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
    logsigma = pred[:, n_dims:]
    mse = -0.5*K.sum(K.square((true-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)

def MutualInfoLoss(c, c_given_x):
    """The mutual information metric we aim to minimize"""
    eps = 1e-8
    conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
    entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

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

        assert self.nZchannels >= 1
        assert self.nZchannels >= self.stride**self.nAElayers
        assert self.latentZdim >= self.Xsize//(self.stride**self.nAElayers)
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
        # visualkeras.layered_view(self.Fx, to_file='Fx.png', legend=True, font=font)
        # visualkeras.layered_view(self.Qs, to_file='Qs.png', legend=True, font=font)
        # visualkeras.layered_view(self.Qc, to_file='Qc.png', legend=True, font=font)
        # visualkeras.layered_view(self.Gz, to_file='Gz.png', legend=True, font=font)
        # visualkeras.layered_view(self.Dn, to_file='Dn.png', legend=True, font=font)
        # visualkeras.layered_view(self.Ds, to_file='Ds.png', legend=True, font=font)
        # visualkeras.layered_view(self.Dc, to_file='Dc.png', legend=True, font=font)
        # visualkeras.layered_view(self.Dx, to_file='Dx.png', legend=True, font=font)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config    

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
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
                [fakeC,fakeS,fakeN] = self.Fx(realX) # encoded z = Fx(X)

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

                # Adversarial ground truths
                realBCE = tf.ones_like(realXcritic)
                fakeBCE = tf.zeros_like(fakeXcritic)

                # Calculate the discriminator loss using the fake and real logits
                AdvDlossX  = self.AdvDlossGAN(realBCE,realXcritic)*self.PenAdvXloss
                AdvDlossX += self.AdvDlossGAN(fakeBCE,fakeXcritic)*self.PenAdvXloss
                AdvDlossC = self.AdvDlossWGAN(realCcritic,fakeCcritic)*self.PenAdvCloss
                AdvDlossS = self.AdvDlossWGAN(realScritic,fakeScritic)*self.PenAdvSloss
                AdvDlossN  = self.AdvDlossGAN(realBCE,realNcritic)*self.PenAdvNloss
                AdvDlossN += self.AdvDlossGAN(fakeBCE,fakeNcritic)*self.PenAdvNloss
                # AdvDlossN = self.AdvDlossWGAN(realNcritic,fakeNcritic)*self.PenAdvNloss
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
            # Fake
            [fakeC,fakeS,fakeN] = self.Fx(realX) # encoded z = Fx(X)


                        


            # Discriminator determines validity of the real and fake C
            fakeCcritic = self.Dc(fakeC)

            # Discriminator determines validity of the real and fake S
            fakeScritic = self.Ds(fakeS)

            # Discriminator determines validity of the real and fake N
            fakeNcritic = self.Dn(fakeN)

            fakeX = self.Gz((realC,realS,realN)) # fake X = Gz(Fx(X))
            # Discriminator determines validity of the real and fake X
            fakeXcritic = self.Dx(fakeX)
            # Reconstruction
            # fakeZ = Concatenate([fakeC,fakeS,fakeN])
            recX = self.Gz((fakeC,fakeS,fakeN))
            zS = self.Qs(fakeX)
            recC = self.Qc(fakeX)
            # Adversarial ground truths
            realBCE = tf.ones_like(fakeXcritic)
            AdvGlossX = self.AdvGlossGAN(realBCE,fakeXcritic)*self.PenAdvXloss
            AdvGlossC = self.AdvGlossWGAN(fakeCcritic)*self.PenAdvCloss
            AdvGlossS = self.AdvGlossWGAN(fakeScritic)*self.PenAdvSloss
            AdvGlossN = self.AdvGlossWGAN(fakeNcritic)*self.PenAdvNloss
            RecGlossX = self.RecXloss(realX,recX)*self.PenRecXloss
            RecGlossC = self.RecCloss(realC,recC)*self.PenRecCloss
            RecGlossS = self.RecSloss(realS,zS)*self.PenRecSloss
            AdvGloss = AdvGlossX + AdvGlossC + AdvGlossS + AdvGlossN + RecGlossX + RecGlossC + RecGlossS
        

        # Get the gradients w.r.t the generator loss
        
        gradFx, gradGz = tape.gradient(AdvGloss,(self.Fx.trainable_variables, self.Gz.trainable_variables))
        # Update the weights of the generator using the generator optimizer
        self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))
        self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))

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
        [fakeC,fakeS,fakeN] = self.Fx(X)
        fakeX = self.Gz((fakeC,fakeS,fakeN))
        return fakeX, fakeC

    def build_Fx(self):
        """
            Conv1D Fx structure
        """
        # To build this model using the functional API, start by creating an input node:

        X = Input(shape=self.Xshape,name="X")

        h = Conv1D(self.nZchannels*self.stride**(-self.nAElayers),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="FxCNN0")(X)
        h = BatchNormalization(momentum=0.95,name="FxBN0")(h)
        #h = BatchNormalization(momentum=hp.Float('BN_1',min_value=0.0,max_value=1,
        #    default=0.95,step=0.05),name="FxBN0")(h)
        h = LeakyReLU(alpha=0.1,name="FxA0")(h)
        h = Dropout(0.2,name="FxDO0")(h)
        #h = Dropout(rate=hp.Float('dropout_1',min_value=0.0,max_value=0.4,
        #    default=0.2,step=0.05),name="FxDO0")(h)

        for n in range(1,self.nAElayers):
            h = Conv1D(self.nZchannels*self.stride**(-self.nAElayers+n),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="FxCNN{:>d}".format(n))(h)
            h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(n))(h)
            h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(n))(h)
            h = Dropout(0.2,name="FxDO{:>d}".format(n))(h)
        
        h = Flatten(name="FxFL{:>d}".format(n+1))(h)
        h = Dense(self.latentZdim,name="FxFW{:>d}".format(n+1))(h)
        h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(n+1))(h)
        z = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(n+1))(h)

        # variable s
        h = Dense(self.latentSdim,name="FxFWmuS")(z)
        Zmu = BatchNormalization(momentum=0.95)(h)

        h = Dense(self.latentSdim,name="FxFWsiS")(z)
        Zlv = BatchNormalization(momentum=0.95)(h)

        Zs = Concatenate(axis=1)([Zmu,Zlv])
        s = SamplingFxS()([Zmu,Zlv])

        # variable c
        h = Dense(self.latentCdim,name="FxFWC")(z)
        h = BatchNormalization(momentum=0.95,name="FxBNC")(h)
        c = Softmax(name="FxAC")(h)
        QcX = Softmax(name="QcAC")(h)
  
        # variable n
        h = Dense(self.latentNdim,name="FxFWN")(z)
        n = BatchNormalization(momentum=0.95,name="FxBNN")(h)


        Fx = keras.Model(X,[c,s,n],name="Fx")
        Fx.summary()

        dot_img_file = 'Fx.png'
        tf.keras.utils.plot_model(Fx, to_file=dot_img_file, show_shapes=True, show_layer_names=True)

        Qs = keras.Model(X,Zs,name="Qs")
        dot_img_file = 'Qs.png'
        tf.keras.utils.plot_model(Qs, to_file=dot_img_file, show_shapes=True)


        Qc = keras.Model(X,QcX,name="Qc")
        dot_img_file = 'Qc.png'
        tf.keras.utils.plot_model(Qc, to_file=dot_img_file, show_shapes=True)

        return Fx,Qs,Qc

    
    
    def build_Gz(self):
        """
            Conv1D Gz structure
            https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
        """

        c = Input(shape=(self.latentCdim,))
        s = Input(shape=(self.latentSdim,))
        n = Input(shape=(self.latentNdim,))

                     

        GzC = Dense(self.latentCdim)(c)
        GzC = Model(c,GzC)

        GzS = Dense(self.latentSdim)(s)
        GzS = Model(s,GzS)


        GzN = Dense(self.latentNdim)(n)
        GzN = Model(n,GzN)

        z = concatenate([GzC.output,GzS.output,GzN.output])

        Gz = Reshape((self.Zsize,self.nZchannels))(z)
        Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)
        Gz = Activation('relu')(Gz)

        for n in range(self.nAElayers):
            Gz = Conv1DTranspose(self.latentZdim//self.stride**n,
                self.kernel,self.stride,padding="same",use_bias=False)(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)
            Gz = Activation('relu')(Gz)
        
        Gz = Conv1DTranspose(self.nXchannels,self.kernel,1,
            padding="same",use_bias=False,activation="tanh")(Gz)

        model = keras.Model(inputs=[GzC.input,GzS.input,GzN.input],outputs=Gz,name="Gz")
        model.summary()


        dot_img_file = 'Gz.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)

        return model
        
    def build_Dx(self):
        """
            Conv1D discriminator structure
        """
        X = Input(shape=self.Xshape,name="X")
        h = Conv1D(self.nZchannels*self.stride**(-self.nDlayers),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="DxCNN0")(X)
        h = BatchNormalization(momentum=0.95,name="DxBN0")(h)
        #h = BatchNormalization(momentum=hp.Float('BN_1',min_value=0.0,max_value=1,
        #    default=0.95,step=0.05),name="FxBN0")(h)
        h = LeakyReLU(alpha=0.1,name="DxA0")(h)
        h = Dropout(0.2,name="DxDO0")(h)

        """h = Conv1D(32,self.kernel,self.stride,input_shape=self.Xshape,padding="same",
            kernel_initializer=init)(X)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dropout(0.25)(h)
        h = Conv1D(64,self.kernel,self.stride,padding="same",
            kernel_initializer=init)(h)
        h = ZeroPadding1D(padding=((0,1)))(h)
        h = BatchNormalization(momentum=0.95)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dropout(0.25)(h)
        h = Conv1D(128,self.kernel,self.stride,padding="same",
            kernel_initializer=init)(h)
        h = BatchNormalization(momentum=0.95)(h)
        h = LeakyReLU(alpha=0.0)(h)
        h = Dropout(0.25)(h)
        h = Conv1D(256,self.kernel,strides=1,padding="same",
            kernel_initializer=init)(h)
        h = BatchNormalization(momentum=0.95)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dropout(0.25)(h)"""

        for n in range(1,self.nDlayers):
            h = Conv1D(self.nZchannels*self.stride**(-self.nDlayers+n),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="DxCNN{:>d}".format(n))(h)
            h = BatchNormalization(momentum=0.95,name="DxBN{:>d}".format(n))(h)
            h = LeakyReLU(alpha=0.2,name="DxA{:>d}".format(n))(h)
            h = Dropout(0.25,name="DxDO{:>d}".format(n))(h)

        h = Flatten()(h)
        Dx = Dense(1,activation='sigmoid')(h)
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

        # model.summary()

        # X = Input(shape=(self.Xshape))
        # Dx = model(X)

        # Discriminator Dx
        model = keras.Model(X,Dx,name="Dx")
        model.summary()


        dot_img_file = 'Dx.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)

        return model


    def build_Dc(self):
        """
            Dense discriminator structure
        """
        init = RandomNormal(stddev=0.02)

        c = Input(shape=(self.latentCdim,))
        h = Dense(3000,kernel_constraint=self.ClipD)(c)
        #h = Dense(units=hp.Int('units_1',min_value=1000,max_value=5000,
        #    step=50,default=3000),kernel_constraint=self.ClipD)(c)
        h = LeakyReLU()(h)
        h = Dense(3000,kernel_constraint=self.ClipD)(h)
        h = LeakyReLU()(h)
        Dc = Dense(1,activation='linear')(h)

        model = keras.Model(c,Dc,name="Dc")
        model.summary()


        dot_img_file = 'Dc.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)

        return model

    def build_Dn(self):
        """
            Dense discriminator structure
        """
        init = RandomNormal(stddev=0.02)

        n = Input(shape=(self.latentNdim,))
        h = Dense(3000)(n)#,kernel_constraint=self.ClipD)(n)
        h = LeakyReLU()(h)
        h = Dense(3000)(h)#,kernel_constraint=self.ClipD)(h)
        h = LeakyReLU()(h) 
        Dn = Dense(1,activation='sigmoid')(h)

        model = keras.Model(n,Dn,name="Dn")
        model.summary()


        dot_img_file = 'Dn.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)

        return model

    def build_Ds(self):
        """
            Dense discriminator structure
        """
        init = RandomNormal(stddev=0.02)

        s = Input(shape=(self.latentSdim,))
        h = Dense(3000,kernel_constraint=self.ClipD)(s)
        h = LeakyReLU()(h)
        h = Dense(3000,kernel_constraint=self.ClipD)(h)
        h = LeakyReLU()(h)
        Ds = Dense(1,activation='linear')(h)

        model = keras.Model(s,Ds,name="Ds")
        model.summary()


        dot_img_file = 'Ds.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)      

        return keras.Model(s,Ds,name="Ds")

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
        optimizers['DxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999) #RMSprop(learning_rate=0.00005)
        optimizers['DcOpt'] = RMSprop(learning_rate=0.00005)
        optimizers['DsOpt'] = RMSprop(learning_rate=0.00005)
        optimizers['DnOpt'] = RMSprop(learning_rate=0.00005)
        optimizers['FxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
        optimizers['GzOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)

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
            save_freq='epoch',period=500)]

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
