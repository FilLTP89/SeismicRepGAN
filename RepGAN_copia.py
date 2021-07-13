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
#import visualkeras

import timeit
import sys
import argparse
import numpy as np
import wandb
#wandb.init()
#wandb.init(settings=wandb.Settings(start_method='fork'))
#wandb.init(settings=wandb.Settings(start_method='thread'))

import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import csv

from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Layer
from tensorflow.keras.layers import Lambda, Concatenate,concatenate, Activation, ZeroPadding1D
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

AdvDmetric_tracker = keras.metrics.Mean(name="metric")
AdvDmetricX_tracker = keras.metrics.Mean(name="metric")
AdvDmetricC_tracker = keras.metrics.Mean(name="metric")
AdvDmetricS_tracker = keras.metrics.Mean(name="metric")
AdvDmetricN_tracker = keras.metrics.Mean(name="metric")
AdvGmetric_tracker = keras.metrics.Mean(name="metric")
RecXmetric_tracker = keras.metrics.Mean(name="metric")
RecCmetric_tracker = keras.metrics.Mean(name="metric")
RecSmetric_tracker = keras.metrics.Mean(name="metric")


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
    parser.add_argument("--epochs",type=int,default=100,help='Number of epochs')
    parser.add_argument("--kernel",type=int,default=3,help='CNN kernel size')
    parser.add_argument("--stride",type=int,default=2,help='CNN stride')
    parser.add_argument("--nCnnLayers",type=int,default=5,help='Number of CNN layers per Coupling Layer')
    parser.add_argument("--Xsize",type=int,default=1024,help='Data space size')
    parser.add_argument("--nX",type=int,default=512,help='Number of signals')
    parser.add_argument("--nXchannels",type=int,default=2,help="Number of data channels")
    parser.add_argument("--latentZdim",type=int,default=1024,help="Latent space dimension")
    parser.add_argument("--batchSize",type=int,default=128,help='input batch size')
    parser.add_argument("--nCritic",type=int,default=5,help='number of discriminator training steps')
    parser.add_argument("--clipValue",type=float,default=0.01,help='clip weight for WGAN')
    parser.add_argument("--dataroot",type=str,default="/gpfs/workdir/invsem07/damaged_1_8P",help="Data root folder")
    parser.add_argument("--idChannels",type=int,nargs='+',default=[21,39],help="Channels")
    parser.add_argument("--dof",type=int,nargs='+',default=[0,1],help="Number of dofs")
    parser.add_argument("--nParams",type=str,default=2,help="Number of parameters")
    parser.add_argument("--case",type=str,default="train_model",help="case")
    parser.add_argument("--avu",type=str,nargs='+',default="U",help="case avu")
    parser.add_argument("--pb",type=str,default="BC",help="case pb")
    parser.add_argument("--CreateData",action='store_true',default=True,help='Create data flag')
    parser.add_argument("--cuda",action='store_true',default=True,help='Use cuda powered GPU')
    parser.add_argument("--signalSampling",type=float,default=0.02,help='signal sampling')
    parser.add_argument("--seqSampling",type=int,default=1,help='sequence sampling')
    parser.add_argument("--seqStart",type=int,default=0,help='sequence start')
    parser.add_argument("--seqLen",type=int,default=250,help='sequence length')

        
    options = parser.parse_args().__dict__
    

    options['Xshape'] = (options['Xsize'], options['nXchannels'])
    options['batchXshape'] = (options['batchSize'],options['Xsize'],options['nXchannels'])
    options['Zsize']  = options['Xsize']//(options['stride']**options['nCnnLayers'])
    options['latentCidx'] = list(range(5))
    options['latentSidx'] = list(range(5,7))
    options['latentNidx'] = list(range(7,options['latentZdim']))
    options['latentCdim'] = len(options['latentCidx'])
    options['latentSdim'] = len(options['latentSidx'])
    options['latentNdim'] = len(options['latentNidx'])
    options['nZchannels'] = options['latentZdim']//options['Zsize']
    options['nCchannels'] = options['latentCdim']//options['Zsize']
    options['nSchannels'] = options['latentSdim']//options['Zsize']
    options['nNchannels'] = options['latentNdim']//options['Zsize']
    options['seqLenEnd'] = options['seqStart']+options['seqLen']*options['seqSampling']
    options['start'] = options['seqStart']*options['signalSampling']
    options['end'] = options['seqLenEnd']*options['signalSampling']
    options['tAxis'] = np.arange(options['start'],options['end'],options['signalSampling']*options['seqSampling'])


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


# class GANMonitor(keras.callbacks.Callback):
#     """A callback to generate and save images after each epoch"""

#     def __init__(self):
#         self.nSignal = self.batchSize

#     def on_epoch_end(self, epoch, logs=None):
#         _, ax = plt.subplots(self.batchSize, 2, figsize=(12, 12))
#         for i, signal in enumerate(Xtrn[0].take(self.nSignal)):
#             rec = self.GiorgiaGAN.Dx(signal)[0].numpy()
#             rec = (prediction * 127.5 + 127.5).astype(np.uint8)
#             signal = (signal[0] * 127.5 + 127.5).numpy().astype(np.uint8)

#             ax[i, 0].imshow(signal)
#             ax[i, 1].imshow(rec)
#             ax[i, 0].set_title("Input signal")
#             ax[i, 1].set_title("Reconstructed signal")
#             ax[i, 0].axis("off")
#             ax[i, 1].axis("off")

#             rec = keras.preprocessing.image.array_to_img(rec)
#             prec.save(
#                 "generated_signal_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
#             )
#         plt.show()
#         plt.close()


def WassersteinDiscriminatorLoss(y_true, y_fake):
    return -tf.reduce_mean(y_true-y_fake)
    


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


def GaussianNLL(true, Zmu, Zlv):
    """
     Gaussian negative loglikelihood loss function 
    """
    #n_dims = int(int(pred.shape[1])/2)

    n_dims = int(Zmu.shape[1])

    #mu = pred[:, 0:n_dims]
    #logsigma = pred[:, n_dims:]

    mu = Zmu
    logsigma = Zlv
    
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

def plotSignal(n,dof,realXC,mean,std,**kwargs):

    matplotlib.style.use('classic')
    matplotlib.rc('font',  size=8, family='serif')
    matplotlib.rc('axes',  titlesize=8)
    matplotlib.rc('text',  usetex=True)
    matplotlib.rc('lines', linewidth=0.5)
    
    fig, axs = plt.subplots(n, n)
    fig.tight_layout()
    
    realX_tensor, realC = realXC
    realX_proto_tensor = tf.make_tensor_proto(realX_tensor)  # convert `tensor a` to a proto tensor
    realX = tf.make_ndarray(realX_proto_tensor)

    for i in range(n):
        for j in range(n):
            FxInput  = np.expand_dims(realX[i+j*n], axis = 0) #controllare se è necessaria trasformazione
            [fakeC,fakeS,fakeN] = Fx.predict(FxInput)
            GzOutput = Gz.predict((fakeC,fakeS,fakeN))
            realXPlot = (realX[i+j*n,:,dof]*std) + mean
            recXPlot = (GzOutput[0,:,dof]*std) + mean

            axs[i,j].plot(tAxis[:],realXplot,'k', tAxis[:],recXPlot,'orange')
            for axis in ['top','bottom','left','right']:
                axs[i,j].spines[axis].set_linewidth(0.5)
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            # Only show ticks on the left and bottom spines
            axs[i,j].yaxis.set_ticks_position('left')
            axs[i,j].xaxis.set_ticks_position('bottom')
            #Pirellone            
            axs[i,j].xaxis.set_major_locator(ticker.LinearLocator(5))
            axs[i,j].yaxis.set_major_locator(ticker.LinearLocator(5))
            axs[i,j].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

            
            if n == 1:
                plt.xlabel(r'time [s]', fontsize=24)
                plt.ylabel(r'normalized displ. [-]', fontsize=24)
                plt.legend([r'instance for inference',r'generated instance'],loc='upper right',fontsize=24)
                plt.savefig('real_vs_rec_signals_{:>s}_dof_{:>d}.png'.format(case,dof), bbox_inches='tight')

            else:
                plt.xlabel(r'$t$ [s]', labelpad=-1, fontsize=8)
                plt.ylabel(r'displ. [m]', labelpad=-2, fontsize=8)
                       
    axs[i,j].legend([r'$\mathbf{v}_{:>d}$'.format(dof),r'$\mathbf{u}_{:>d} $'.format(dof)],loc='upper right')

    plt.savefig('real_vs_rec_signals_multi_{:>s}_dof_{:>d}.png'.format(case,dof), bbox_inches='tight')
    


class RepGAN(Model):

    def __init__(self,options):
        super(RepGAN, self).__init__()
        """
            Setup
        """
        self.__dict__.update(options)

        assert self.nZchannels >= 1
        assert self.nZchannels >= self.stride**self.nCnnLayers
        assert self.latentZdim >= self.Xsize//(self.stride**self.nCnnLayers)
        
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
            RecGlossX_tracker,RecGlossC_tracker,RecGlossS_tracker, AdvDmetric_tracker,AdvGmetric_tracker,AdvDmetricX_tracker,
            AdvDmetricC_tracker, AdvDmetricS_tracker, AdvDmetricN_tracker, RecXmetric_tracker, RecCmetric_tracker, RecSmetric_tracker]

    def compile(self,optimizers,losses,metrics):
        super(RepGAN, self).compile()
        """
            Optimizers
        """
        self.__dict__.update(optimizers)
        """
            Losses
        """
        self.__dict__.update(losses)
        """
            Metrics
        """
        self.__dict__.update(metrics)

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


        # Get the batch size
        #batchSize = tf.shape(realX)[0]
        #if self.batchSize != batchSize:
        self.batchSize = tf.shape(realX)[0]

        #dim = tf.compat.v1.placeholder(tf.int32,shape=[None, 1])
        #realBCE = tf.ones(shape=tf.stack([tf.shape(dim)[0], 1]))
        #fakeBCE = tf.zeros(shape=tf.stack([tf.shape(dim)[0], 1]))

        #with tf.compat.v1.Session() as sess:
        #    print(sess.run(realBCE,fakeBCE, feed_dict={dim: 1}))
                

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
                AdvDlossX = self.AdvDlossGAN(realBCE,realXcritic)*self.PenAdvXloss
                AdvDlossX += self.AdvDlossGAN(fakeBCE,fakeXcritic)*self.PenAdvXloss
                AdvDlossC = self.AdvDlossWGAN(realCcritic,fakeCcritic)*self.PenAdvCloss
                AdvDlossS = self.AdvDlossWGAN(realScritic,fakeScritic)*self.PenAdvSloss
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
            # Fake
            [fakeC,fakeS,fakeN] = self.Fx(realX) # encoded z = Fx(X)

            fakeX = self.Gz((realC,realS,realN)) # fake X = Gz(Fx(X))

                        
            # Discriminator determines validity of the real and fake X
            fakeXcritic = self.Dx(fakeX)


            # Discriminator determines validity of the real and fake C
            fakeCcritic = self.Dc(fakeC)

            # Discriminator determines validity of the real and fake S
            fakeScritic = self.Ds(fakeS)

            # Discriminator determines validity of the real and fake N
            fakeNcritic = self.Dn(fakeN)

            # Reconstruction
            # fakeZ = Concatenate([fakeC,fakeS,fakeN])
            recX = self.Gz((fakeC,fakeS,fakeN))
            Zmu,Zlv = self.Qs(fakeX)
            ZS = tf.keras.layers.concatenate([Zmu,Zlv],axis=-1)

            recC = self.Qc(fakeX)
            
            AdvGlossX = self.AdvGlossGAN(realBCE,realXcritic)*self.PenAdvXloss
            AdvGlossX += self.AdvGlossGAN(fakeBCE,fakeXcritic)*self.PenAdvXloss
            AdvGlossC = self.AdvGlossWGAN(fakeCcritic)*self.PenAdvCloss
            AdvGlossS = self.AdvGlossWGAN(fakeScritic)*self.PenAdvSloss
            AdvGlossN = self.AdvGlossWGAN(fakeNcritic)*self.PenAdvNloss
            RecGlossX = self.RecXloss(realX,recX)*self.PenRecXloss
            RecGlossC = self.RecCloss(realC,recC)*self.PenRecCloss
            RecGlossS = self.RecSloss(realS,Zmu,Zlv)*self.PenRecSloss
            AdvGloss = AdvGlossX + AdvGlossC + AdvGlossS + AdvGlossN + RecGlossX + RecGlossC + RecGlossS
        

        # Get the gradients w.r.t the generator loss
        gradFx, gradGz = tape.gradient(AdvGloss,
            (self.Fx.trainable_variables, self.Gz.trainable_variables))
        
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
      

    # def test_step(self, realXC):

    #     # Unpack the data
    #     realX, realC = realXC

    #     realS = tf.random.normal(mean=0.0,stddev=0.5,shape=[self.batchSize,self.latentSdim])
    #     realN = tf.random.normal(mean=0.0,stddev=0.3,shape=[self.batchSize,self.latentNdim])

    #     # Compute predictions

    #     # Generate fake latent code from real signals
    #     [fakeC,fakeS,fakeN] = self.Fx(realX, training=False) # encoded z = Fx(X)

    #     fakeX = self.Gz((realC,realS,realN), training=False) # fake X = Gz(Fx(X))

    #     recX = self.Gz((fakeC,fakeS,fakeN), training=False)
    #     Zmu,Zlv = self.Qs(fakeX, training=False)
    #     ZS = tf.keras.layers.concatenate([Zmu,Zlv],axis=-1)
    #     recS = SamplingFxS()([Zmu,Zlv])

    #     recC = self.Qc(fakeX, training=False)
                
    #     # Discriminator determines validity of the real and fake X
    #     fakeXcritic = self.Dx(fakeX, training=False)
    #     realXcritic = self.Dx(realX, training=False)

    #     # Discriminator determines validity of the real and fake C
    #     fakeCcritic = self.Dc(fakeC, training=False)
    #     realCcritic = self.Dc(realC, training=False)

    #     # Discriminator determines validity of the real and fake N
    #     fakeNcritic = self.Dn(fakeN, training=False)
    #     realNcritic = self.Dn(realN, training=False) 

    #     # Discriminator determines validity of the real and fake S
    #     fakeScritic = self.Ds(fakeS, training=False)
    #     realScritic = self.Ds(realS, training=False)

    #     # Adversarial ground truths
    #     realBCE = tf.ones_like(realXcritic)
    #     fakeBCE = tf.zeros_like(fakeXcritic)

    #     # Updates the metrics tracking the loss
    #     #self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #     AdvDlossX = self.AdvDlossGAN(realBCE,realXcritic)*self.PenAdvXloss
    #     AdvDlossX += self.AdvDlossGAN(fakeBCE,fakeXcritic)*self.PenAdvXloss
    #     AdvDlossC = self.AdvDlossWGAN(realCcritic,fakeCcritic)*self.PenAdvCloss
    #     AdvDlossS = self.AdvDlossWGAN(realScritic,fakeScritic)*self.PenAdvSloss
    #     AdvDlossN = self.AdvDlossWGAN(realNcritic,fakeNcritic)*self.PenAdvNloss
    #     AdvDloss = AdvDlossX + AdvDlossC + AdvDlossS + AdvDlossN

    #     AdvGlossX = self.AdvGlossGAN(realBCE,realXcritic)*self.PenAdvXloss
    #     AdvGlossX += self.AdvGlossGAN(fakeBCE,fakeXcritic)*self.PenAdvXloss
    #     AdvGlossC = self.AdvGlossWGAN(fakeCcritic)*self.PenAdvCloss
    #     AdvGlossS = self.AdvGlossWGAN(fakeScritic)*self.PenAdvSloss
    #     AdvGlossN = self.AdvGlossWGAN(fakeNcritic)*self.PenAdvNloss
    #     RecGlossX = self.RecXloss(realX,recX)*self.PenRecXloss
    #     RecGlossC = self.RecCloss(realC,recC)*self.PenRecCloss
    #     RecGlossS = self.RecSloss(realS,Zmu,Zlv)*self.PenRecSloss
    #     AdvGloss = AdvGlossX + AdvGlossC + AdvGlossS + AdvGlossN + RecGlossX + RecGlossC + RecGlossS

    #     # Update the metrics.
    #     #self.compiled_metrics.update_state(y, y_pred)
    #     AdvDmetricXreal = self.AdvDmetricGAN.update_state(realBCE,realXcritic)
    #     AdvDmetricXfake = self.AdvDmetricGAN.update_state(fakeBCE,fakeXcritic)
    #     AdvDmetricX = AdvDmetricXreal + AdvDmetricXfake
    #     AdvDmetricC = self.AdvDmetricWGAN.update_state(realCcritic,fakeCcritic)
    #     AdvDmetricS = self.AdvDmetricWGAN.update_state(realScritic,fakeScritic)
    #     AdvDmetricN = self.AdvDmetricWGAN.update_state(realNcritic,fakeNcritic)
    #     AdvDmetric = AdvDmetricX + AdvDmetricC + AdvDmetricS + AdvDmetricN

    #     AdvGmetricXreal = self.AdvGmetricGAN.update_state(realBCE,realXcritic)
    #     AdvGmetricXfake = self.AdvGmetricGAN.update_state(fakeBCE,fakeXcritic)
    #     AdvGmetricX = AdvGmetricXreal + AdvGmetricXfake
    #     AdvGmetricC = self.AdvGmetricWGAN.update_state(realCcritic,fakeCcritic)
    #     AdvGmetricS = self.AdvGmetricWGAN.update_state(realScritic,fakeScritic)
    #     AdvGmetricN = self.AdvGmetricWGAN.update_state(realNcritic,fakeNcritic)
    #     RecXmetric = self.RecXmetric.update_state(realX,recX)
    #     RecCmetric = self.RecCmetric.update_state(realC,recC)
    #     RecSmetric = self.RecSmetric.update_state(realS,recS)
    #     AdvGmetric = AdvGmetricX + AdvGmetricC + AdvGmetricS + AdvGmetricN + RecXmetric + RecCmetric + RecSmetric

    #     AdvDmetric_tracker.update_state(AdvDmetric)
    #     AdvGmetric_tracker.update_state(AdvGmetric)
    #     AdvDmetricX_tracker.update_state(AdvDmetricX)
    #     AdvDmetricC_tracker.update_state(AdvDmetricC)
    #     AdvDmetricS_tracker.update_state(AdvDmetricS)
    #     AdvDmetricN_tracker.update_state(AdvDmetricN)
    #     RecXmetric_tracker.update_state(RecXmetric)
    #     RecCmetric_tracker.update_state(RecCmetric)
    #     RecSmetric_tracker.update_state(RecSmetric)

    #     # Return a dict mapping metric names to current value.
    #     # Note that it will include the loss (tracked in self.metrics).
    #     return {"AdvDloss": AdvDLoss_tracker.result(),"AdvGloss": AdvGLoss_tracker.result(), "AdvDlossX": AdvDlossX_tracker.result(),
    #         "AdvDlossC": AdvDlossC_tracker.result(),"AdvDlossS": AdvDlossS_tracker.result(),"AdvDlossN": AdvDlossN_tracker.result(),
    #         "RecGlossX": RecGlossX_tracker.result(), "RecGlossC": RecGlossC_tracker.result(), "RecGlossS": RecGlossS_tracker.result(),
    #         "AdvDmetric": AdvDmetric_tracker.result(),"AdvGmetric": AdvGmetric_tracker.result(), "AdvDmetricX": AdvDmetricX_tracker.result(),
    #         "AdvDmetricC": AdvDmetricC_tracker.result(),"AdvDmetricS": AdvDmetricS_tracker.result(),"AdvDmetricN": AdvDmetricN_tracker.result(),
    #         "RecXmetric": RecXmetric_tracker.result(), "RecCmetric": RecCmetric_tracker.result(), "RecSmetric": RecSmetric_tracker.result()}


#     def predict(self, realXC):
#         """
#         Performs custom prediction.
# .
#         """
#         realX, realC = realXC

#         # Generate fake latent code from real signals
#         [fakeC,fakeS,fakeN] = self.Fx(realX) # encoded z = Fx(X)

#         recX = self.Gz((fakeC,fakeS,fakeN))
                
        
#         inputs = np.asarray(instances)
#         preprocessed_inputs = self._preprocessor.preprocess(inputs)
#         outputs = self._model.predict(preprocessed_inputs)

#         return recX



    def call(self, X):
        [fakeC,_,_] = self.Fx(X)
        return fakeC

    def build_Fx(self):
        """
            Conv1D Fx structure
        """
        # To build this model using the functional API, start by creating an input node:
        init = RandomNormal(stddev=0.02)

        X = Input(shape=self.Xshape,name="X")

        h = Conv1D(self.nZchannels*self.stride**(-self.nCnnLayers),
                self.kernel,self.stride,padding="same",kernel_initializer=init,
                data_format="channels_last",name="FxCNN0")(X)
        h = BatchNormalization(momentum=0.95,name="FxBN0")(h)
        #h = BatchNormalization(momentum=hp.Float('BN_1',min_value=0.0,max_value=1,
        #    default=0.95,step=0.05),name="FxBN0")(h)
        h = LeakyReLU(alpha=0.1,name="FxA0")(h)
        h = Dropout(0.2,name="FxDO0")(h)
        #h = Dropout(rate=hp.Float('dropout_1',min_value=0.0,max_value=0.4,
        #    default=0.2,step=0.05),name="FxDO0")(h)

        for n in range(1,self.nCnnLayers):
            h = Conv1D(self.nZchannels*self.stride**(-self.nCnnLayers+n),
                self.kernel,self.stride,padding="same",kernel_initializer=init,
                data_format="channels_last",name="FxCNN{:>d}".format(n))(h)
            h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(n))(h)
            h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(n))(h)
            h = Dropout(0.2,name="FxDO{:>d}".format(n))(h)
        
        h = Flatten(name="FxFL{:>d}".format(n+1))(h)
        h = Dense(self.latentZdim,kernel_initializer=init,name="FxFW{:>d}".format(n+1))(h)
        h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(n+1))(h)
        z = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(n+1))(h)

        # variable s
        h = Dense(self.latentSdim,kernel_initializer=init,name="FxFWmuS")(z)
        Zmu = BatchNormalization(momentum=0.95)(h)

        h = Dense(self.latentSdim,kernel_initializer=init,name="FxFWsiS")(z)
        Zlv = BatchNormalization(momentum=0.95)(h)

        s = SamplingFxS()([Zmu,Zlv])

        # variable c
        h = Dense(self.latentCdim,kernel_initializer=init,name="FxFWC")(z)
        h = BatchNormalization(momentum=0.95,name="FxBNC")(h)
        c = Softmax(name="FxAC")(h)
        QcX = Softmax(name="QcAC")(h)
  
        # variable n
        h = Dense(self.latentNdim,kernel_initializer=init,name="FxFWN")(z)
        n = BatchNormalization(momentum=0.95,name="FxBNN")(h)


        Fx = keras.Model(X,[c,s,n],name="Fx")
        Fx.summary()

        dot_img_file = 'Fx.png'
        tf.keras.utils.plot_model(Fx, to_file=dot_img_file, show_shapes=True, show_layer_names=True)

        Qs = keras.Model(X,[Zmu,Zlv],name="Qs")
        dot_img_file = 'Qs.png'
        tf.keras.utils.plot_model(Qs, to_file=dot_img_file, show_shapes=True)


        Qc = keras.Model(X,QcX,name="Qc")
        dot_img_file = 'Qc.png'
        tf.keras.utils.plot_model(Qc, to_file=dot_img_file, show_shapes=True)

        return Fx,Qs,Qc

    
    
    def build_Gz(self):
        """
            Conv1D Gz structure
        """
        init = RandomNormal(stddev=0.02)
        
        c = Input(shape=(self.latentCdim,))
        s = Input(shape=(self.latentSdim,))
        n = Input(shape=(self.latentNdim,))

                     
        #GzC = Dense(self.Zsize*self.nCchannels,use_bias=False)(c)
        #GzC = Reshape((self.Zsize,self.nCchannels))(GzC)
        #GzC = Model(c,GzC)

        GzC = Dense(self.latentCdim,kernel_initializer=init,use_bias=False)(c)
        GzC = Model(c,GzC)

        #GzS = Dense(self.Zsize*self.nSchannels,use_bias=False)(s)
        #GzS = Reshape((self.Zsize,self.nSchannels))(GzS)
        #GzS = Model(s,GzS)

        GzS = Dense(self.latentSdim,kernel_initializer=init,use_bias=False)(s)
        GzS = Model(s,GzS)

        #GzN = Dense(self.Zsize*self.nNchannels,use_bias=False)(n)
        #GzN = Reshape((self.Zsize,self.nNchannels))(GzN)
        #GzN = Model(n,GzN)

        GzN = Dense(self.latentNdim,kernel_initializer=init,use_bias=False)(n)
        GzN = Model(n,GzN)

        z = concatenate([GzC.output,GzS.output,GzN.output])


        Gz = Reshape((self.Zsize,self.nZchannels))(z)
        Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)
        Gz = Activation('relu')(Gz)

        for n in range(self.nCnnLayers):
            Gz = Conv1DTranspose(self.latentZdim//self.stride**n,
                self.kernel,self.stride,kernel_initializer=init,padding="same")(Gz)
            Gz = BatchNormalization(axis=-1,momentum=0.95)(Gz)
            Gz = Activation('relu')(Gz)
        
        Gz = Conv1DTranspose(self.nXchannels,self.kernel,1,kernel_initializer=init,padding="same")(Gz)

        model = keras.Model([GzC.input,GzS.input,GzN.input],Gz,name="Gz")
        model.summary()


        dot_img_file = 'Gz.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)

        return model
        
    def build_Dx(self):
        """
            Conv1D discriminator structure
        """
        init = RandomNormal(stddev=0.02)

        X = Input(shape=self.Xshape,name="X")        
        h = Conv1D(32,self.kernel,self.stride,input_shape=self.Xshape,padding="same",
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
        h = Dropout(0.25)(h)
        h = Flatten()(h)
        Dx = Dense(1,activation='sigmoid',
            kernel_initializer=init)(h)
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
        h = Dense(3000,kernel_initializer=init,kernel_constraint=self.ClipD)(c)
        #h = Dense(units=hp.Int('units_1',min_value=1000,max_value=5000,
        #    step=50,default=3000),kernel_initializer=init,kernel_constraint=self.ClipD)(c)
        h = LeakyReLU()(h)
        h = Dense(3000,kernel_initializer=init,kernel_constraint=self.ClipD)(h)
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
        h = Dense(3000,kernel_initializer=init,kernel_constraint=self.ClipD)(n)
        h = LeakyReLU()(h)
        h = Dense(3000,kernel_initializer=init,kernel_constraint=self.ClipD)(h)
        h = LeakyReLU()(h) 
        Dn = Dense(1,activation='linear')(h)  

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
        h = Dense(3000,kernel_initializer=init,kernel_constraint=self.ClipD)(s)
        h = LeakyReLU()(h)
        h = Dense(3000,kernel_initializer=init,kernel_constraint=self.ClipD)(h)
        h = LeakyReLU()(h)
        Ds = Dense(1,activation='linear')(h)

        model = keras.Model(s,Ds,name="Ds")
        model.summary()


        dot_img_file = 'Ds.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)      

        return keras.Model(s,Ds,name="Ds")

#hyperModel = RepGAN(nCnnLayers,latentZdim,nCritic,clipValue)

#tuner = kt.Hyperband(hyperModel, objective="val_accuracy",max_epochs=30,
#    seed=1, executions_per_trial=2, directory='hyperband', project_name='RepGAN')

#tuner.search_space_summary()

#tuner.search(Xtrn,epochs=options["epochs"],validation_data=Xvld,callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],)

#tuner.results_summary()

#bestModel = tuner.get_best_models(1)[0]

#bestHyperparameters = tuner.get_best_hyperparameters(1)[0]


def main(DeviceName):

    options = ParseOptions()

    if not options['cuda']:
        DeviceName = "/cpu:0"

    with tf.device(DeviceName):
        

        optimizers = {}
        optimizers['DxOpt'] = RMSprop(learning_rate=0.00005)
        #optimizers['DxOpt'] = RMSprop(hp.Float('learning_rate',min_value=5e-6,
        #    max_value=5e-4,sampling='LOG',default=5e-5))
        optimizers['DcOpt'] = RMSprop(learning_rate=0.00005)
        optimizers['DsOpt'] = RMSprop(learning_rate=0.00005)
        optimizers['DnOpt'] = RMSprop(learning_rate=0.00005)
        optimizers['FxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        optimizers['GzOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

        losses = {}
        losses['AdvDlossWGAN'] = WassersteinDiscriminatorLoss
        losses['AdvGlossWGAN'] = WassersteinGeneratorLoss
        losses['AdvDlossGAN'] = tf.keras.losses.BinaryCrossentropy()
        losses['AdvGlossGAN'] = tf.keras.losses.BinaryCrossentropy()
        losses['RecSloss'] = GaussianNLL
        losses['RecXloss'] = tf.keras.losses.MeanSquaredError()
        losses['RecCloss'] = MutualInfoLoss
        losses['PenAdvXloss'] = 1.
        losses['PenAdvCloss'] = 1.
        losses['PenAdvSloss'] = 1.
        losses['PenAdvNloss'] = 1.
        losses['PenRecXloss'] = 1.
        losses['PenRecCloss'] = 1.
        losses['PenRecSloss'] = 1.
        losses['PenGradX'] = 10.

        metrics = {}
        metrics['AdvDmetricWGAN'] = tf.keras.metrics.Accuracy()
        metrics['AdvGmetricWGAN'] = tf.keras.metrics.Accuracy()
        metrics['AdvDmetricGAN'] = tf.keras.metrics.BinaryCrossentropy()
        metrics['AdvGmetricGAN'] = tf.keras.metrics.BinaryCrossentropy()
        metrics['RecSmetric'] = tf.keras.metrics.BinaryAccuracy()
        metrics['RecXmetric'] = tf.keras.metrics.Accuracy()
        metrics['RecCmetric'] = tf.keras.metrics.SparseCategoricalCrossentropy()
       
    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # # Open a strategy scope.
    # with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.  
          
        # Instantiate the RepGAN model.
        GiorgiaGAN = RepGAN(options)


        # Compile the RepGAN model.
        GiorgiaGAN.compile(optimizers,losses,metrics)

        if options['CreateData']:
            # Create the dataset
            Xtrn,  Xvld, _, mean, std = mdof.CreateData(**options)
        else:
            # Load the dataset
            Xtrn,  Xvld, _, mean, std = mdof.LoadData(**options)
        


        # Callbacks
        #plotter = GANMonitor()
        
        callbacks = [keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}", 
            save_freq='epoch',period=1)]

        history = GiorgiaGAN.fit(Xtrn,epochs=options["epochs"],validation_data=Xvld,
            callbacks=callbacks)

        # Print results
        plt.plot(history.history['AdvDloss'], color='b')
        plt.plot(history.history['AdvGloss'], color='g')
        plt.plot(history.history['AdvDlossX'], color='r')
        plt.plot(history.history['AdvDlossC'], color='c')
        plt.plot(history.history['AdvDlossS'], color='m')
        plt.plot(history.history['AdvDlossN'], color='gold')
        #plt.plot(history.history['AdvDlossPenGradX'])
        #plt.plot(history.history['AdvGlossX'])
        #plt.plot(history.history['AdvGlossC'])
        #plt.plot(history.history['AdvGlossS'])
        #plt.plot(history.history['AdvGlossN'])
        plt.plot(history.history['RecGlossX'], color='darkorange')
        plt.plot(history.history['RecGlossC'], color='lime')
        plt.plot(history.history['RecGlossS'], color='grey')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['AdvDloss', 'AdvGloss','AdvDlossX','AdvDlossC','AdvDlossS','AdvDlossN',
            'RecGlossX','RecGlossC','RecGlossS'], loc='upper left')
        plt.savefig('loss.png',bbox_inches = 'tight')
        plt.show()

        n = 4

        for i in options['dof']:
            plotSignal(n,i,Xtrn,mean,std,**options)

        # Create test dataset
        Xtest,  Xvld, _ , mean, std = mdof.CreateData(**options)

        # Evaluate the model
        score = GiorgiaGAN.evaluate(Xtest)

        #realX_tensor = Xtest[0]
        #realX_proto_tensor = tf.make_tensor_proto(realX_tensor)  # convert `tensor a` to a proto tensor
        #realX = tf.make_ndarray(realX_proto_tensor)

        # Make predictions
        #for i in range(self.batchSize):
        #    for j in range(self.batchSize):
        #        FxInput  = np.expand_dims(realX[i+j*self.batchSize], axis = 0)
        #        [fakeC,fakeS,fakeN] = Fx.predict(FxInput)
        #        GzOutput = Gz.predict((fakeC,fakeS,fakeN))
        FxInput  = Xtest[0]
        [fakeC,fakeS,fakeN] = Fx.predict(FxInput)
        GzOutput = Gz.predict((fakeC,fakeS,fakeN))
                 
        

        # # Load the checkpoints
        # weight_file = checkpoint_dir + "/ckpt-{epoch}"
        # GiorgiaGAN.load_weights(weight_file).expect_partial()
        # print("Weights loaded successfully")

        # _, ax = plt.subplots(4, 2, figsize=(10, 15))
        # for i, signal in enumerate(Xtrn[0]):
        #     rec = GiorgiaGAN.Dx(signal, training=False)[0].numpy()
        #     rec = (rec * 127.5 + 127.5).astype(np.uint8)
        #     real = (real[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        #     ax[i, 0].imshow(signal)
        #     ax[i, 1].imshow(rec)
        #     ax[i, 0].set_title("Input signal")
        #     ax[i, 1].set_title("Reconstructed signal")
        #     ax[i, 0].axis("off")
        #     ax[i, 1].axis("off")
        #     rec = keras.preprocessing.image.array_to_img(rec)
        #     rec.save("generated_signal_{i}.png".format(i=i))
        # plt.tight_layout()
        # plt.show()

        # Save model
        #GiorgiaGAN.save('/gpfs/workdir/invsem07/GiorgiaGAN/model.h5')
        #return history['AdvDloss'], history['AdvGloss'], history['AdvDlossX'], history['AdvDlossC'], history['AdvDlossS'],
        #    history['AdvDlossN'], history['RecGlossX'], history['RecGlossC'], history['RecGlossS']

        # Generate generalization metrics
        #score = GiorgiaGAN.evaluate(Xtrn, verbose=0)
        #print('Test loss: %.5f, Test accuracy: %.5f' % (score[0], score[1]))

       



               



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