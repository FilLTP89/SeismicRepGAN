# TO EJECTUTE
# bokeh serve --show .\src_rmsp\plot_results.py
#from load_data import ut_data_loader
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import statistics

import MDOFload_source as mdof
from RepGAN_ultimo import RepGAN,ParseOptions,WassersteinDiscriminatorLoss,WassersteinGeneratorLoss,GaussianNLL
from plot_tools import *
from numpy.lib.type_check import imag
import sklearn
from tensorflow import keras
from PIL import Image
import io
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import os


# BOKEH runs
checkpoint_dir = "./ckpt_1/"
options = ParseOptions()

# MODEL LOADING
optimizers = {}
optimizers['DxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['DcOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['DsOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['DnOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['FxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['QOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['GzOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['GqOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['DomainOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)

losses = {}
losses['AdvDlossWGAN'] = WassersteinDiscriminatorLoss
losses['AdvGlossWGAN'] = WassersteinGeneratorLoss
losses['AdvDlossGAN'] = tf.keras.losses.BinaryCrossentropy()
losses['AdvGlossGAN'] = tf.keras.losses.BinaryCrossentropy()
losses['RecSloss'] = GaussianNLL
losses['RecXloss'] = tf.keras.losses.MeanAbsoluteError()
losses['RecCloss'] = tf.keras.losses.CategoricalCrossentropy()

loss_weights = {}
loss_weights['PenAdvXloss'] = 1.
loss_weights['PenAdvCloss'] = 1.
loss_weights['PenAdvSloss'] = 1.
loss_weights['PenAdvNloss'] = 1.
loss_weights['PenRecXloss'] = 1.
loss_weights['PenRecCloss'] = 1.
loss_weights['PenRecSloss'] = 1.
loss_weights['PenDomainloss'] = tf.keras.backend.variable(1.)

# Instantiate the RepGAN model.
RepGAN_fdw = RepGAN(options)

# Compile the RepGAN model.
RepGAN_fdw.compile(optimizers,losses,loss_weights)  # run_eagerly=True

# data structure
'''def train_step(self, realXC):

    realX, realC = realXC
'''

Xtrn, Xvld, _ = mdof.LoadData(**options)

RepGAN_fdw.build(input_shape=(options['batchSize'], options['Xsize'], options['nXchannels']))

latest = tf.train.latest_checkpoint(checkpoint_dir)
#print('restoring model from ' + latest)
RepGAN_fdw.load_weights(latest)
#initial_epoch = int(latest[len(checkpoint_dir) + 7:])
RepGAN_fdw.summary()
RepGAN_fdw.Fx.trainable = False
RepGAN_fdw.Gq.trainable = False
RepGAN_fdw.Q.trainable = False
RepGAN_fdw.Gz.trainable = False
RepGAN_fdw.Dx.trainable = False
RepGAN_fdw.Dc.trainable = False
RepGAN_fdw.Ds.trainable = False
RepGAN_fdw.Dn.trainable = False
RepGAN_fdw.Domain.trainable = False


PlotBokeh(RepGAN_fdw,Xvld,**options)
