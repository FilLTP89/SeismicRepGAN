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

from utils import *
import math as mt

import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float32')
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow_probability as tfp
from tensorflow.keras.constraints import Constraint

import MDOFload as mdof
import matplotlib.pyplot as plt
from plot_tools import *
from copy import deepcopy


from RepGAN_model import RepGAN
import RepGAN_losses


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [options['checkpoint_dir'] + "/" + name for name in os.listdir(options['checkpoint_dir'])]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()

def Main(DeviceName):

    options = ParseOptions()

    if not options['cuda']:
        DeviceName = "/cpu:0"

    with tf.device(DeviceName):
        
        losses = RepGAN_losses.getLosses(**options)

        # Instantiate the RepGAN model.
        GiorgiaGAN = RepGAN(options)

        # Compile the RepGAN model.
        GiorgiaGAN.compile(optimizers,losses) #run_eagerly=True

        GiorgiaGAN.build(input_shape=(options['batchSize'],options['Xsize'],
                                options['nXchannels']))

        GiorgiaGAN.compute_output_shape(input_shape=(options['batchSize'],options['Xsize'],
                                options['nXchannels']))


        if options['CreateData']:
            # Create the dataset
            Xtrn,  Xvld, _ = mdof.CreateData(**options)
        else:
            # Load the dataset
            Xtrn, Xvld, _ = mdof.LoadData(**options)

        #validation_data=Xvld
        history = GiorgiaGAN.fit(Xtrn,epochs=options["epochs"],
            callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=options['checkpoint_dir'] + "/ckpt-{epoch}.ckpt", save_freq='epoch',period=500)]) #CustomLearningRateScheduler(schedule), NewCallback(p,epochs)

        GiorgiaGAN.DumpModels()

        PlotLoss(history,options['results_dir']) # Plot loss
        
if __name__ == '__main__':
    DeviceName = tf.test.gpu_device_name()
    Main(DeviceName)

