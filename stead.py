import argparse
import math as mt

import tensorflow as tf
#tf.config.run_functions_eagerly(True)

import numpy as np
import pandas as pd
from os.path import join as osj
import h5py

from tensorflow import keras
import timeit
from numpy.random import randn
from numpy.random import randint
from ss_process import rsp
from scipy.signal import detrend, windows



def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",type=str,default="/gpfs/workdir/invsem07/STEAD",help="Data root folder")
    parser.add_argument("--cuda",action='store_true',default=False,help='Use cuda powered GPU')
    parser.add_argument('--dtm',type=float,default=0.01,help='time-step [s]')
    parser.add_argument('--ntm',type=int,default=4096,help='Number of time steps')
    parser.add_argument('--imageSize', type=int, default=4096, help='the height / width of the input image to network')
    parser.add_argument('--nsy',type=int,default=350,help='number of synthetics [1]')
    parser.add_argument('--cutoff', type=float, default=1., help='cutoff frequency')
    options = parser.parse_args().__dict__


    return options

options = ParseOptions()

md = {'dtm':0.01,'cutoff':options["cutoff"],'ntm':options["imageSize"]}
md['vTn'] = np.arange(0.0,3.05,0.05,dtype=np.float64)
md['nTn'] = md['vTn'].size

vtm = md['dtm']*np.arange(0,md['ntm'])
tar = np.zeros((options["nsy"],2))
acc  = -999.9*np.ones(shape=(options["nsy"],3,md['ntm']))

# parse hdf5 database
eqd = h5py.File('/gpfs/workdir/invsem07/STEAD/waveforms_11_13_19.hdf5','r')['earthquake']['local']
eqm = pd.read_csv(osj(options["dataroot"],'metadata_11_13_19.csv'))
eqm = eqm.loc[eqm['trace_category'] == 'earthquake_local']
eqm = eqm.loc[eqm['source_magnitude'] <= 5.0]
eqm = eqm.sample(frac=options["nsy"]/len(eqm)).reset_index(drop=True)

w = windows.tukey(md['ntm'],5/100)
for i in eqm.index:
        tn = eqm.loc[i,'trace_name']
        bi = int(eqd[tn].attrs['p_arrival_sample'])
        for j in range(3):
            acc[i,j,:] = detrend(eqd[tn][bi:bi+options["ntm"],j])*w

signal = tf.convert_to_tensor(acc)
n = tf.cast(signal,dtype=float)

np.savetxt("vtm.csv", vtm, delimiter=",")

signal = np.zeros((options["nsy"],options["ntm"]),dtype=np.float32)
signal[:,:] = n[:,1,:] # {0,1,2} = {x,y,z}
np.savetxt("signal_350.csv", signal, delimiter=",")





