# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
__author__ = "Filippo Gatti Giorgia Colombera"
__copyright__ = "Copyright 2021, CentraleSupélec (LMPS UMR CNRS 9026)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__Maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

import glob
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from os.path import join as opj
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.utils import shuffle
import dask as ds
import dask.delayed as dd
import dask.dataframe as df
import dask.array as da
import h5py
from matplotlib import pyplot as plt
import seaborn as sn
import sys
import csv
from random import randint

def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


def read_th_engine(filename):
    # reads each csv file to a pandas.DataFrame
    df_csv = pd.read_csv(filename, header=None,
                         names=["ACC{:>s}".format(filename.split("Acc_")[1].strip(".csv"))],
                         dtype={"ACC{:>s}".format(filename.split("Acc_")[1].strip(".csv")): np.float64},)
    return df_csv

def CreateData(**kwargs):
    CreateData.__globals__.update(kwargs)

    data = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    noise = np.zeros((nX, nXchannels, Xsize), dtype=np.float32)
    damage_index = np.zeros((nX,),dtype=np.float32)
    
    for i in range(signal):
        
        fid_list_th = sorted(glob.glob(opj(rawdata_dir[i], "Acc_*.csv")))
        th_channels = [int(fname.split("Acc_")[1].strip(".csv")) for fname in fid_list_th]
        
        # load time histories
        df_th = pd.concat([read_th_engine(fname) for fname in fid_list_th],axis=1)
        
        # load the damage index
        df_di = pd.read_csv(opj(rawdata_dir[i], "damage_index.csv"), header=None,
                            names=["DI"], dtype={"DI": np.float64},)
        damage_index = df_di.to_numpy()[:nX]
        
        max_nX = df_di.shape[0]
        assert max_nX>nX
        step = nX//signal
        max_step = max_nX//signal
        
        for s in range(signal):
            iBeg = s*max_step*Xsize
            iEnd = iBeg+step*Xsize-1
            for c in idChannels:
                data[s*step:(s+1)*step, c-1, :] = \
                    df_th.loc[iBeg:iEnd, "ACC{:>d}".format(c)].to_numpy().astype(np.float32).reshape(step,Xsize)

    percentage = 0.05
    noise = np.random.normal(0, data.std(), data.size).reshape(data.shape)*percentage
    # data += noise
    pga = np.tile(np.atleast_3d(np.max(data,axis=-1)),(1,1,Xsize))
    
    X = data/pga
    
    X = np.pad(X,((0,0),(0,0),(0,nxtpow2(X.shape[-1])-X.shape[-1])))
    X = np.swapaxes(X,1,2)

    n = []
    fid_th = open(opj(store_dir,"magnitude.csv"))
    th = csv.reader(fid_th)
    for row in th:
        n.append(row)

    m = np.array(n,dtype=np.float32)
    magnitude = np.zeros((nX,1),dtype=np.float32)
    for i in range(int(nX/N/signal)):
        magnitude[i+int(nX/signal)] = m[i]
        magnitude[i+int(nX/signal)+int(nX/signal/N)] = m[i]

    damage_class = np.zeros((nX,latentCdim),dtype=np.float32)
   
    for i in range(latentCdim):
        damage_class[nX//latentCdim*i:nX//latentCdim*(i+1),i] = 1.0

    for i in range(latentCdim):
        h5f = h5py.File(opj(store_dir,"Damaged_{:>d}.h5".format(i)),'w')
        h5f.create_dataset('X{:>d}'.format(i), data=X[nX//latentCdim*i:nX//latentCdim*(i+1),:,:])
        h5f.create_dataset('c{:>d}'.format(i), data=damage_class[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.create_dataset('magnitude{:>d}'.format(i), data=magnitude[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.create_dataset('d{:>d}'.format(i), data=damage_index[nX//latentCdim*i:nX//latentCdim*(i+1)])
        h5f.close()

    h5f = h5py.File(opj(store_dir,"Data.h5"),'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('damage_class', data=damage_class)
    h5f.create_dataset('magnitude', data=magnitude)
    h5f.create_dataset('damage_index', data=damage_index)
    h5f.close()

    X, c, magnitude, damage_index = shuffle(
        X, damage_class, magnitude, damage_index, random_state=0)

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Mtrn, Mvld, Dtrn, Dvld = train_test_split(X, damage_class, magnitude, damage_index,
                                                                      random_state=0,
                                                                      test_size=0.1)
    
    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,(Ctrn,Mtrn,Dtrn))).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,(Cvld,Mvld,Dvld))).batch(batchSize)
        )

def LoadData(**kwargs):
    LoadData.__globals__.update(kwargs)

    fid_th = opj(store_dir,"Data.h5")
    
    h5f = h5py.File(fid_th,'r')
    X = h5f['X'][...]
    c = h5f['damage_class'][...]
    magnitude = h5f['magnitude'][...]
    d = h5f['damage_index'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Mtrn, Mvld, Dtrn, Dvld = train_test_split(X,c,magnitude,d,
                                                                      random_state=0,
                                                                      test_size=0.1)
    
    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,(Ctrn,Mtrn,Dtrn))).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,(Cvld,Mvld,Dvld))).batch(batchSize)
        )


def Load_Un_Damaged(i,**kwargs):
    Load_Un_Damaged.__globals__.update(kwargs)

    fid_th = opj(store_dir,"Damaged_{:>d}.h5".format(i))
    h5f = h5py.File(fid_th,'r')
    X = h5f['X{:>d}'.format(i)][...]
    c = h5f['c{:>d}'.format(i)][...]
    magnitude = h5f['magnitude{:>d}'.format(i)][...]
    d = h5f['d{:>d}'.format(i)][...]

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Mtrn, Mvld, Dtrn, Dvld = train_test_split(X, c, magnitude, d,
                                                                      random_state=0,
                                                                      test_size=0.1)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn,Mtrn,Dtrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Mvld,Dvld)).batch(batchSize)
        )