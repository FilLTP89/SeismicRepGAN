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

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from os.path import join as opj
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.utils import shuffle
import dask
import dask.dataframe as dd
import dask.array as da
import h5py
from matplotlib import pyplot as plt
import seaborn as sn
import sys
import csv
from random import randint

def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def CreateData(**kwargs):
    CreateData.__globals__.update(kwargs)

    data_0 = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    data = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)       
    
    
    for i in range(signal):

        for channel in idChannels:
            #load the measurements recorded by each single channel
            dataSrc = open(opj(rawdata_dir[i],"Acc_{:>d}.csv".format(channel)))
            file = csv.reader(dataSrc)


            i1 = int(nX/signal)*i
            i2 = 0
            for row in file:
                data_0[i1,channel-1,i2] = np.array([row])
                i2 = i2+1
                if i2 == Xsize:
                    i2 = 0
                    i1 = i1+1

    percentage = 0.05
    noise = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    for i in range(nX):
        for j in range(nXchannels):
            noise[i,j,:] = np.random.normal(0,data_0[i,j,:].std(),Xsize)*percentage
            data[i,j,:] = data_0[i,j,:] + noise[i,j,:]
            
    X = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)

    j = 0
    for i in range(data.shape[0]):
        pga = np.zeros((nXchannels))
        for n in range(nXchannels):
            pga[n] = np.max(np.absolute(data[i,n,:]))
        pga_value = np.max(pga)
        if pga_value!=0:
            X[j,:,:] = data[i,:,:]/pga_value
            j = j+1
    
    
    X = np.pad(X,((0,0),(0,0),(0,nxtpow2(X.shape[-1])-X.shape[-1])))
    X = np.swapaxes(X,1,2)

    n = []
    dataSrc = open(opj(store_dir,"magnitude.csv"))
    file = csv.reader(dataSrc)
    for row in file:
        n.append(row)
        
    m = np.array(n,dtype=np.float32)
    mag = np.zeros((nX,1),dtype=np.float32)
    for i in range(int(nX/N/signal)):
        mag[i+int(nX/signal)] = m[i]
        mag[i+int(nX/signal)+int(nX/signal/N)] = m[i]

    # Park and Ang damage index
    for i in range(signal):
        n = []
        dataSrc = open(opj(rawdata_dir[i],"damage_index.csv"))
        file = csv.reader(dataSrc)
        for row in file:
            n.append(row)
        c_i = np.array(n,dtype=np.float32)
        if i == 0:
            d = c_i
        else:
            d = np.concatenate((d,c_i))
    

    c = np.zeros((nX,latentCdim),dtype=np.float32)
   
    for i in range(latentCdim):
        c[nX//latentCdim*i:nX//latentCdim*(i+1),i] = 1.0

    for i in range(latentCdim):
        h5f = h5py.File("/gpfs/workdir/colombergi/GiorgiaGAN/input data/Damaged_{:>d}.h5".format(i),'w')
        h5f.create_dataset('X{:>d}'.format(i), data=X[nX//latentCdim*i:nX//latentCdim*(i+1),:,:])
        h5f.create_dataset('c{:>d}'.format(i), data=c[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.create_dataset('mag{:>d}'.format(i), data=mag[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.create_dataset('d{:>d}'.format(i), data=d[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.close()

    h5f = h5py.File("/gpfs/workdir/colombergi/GiorgiaGAN/input data/Data.h5",'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('c', data=c)
    h5f.create_dataset('mag', data=mag)
    h5f.create_dataset('d', data=d)
    h5f.close()

    X,c,mag,d = shuffle(X,c,mag,d, random_state=0)

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Mtrn, Mvld, Dtrn, Dvld = train_test_split(X,c,mag,d,random_state=0)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn,Mtrn,Dtrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Mvld,Dvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Mvld,Dvld)).batch(batchSize)
        )
    


def LoadData(**kwargs):
    LoadData.__globals__.update(kwargs)

    dataSrc = opj(store_dir,"Data.h5")
    
    h5f = h5py.File(dataSrc,'r')
    X = h5f['X'][...]
    c = h5f['c'][...]
    mag = h5f['mag'][...]
    d = h5f['d'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Mtrn, Mvld, Dtrn, Dvld = train_test_split(X,c,mag,d,random_state=0)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn,Mtrn,Dtrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Mvld,Dvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Mvld,Dvld)).batch(batchSize)
        )


def Load_Un_Damaged(i,**kwargs):
    Load_Un_Damaged.__globals__.update(kwargs)

    dataSrc = opj(store_dir,"Damaged_{:>d}.h5".format(i))
    h5f = h5py.File(dataSrc,'r')
    X = h5f['X{:>d}'.format(i)][...]
    c = h5f['c{:>d}'.format(i)][...]
    mag = h5f['mag{:>d}'.format(i)][...]
    d = h5f['d{:>d}'.format(i)][...]

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Mtrn, Mvld, Dtrn, Dvld = train_test_split(X,c,mag,d,random_state=0,shuffle=False)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn,Mtrn,Dtrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Mvld,Dvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Mvld,Dvld)).batch(batchSize)
        )