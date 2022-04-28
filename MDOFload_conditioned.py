 # MDOFload

# load data function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
import openturns as ot
import openturns.viewer as viewer
from random import randint
import pickle

def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def CreateData(**kwargs):
    CreateData.__globals__.update(kwargs)

    data = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)

    # for i in range(latentCdim):
    #     i1 = 0
    #     for channel in idChannels:
    #         #load the measurements recorded by each single channel
    #         dataSrc = opj(dataroot[i],"damaged_{:>s}_{:>s}_concat_dof_{:>d}.csv".format(avu,pb,channel))

    #         sdof=np.genfromtxt(dataSrc).astype(np.float32)

    #         i2=0
    #         for i3 in range(i*nX//latentCdim,nX//latentCdim*(i+1)):
    #             data[i3,i1,0:Xsize]=sdof[i2:(i2+Xsize)]
    #             i2=i2+Xsize

    #         i1 =i1+1
    n = []
    dataSrc = open("/gpfs/workdir/invsem07/GiorgiaGAN/magnitude.csv")
    file = csv.reader(dataSrc)
    for row in file:
        n.append(row)

    mag = np.array(n,dtype=np.float32)

    n = []
    dataSrc = open("/gpfs/workdir/invsem07/GiorgiaGAN/source_distance.csv")
    file = csv.reader(dataSrc)
    for row in file:
        n.append(row)

    dis = np.array(n,dtype=np.float32)

    with open('/gpfs/workdir/invsem07/GiorgiaGAN/NDOF_code/PortiqueElasPlas_E_2000/NDOF.pkl', 'rb') as f:
        val = pickle.load(f)
    
    v = np.zeros((nX,Vdim),dtype=np.float32)

    for i in range(N):
        k0, fy = val[i]
        for j in range(int(nX/N/latentCdim)):
            v[j*(i+1),0] = 0
            v[j*(i+1),1] = 0
            v[j*(i+1),2] = k0
            v[j*(i+1),3] = fy
            v[j*(i+1)+int(nX/latentCdim),0] = mag[j]
            v[j*(i+1)+int(nX/latentCdim),1] = dis[j]
            v[j*(i+1)+int(nX/latentCdim),2] = k0
            v[j*(i+1)+int(nX/latentCdim),3] = fy
    
    
    for i in range(latentCdim):

        for channel in idChannels:
            #load the measurements recorded by each single channel
            dataSrc = open(os.path.join(dataroot[i],"Acc_{:>d}.csv".format(channel)))
            file = csv.reader(dataSrc)


            i1 = int(nX/latentCdim)*i
            i2 = 0
            for row in file:
                data[i1,channel-1,i2] = np.array([row])
                i2 = i2+1
                if i2 == Xsize:
                    i2 = 0
                    i1 = i1+1
    
    
    X = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)

    load = np.loadtxt('/gpfs/workdir/invsem07/GiorgiaGAN/acc_x.txt')
    load1 = np.loadtxt('/gpfs/workdir/invsem07/GiorgiaGAN/NDOF_code/noise_x.txt')
    acc = np.zeros((int(load.shape[0]/4),load.shape[1]-1))
    acc1 = np.zeros((int(load1.shape[0]/4),load.shape[1]-1))
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            acc[i,j] = load[i*4,j+1]
            acc1[i,j] = load1[i*4,j+1]

    H = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)

    for k in range(N):
        for i in range(int(nX/N/latentCdim)):
            for j in range(nXchannels):
                H[i+1000*k,j,:] = np.fft.fft(data[i,j,:])/np.fft.fft(acc1[:,i])
                H[i+1000*k+int(nX/latentCdim),j,:] = np.fft.fft(data[i+int(nX/latentCdim),j,:])/np.fft.fft(acc[:,i])
    
    j = 0
    for i in range(data.shape[0]):
        pga = np.zeros((nXchannels))
        pgaH = np.zeros((nXchannels))
        for n in range(nXchannels):
            pga[n] = np.max(np.absolute(data[i,n,:]))
            pgaH[n] = np.max(np.absolute(H[i,n,:]))
        pga_value = np.max(pga)
        pga_valueH = np.max(pgaH)
        if pga_value!=0:
            X[j,:,:] = H[i,:,:]/pga_valueH
            j = j+1
    
            
    t = np.zeros(X.shape[2])
    for k in range(X.shape[2]-1):
        t[k+1] = (k+1)*0.04

    n = X.shape[2]
    timestep = 0.04
    freq = np.fft.fftfreq(n, d=timestep)

    s = int(H.shape[2]/2)
            
    # for i in range(X.shape[0]):
    #     pga = np.zeros((nXchannels))
    #     for n in range(nXchannels):
    #         pga[n] = np.max(np.absolute(X[i,n,:]))
    #     pga_value = np.max(pga)
    #     if pga_value==0:
    #         print(pga,i)
    


    X = np.pad(X,((0,0),(0,0),(0,nxtpow2(X.shape[-1])-X.shape[-1])))
    X = np.swapaxes(X,1,2)

    # for j in range(X.shape[1]):
    #     for k in range(10):
    #         i = randint(0, X.shape[0]-1)
    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.loglog(freq[:s], np.abs(H[i,j,s:]), color='black')
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_TF/abs_H_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.loglog(freq[:s], np.abs(X[i,j,s:]), color='black')
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_TF/abs_X_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()
    

    c = np.zeros((nX,latentCdim),dtype=np.float32)
    for i in range(latentCdim):
        c[nX//latentCdim*i:nX//latentCdim*(i+1),i] = 1.0

    
    X,c,v = shuffle(X,c,v, random_state=0)

    for i in range(latentCdim):
        h5f = h5py.File("Damaged_conditioned_{:>d}.h5".format(i),'w')
        h5f.create_dataset('X{:>d}'.format(i), data=X[nX//latentCdim*i:nX//latentCdim*(i+1),:,:])
        h5f.create_dataset('c{:>d}'.format(i), data=c[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.create_dataset('v{:>d}'.format(i), data=v[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.close()

    h5f = h5py.File("Data_conditioned.h5",'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('c', data=c)
    h5f.create_dataset('v', data=v)
    h5f.close()


    
    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Vtrn, Vvld = train_test_split(X,c,v,random_state=0)


    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn,Vtrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Vvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Vvld)).batch(batchSize)
        )
    


def LoadData(**kwargs):
    LoadData.__globals__.update(kwargs)

    dataSrc = opj("/gpfs/workdir/invsem07/GiorgiaGAN/checkpoint_c/13_04/Data_conditioned.h5")
    
    h5f = h5py.File(dataSrc,'r')
    X = h5f['X'][...]
    c = h5f['c'][...]
    v = h5f['v'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Vtrn, Vvld = train_test_split(X,c,v,random_state=0)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn,Vtrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Vvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Vvld)).batch(batchSize)
        )


def Load_Un_Damaged(i,**kwargs):
    Load_Un_Damaged.__globals__.update(kwargs)

    dataSrc = opj("/gpfs/workdir/invsem07/GiorgiaGAN/checkpoint_c/13_04/Damaged_conditioned_{:>d}.h5".format(i))
    h5f = h5py.File(dataSrc,'r')
    X = h5f['X{:>d}'.format(i)][...]
    c = h5f['c{:>d}'.format(i)][...]
    v = h5f['v{:>d}'.format(i)][...]

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Vtrn, Vvld = train_test_split(X,c,v,random_state=0,shuffle=False)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn,Vtrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Vvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Vvld)).batch(batchSize)
        )
