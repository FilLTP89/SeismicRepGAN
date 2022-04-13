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
from random import randint

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

    # for j in range(data.shape[1]):
    #     for k in range(10):
    #         i = randint(0, data.shape[0]-1)
    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(data[i,j,:], color='black')
    #         #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results/signal_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results/reconstruction_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()
    
    j = 0
    for i in range(data.shape[0]):
        pga = np.zeros((nXchannels))
        for n in range(nXchannels):
            pga[n] = np.max(np.absolute(data[i,n,:]))
        pga_value = np.max(pga)
        if pga_value!=0:
            X[j,:,:] = data[i,:,:]/pga_value
            j = j+1
    
                
    # for i in range(X.shape[0]):
    #     pga = np.zeros((nXchannels))
    #     for n in range(nXchannels):
    #         pga[n] = np.max(np.absolute(X[i,n,:]))
    #     pga_value = np.max(pga)
    #     if pga_value==0:
    #         print(pga,i)
    

    X = np.pad(X,((0,0),(0,0),(0,nxtpow2(X.shape[-1])-X.shape[-1])))
    X = np.swapaxes(X,1,2)

    # for j in range(X.shape[2]):
    #     for k in range(10):
    #         i = randint(0, X.shape[0]-1)
    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(X[i,:,j], color='black')
    #         #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #         hax.set_ylim([-1.0, 1.0])
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results/signal_n_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results/reconstruction_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()
    

    c = np.zeros((nX,latentCdim),dtype=np.float32)
    for i in range(latentCdim):
        c[nX//latentCdim*i:nX//latentCdim*(i+1),i] = 1.0

    
    X,c = shuffle(X,c, random_state=0)

    for i in range(latentCdim):
        h5f = h5py.File("Damaged_{:>d}.h5".format(i),'w')
        h5f.create_dataset('X{:>d}'.format(i), data=X[nX//latentCdim*i:nX//latentCdim*(i+1),:,:])
        h5f.create_dataset('c{:>d}'.format(i), data=c[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.close()

    h5f = h5py.File("Data.h5",'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('c', data=c)
    h5f.close()


    
    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld = train_test_split(X,c,random_state=0)


    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize)
        )
    


def LoadData(**kwargs):
    LoadData.__globals__.update(kwargs)

    dataSrc = opj("Data.h5")
    
    h5f = h5py.File(dataSrc,'r')
    X = h5f['X'][...]
    c = h5f['c'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld = train_test_split(X,c,random_state=0)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize)
        )


def Load_Un_Damaged(i,**kwargs):
    Load_Un_Damaged.__globals__.update(kwargs)

    dataSrc = opj("Damaged_{:>d}.h5".format(i))
    h5f = h5py.File(dataSrc,'r')
    X = h5f['X{:>d}'.format(i)][...]
    c = h5f['c{:>d}'.format(i)][...]

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld = train_test_split(X,c,random_state=0,shuffle=False)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize)
        )
