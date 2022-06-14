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

    data_0 = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    data = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)       
    
    
    for i in range(signal):

        for channel in idChannels:
            #load the measurements recorded by each single channel
            dataSrc = open(os.path.join(dataroot[i],"Acc_{:>d}.csv".format(channel)))
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

    # t = np.zeros(data.shape[2])
    # for k in range(data.shape[2]-1):
    #     t[k+1] = (k+1)*0.04

    
    # for k in range(10):
    #     i = randint(0, data.shape[0]-1)
    #     for j in range(nXchannels):
    #         fig, axs = plt.subplots(2, 2, figsize=(24,12))
    #         axs[0,0].plot(t,data_0[i,j,:],color='black')
    #         axs[0,0].set_title('Signals without noise')
    #         axs[0,0].set_ylabel(r'$X (t) \hspace{0.5} [1]$')
    #         axs[0,0].set_xlabel(r'$t \hspace{0.5} [s]$')
    #         axs[1,0].plot(t,noise[i,j,:],color='blue')
    #         axs[1,0].set_title('Added noise')
    #         axs[1,0].set_ylabel(r'$X (t) \hspace{0.5} [1]$')
    #         axs[1,0].set_xlabel(r'$t \hspace{0.5} [s]$')
    #         axs[0,1].plot(t,data[i,j,:],color='red')
    #         axs[0,1].set_title('Signals with noise')
    #         axs[0,1].set_ylabel(r'$X (t) \hspace{0.5} [1]$')
    #         axs[1,1].set_xlabel(r'$t \hspace{0.5} [s]$')
    #         axs[1,1].plot(t,data_0[i,j,:],color='black')
    #         axs[1,1].plot(t,data[i,j,:],color='red',linestyle="--")
    #         axs[1,1].set_title('Signals with noise')
    #         axs[1,1].set_ylabel(r'$X (t) \hspace{0.5} [1]$')
    #         axs[1,1].set_xlabel(r'$t \hspace{0.5} [s]$')
    #         plt.savefig('/gpfs/workdir/colombergi/GiorgiaGAN/results_tesi/Signals_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()

            
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
    dataSrc = open("/gpfs/workdir/colombergi/GiorgiaGAN/input data/magnitude.csv")
    file = csv.reader(dataSrc)
    for row in file:
        n.append(row)

    # c = np.zeros((nX,latentCdim),dtype=np.float32)
    # for i in range(latentCdim):
    #     c[nX//latentCdim*i:nX//latentCdim*(i+1),i] = 1.0

    # for i in range(latentCdim):
    #     h5f = h5py.File("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint_ultimo/06_05/Damaged_{:>d}.h5".format(i),'w')
    #     h5f.create_dataset('X{:>d}'.format(i), data=X[nX//latentCdim*i:nX//latentCdim*(i+1),:,:])
    #     h5f.create_dataset('c{:>d}'.format(i), data=c[nX//latentCdim*i:nX//latentCdim*(i+1),:])
    #     h5f.create_dataset('mag{:>d}'.format(i), data=mag[nX//latentCdim*i:nX//latentCdim*(i+1),:])
    #     h5f.close()
    
        
    m = np.array(n,dtype=np.float32)
    mag = np.zeros((nX,1),dtype=np.float32)
    for i in range(int(nX/N/signal)):
        mag[i+int(nX/signal)] = m[i]
        mag[i+int(nX/signal)+int(nX/signal/N)] = m[i]

    # Park and Ang damage index
    for i in range(signal):
        n = []
        dataSrc = open(os.path.join(dataroot_index[i],"damage_index.csv"))
        file = csv.reader(dataSrc)
        for row in file:
            n.append(row)
        c_i = np.array(n,dtype=np.float32)
        if i == 0:
            d = c_i
        else:
            d = np.concatenate((d,c_i))
    

    c = np.zeros((nX,latentCdim),dtype=np.float32)

    # i1 = 0
    # i2 = 0
    # i3 = 0

    # for i in range(nX):
    #     if d[i] <= 0.4:
    #         c[i,0] = 1.0
    #         i1 = i1+1
    #         if i1 == 1:
    #             c0 = (c[i,:]).reshape((i1,latentCdim))
    #             m0 = (mag[i,0]).reshape((i1,1))
    #             d0 = (d[i,0]).reshape((i1,1))
    #             X0 = (X[i,:,:]).reshape((i1,Xsize,nXchannels))
    #         else:
    #             c0 = np.concatenate((c0,(c[i,:]).reshape((1,latentCdim)))).reshape((i1,latentCdim))
    #             m0 = np.concatenate((m0,(mag[i,0]).reshape((1,1)))).reshape((i1,1))
    #             d0 = np.concatenate((d0,(d[i,0]).reshape((1,1)))).reshape((i1,1))
    #             X0 = np.concatenate((X0,(X[i,:,:]).reshape((1,Xsize,nXchannels)))).reshape((i1,Xsize,nXchannels))

    #     elif 0.4 < d[i] <= 1:
    #         c[i,1] = 1.0
    #         i2 = i2+1
    #         if i2 == 1:
    #             c1 = (c[i,:]).reshape((i2,latentCdim))
    #             m1 = (mag[i,0]).reshape((i2,1))
    #             d1 = (d[i,0]).reshape((i2,1))
    #             X1 = (X[i,:,:]).reshape((i2,Xsize,nXchannels))
    #         else:
    #             c1 = np.concatenate((c1,(c[i,:]).reshape((1,latentCdim)))).reshape((i2,latentCdim))
    #             m1 = np.concatenate((m1,(mag[i,0]).reshape((1,1)))).reshape((i2,1))
    #             d1 = np.concatenate((d1,(d[i,0]).reshape((1,1)))).reshape((i2,1))
    #             X1 = np.concatenate((X1,(X[i,:,:]).reshape((1,Xsize,nXchannels)))).reshape((i2,Xsize,nXchannels))

    #     else:
    #         c[i,2] = 1.0
    #         i3 = i3+1
    #         if i3 == 1:
    #             c2 = (c[i,:]).reshape((i3,latentCdim))
    #             m2 = (mag[i,0]).reshape((i3,1))
    #             d2 = (d[i,0]).reshape((i3,1))
    #             X2 = (X[i,:,:]).reshape((i3,Xsize,nXchannels))
    #         else:
    #             c2 = np.concatenate((c2,(c[i,:]).reshape((1,latentCdim)))).reshape((i3,latentCdim))
    #             m2 = np.concatenate((m2,(mag[i,0]).reshape((1,1)))).reshape((i3,1))
    #             d2 = np.concatenate((d2,(d[i,0]).reshape((1,1)))).reshape((i2,1))
    #             X2 = np.concatenate((X2,(X[i,:,:]).reshape((1,Xsize,nXchannels)))).reshape((i3,Xsize,nXchannels))
    
   
    for i in range(latentCdim):
        c[nX//latentCdim*i:nX//latentCdim*(i+1),i] = 1.0

    for i in range(latentCdim):
        h5f = h5py.File("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint_ultimo/06_05/Damaged_{:>d}.h5".format(i),'w')
        h5f.create_dataset('X{:>d}'.format(i), data=X[nX//latentCdim*i:nX//latentCdim*(i+1),:,:])
        h5f.create_dataset('c{:>d}'.format(i), data=c[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.create_dataset('mag{:>d}'.format(i), data=mag[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.create_dataset('d{:>d}'.format(i), data=d[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.close()

    h5f = h5py.File("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint_ultimo/06_05/Data.h5",'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('c', data=c)
    h5f.create_dataset('mag', data=mag)
    h5f.create_dataset('d', data=d)
    h5f.close()
    
    
    # h5f = h5py.File("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint_ultimo/06_05/Damaged_0.h5",'w')
    # h5f.create_dataset('X0', data=X0)
    # h5f.create_dataset('c0', data=c0)
    # h5f.create_dataset('m0', data=m0)
    # h5f.create_dataset('d0', data=d0)
    # h5f.close() 

    # h5f = h5py.File("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint_ultimo/06_05/Damaged_1.h5",'w')
    # h5f.create_dataset('X1', data=X1)
    # h5f.create_dataset('c1', data=c1)
    # h5f.create_dataset('m1', data=m1)
    # h5f.create_dataset('d1', data=d1)
    # h5f.close() 

    # h5f = h5py.File("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint_ultimo/06_05/Damaged_2.h5",'w')
    # h5f.create_dataset('X2', data=X2)
    # h5f.create_dataset('c2', data=c2)
    # h5f.create_dataset('m2', data=m2)
    # h5f.create_dataset('d2', data=d2)
    # h5f.close()   

    # h5f = h5py.File("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint_ultimo/06_05/Data.h5",'w')
    # h5f.create_dataset('X', data=X)
    # h5f.create_dataset('c', data=c)
    # h5f.create_dataset('mag', data=mag)
    # h5f.create_dataset('d', data=d)
    # h5f.close()

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

    dataSrc = opj("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint_ultimo/06_05/Data.h5")
    
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

    dataSrc = opj("/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint_ultimo/06_05/Damaged_{:>d}.h5".format(i))
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

    
