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

    # percentage = 0.05
    # noise = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    # for i in range(nX):
    #     for j in range(nXchannels):
    #         noise[i,j,:] = np.random.normal(0,data_0[i,j,:].std(),Xsize)*percentage
    #         data[i,j,:] = data_0[i,j,:] + noise[i,j,:]

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
    #         plt.savefig('./results_tesi/Signals_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()

    # New dataset
    load = np.loadtxt('./acc_x.txt')
    load1 = np.loadtxt('./NDOF_code/noise_x.txt')
    acc = np.zeros((int(load.shape[0]/4),load.shape[1]-1))
    acc1 = np.zeros((int(load1.shape[0]/4),load.shape[1]-1))
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            acc[i,j] = load[i*4,j+1]
            acc1[i,j] = load1[i*4,j+1]

    H = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)

    for k in range(N):
        for i in range(int(nX/N/signal)):
            for j in range(nXchannels):
                H[i+int(nX/N/signal)*k,j,:] = np.fft.fft(data[i,j,:])/np.fft.fft(acc1[:,i])
                H[i+int(nX/N/signal)*k+int(nX/N),j,:] = np.fft.fft(data[i+int(nX/N),j,:])/np.fft.fft(acc[:,i])

    # # Dataset tesi
    # dataSrc1 = open("./signal_500.csv")
    # sdof_1 = csv.reader(dataSrc1,delimiter=',')

    # n1 = []
    # for row in sdof_1:
    #     n1.append(row)

    # load = np.array(n1,dtype=np.float32)

    
    # acc = np.zeros((int(nX/signal),Xsize))
    # for i in range(acc.shape[0]):
    #     for j in range(acc.shape[1]):
    #         acc[i,j] = load[i,j*4]
    
    # H = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    # for k in range(signal):
    #     for i in range(int(nX/signal)):
    #         for j in range(nXchannels):
    #             H[i+int(nX/signal)*k,j,:] = np.fft.fft(data[i,j,:])/np.fft.fft(acc[i,:])
    
    X = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)

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
    
      

    # t = np.zeros(X.shape[2])
    # for k in range(X.shape[2]-1):
    #     t[k+1] = (k+1)*0.04

    # n = X.shape[2]
    # timestep = 0.04
    # freq = np.fft.fftfreq(n, d=timestep)

    # s = int(H.shape[2]/2)
        

    # for j in range(X.shape[1]):
    #     for k in range(10):
    #         i = randint(0, X.shape[0]-1)
    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.loglog(freq[:s], np.abs(H[i,j,s:]), color='black')
    #         plt.savefig('./results_TF/abs_H_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.loglog(freq[:s], np.abs(X[i,j,s:]), color='black')
    #         plt.savefig('./results_TF/abs_X_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()
    

    X = np.pad(X,((0,0),(0,0),(0,nxtpow2(X.shape[-1])-X.shape[-1])))
    X = np.swapaxes(X,1,2)

    
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

    i1 = 0
    i2 = 0
    i3 = 0

    for i in range(nX):
        if d[i] <= 0.4:
            c[i,0] = 1.0
            i1 = i1+1
            if i1 == 1:
                c0 = (c[i,:]).reshape((i1,latentCdim))
                X0 = (X[i,:,:]).reshape((i1,Xsize,nXchannels))
            else:
                c0 = np.concatenate((c0,(c[i,:]).reshape((1,latentCdim)))).reshape((i1,latentCdim))
                X0 = np.concatenate((X0,(X[i,:,:]).reshape((1,Xsize,nXchannels)))).reshape((i1,Xsize,nXchannels))

        elif 0.4 < d[i] <= 1:
            c[i,1] = 1.0
            i2 = i2+1
            if i2 == 1:
                c1 = (c[i,:]).reshape((i2,latentCdim))
                X1 = (X[i,:,:]).reshape((i2,Xsize,nXchannels))
            else:
                c1 = np.concatenate((c1,(c[i,:]).reshape((1,latentCdim)))).reshape((i2,latentCdim))
                X1 = np.concatenate((X1,(X[i,:,:]).reshape((1,Xsize,nXchannels)))).reshape((i2,Xsize,nXchannels))

        else:
            c[i,2] = 1.0
            i3 = i3+1
            if i3 == 1:
                c2 = (c[i,:]).reshape((i3,latentCdim))
                X2 = (X[i,:,:]).reshape((i3,Xsize,nXchannels))
            else:
                c2 = np.concatenate((c2,(c[i,:]).reshape((1,latentCdim)))).reshape((i3,latentCdim))
                X2 = np.concatenate((X2,(X[i,:,:]).reshape((1,Xsize,nXchannels)))).reshape((i3,Xsize,nXchannels))
    
    X,c = shuffle(X,c, random_state=0)

    h5f = h5py.File("./checkpoint_ultimo/14_04/Damaged_0.h5",'w')
    h5f.create_dataset('X0', data=X0)
    h5f.create_dataset('c0', data=c0)
    h5f.close() 

    h5f = h5py.File("./checkpoint_ultimo/14_04/Damaged_1.h5",'w')
    h5f.create_dataset('X1', data=X1)
    h5f.create_dataset('c1', data=c1)
    h5f.close() 

    h5f = h5py.File("./checkpoint_ultimo/14_04/Damaged_2.h5",'w')
    h5f.create_dataset('X2', data=X2)
    h5f.create_dataset('c2', data=c2)
    h5f.close()   


    h5f = h5py.File("./checkpoint_ultimo/14_04/Data.h5",'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('c', data=c)
    h5f.close()

    # c = np.zeros((nX,latentCdim),dtype=np.float32)
    # for i in range(latentCdim):
    #     c[nX//latentCdim*i:nX//latentCdim*(i+1),i] = 1.0

    
    X,c = shuffle(X,c, random_state=0)
    
    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld = train_test_split(X,c,random_state=0)


    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize)
        )
    


def LoadData(**kwargs):
    LoadData.__globals__.update(kwargs)

    dataSrc = opj("./checkpoint_ultimo/14_04/Data.h5")
    
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

    dataSrc = opj("./checkpoint_ultimo/14_04/Damaged_{:>d}.h5".format(i))
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
