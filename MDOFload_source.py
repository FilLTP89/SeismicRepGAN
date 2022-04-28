 # MDOFload

# load data function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import tensorflow as tf
import numpy as np
import pandas as pd
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

def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def CreateData(**kwargs):
    CreateData.__globals__.update(kwargs)

    X = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    

    for i in range(latentCdim):
        i1 = 0
        for channel in idChannels:
            #load the measurements recorded by each single channel
            dataSrc = opj(dataroot_source[i],"damaged_{:>s}_{:>s}_concat_dof_{:>d}.csv".format(avu,pb,channel))

            sdof=np.genfromtxt(dataSrc).astype(np.float32)

            i2=0

            for i3 in range(i*nX//latentCdim,nX//latentCdim*(i+1)):
                X[i3,i1,0:Xsize]=sdof[i2:(i2+Xsize)]
                i2=i2+Xsize

            i1 =i1+1

    Xs = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    
    for i in range(nX//latentCdim):
        pga = np.zeros((latentCdim))
        for n in range(latentCdim):
            pga[n] = np.max(np.absolute(X[nX//latentCdim*n+i,:,:]))
        pga_value = np.max(pga)
        for n in range(latentCdim):
            Xs[nX//latentCdim*n+i,:,:] = X[nX//latentCdim*n+i,:,:]/pga_value

    Xs = np.pad(Xs,((0,0),(0,0),(0,nxtpow2(Xs.shape[-1])-Xs.shape[-1])))
    Xs = np.swapaxes(Xs,1,2)

    cs = np.zeros((nX,latentCdim),dtype=np.float32)
    for i in range(latentCdim):
        cs[nX//latentCdim*i:nX//latentCdim*(i+1),i] = 1.0

    ds = np.zeros((nX,domain),dtype=np.float32)
    ds[:,0] = 1.0

    for i in range(latentCdim):
        h5f = h5py.File("Damaged_source_{:>d}.h5".format(i),'w')
        h5f.create_dataset('X{:>d}'.format(i), data=Xs[nX//latentCdim*i:nX//latentCdim*(i+1),:,:])
        h5f.create_dataset('c{:>d}'.format(i), data=cs[nX//latentCdim*i:nX//latentCdim*(i+1),:])
        h5f.create_dataset('ds', data=ds[:nX//latentCdim,:])
        h5f.close()
    
    Xs, cs, ds = shuffle(Xs, cs, ds, random_state=0)

    h5f = h5py.File("Data_source.h5",'w')
    h5f.create_dataset('Xs', data=Xs)
    h5f.create_dataset('cs', data=cs)
    h5f.create_dataset('ds', data=ds)
    h5f.close()

    X = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    

    for i in range(latentCdim):
        i1 = 0
        for channel in idChannels:
            #load the measurements recorded by each single channel
            dataSrc = opj(dataroot_target[i],"damaged_{:>s}_{:>s}_concat_dof_{:>d}.csv".format(avu,pb,channel))

            sdof=np.genfromtxt(dataSrc).astype(np.float32)

            i2=0

            for i3 in range(nX//latentCdim):
                X[i3*i,i1,0:Xsize]=sdof[i2:(i2+Xsize)]
                i2=i2+Xsize

            i1 =i1+1

    Xt = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    
    for i in range(nX//latentCdim):
        pga = np.zeros((latentCdim))
        for n in range(latentCdim):
            pga[n] = np.max(np.absolute(X[nX//latentCdim*n+i,:,:]))
        pga_value = np.max(pga)
        for n in range(latentCdim):
            Xt[nX//latentCdim*n+i,:,:] = X[nX//latentCdim*n+i,:,:]/pga_value

    Xt = np.pad(Xt,((0,0),(0,0),(0,nxtpow2(Xt.shape[-1])-Xt.shape[-1])))
    Xt = np.swapaxes(Xt,1,2)

    dt = np.zeros((nX,domain),dtype=np.float32)
    dt[:,1] = 1.0

    for i in range(latentCdim):
        h5f = h5py.File("Damaged_target_{:>d}.h5".format(i),'w')
        h5f.create_dataset('X{:>d}'.format(i), data=Xt[nX//latentCdim*i:nX//latentCdim*(i+1),:,:])
        h5f.create_dataset('dt', data=dt[:nX//latentCdim,:])
    
    Xt, dt = shuffle(Xt, dt, random_state=0)

    h5f = h5py.File("Data_target.h5",'w')
    h5f.create_dataset('Xt', data=Xt)
    h5f.create_dataset('dt', data=dt)
    h5f.close()

    
    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn_s, Xvld_s, Ctrn_s, Cvld_s, Dtrn_s, Dvld_s, Xtrn_t, Xvld_t, Dtrn_t, Dvld_t = train_test_split(Xs,cs,ds,Xt,dt,random_state=0)


    return (
        tf.data.Dataset.from_tensor_slices((Xtrn_s,Ctrn_s,Dtrn_s,Xtrn_t,Dtrn_t)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld_s,Cvld_s,Dvld_s,Xvld_t,Dvld_t)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld_s,Cvld_s,Dvld_s,Xvld_t,Dvld_t)).batch(batchSize)
        )
    


def LoadData(**kwargs):
    LoadData.__globals__.update(kwargs)

    dataSrc = opj("Data_source.h5")
    
    h5f = h5py.File(dataSrc,'r')
    Xs = h5f['Xs'][...]
    cs = h5f['cs'][...]
    ds = h5f['ds'][...]
    h5f.close()

    dataSrc = opj("Data_target.h5")
    
    h5f = h5py.File(dataSrc,'r')
    Xt = h5f['Xt'][...]
    dt = h5f['dt'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn_s, Xvld_s, Ctrn_s, Cvld_s, Dtrn_s, Dvld_s, Xtrn_t, Xvld_t, Dtrn_t, Dvld_t = train_test_split(Xs,cs,ds,Xt,dt,random_state=0)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn_s,Ctrn_s,Dtrn_s,Xtrn_t,Dtrn_t)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld_s,Cvld_s,Dvld_s,Xvld_t,Dvld_t)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld_s,Cvld_s,Dvld_s,Xvld_t,Dvld_t)).batch(batchSize)
        )


def Load_Un_Damaged(i,**kwargs):
    Load_Un_Damaged.__globals__.update(kwargs)

    dataSrc = opj("Damaged_source_{:>d}.h5".format(i))
    h5f = h5py.File(dataSrc,'r')
    Xs = h5f['X{:>d}'.format(i)][...]
    cs = h5f['c{:>d}'.format(i)][...]
    ds = h5f['ds'][...]

    dataSrc = opj("Damaged_target_{:>d}.h5".format(i))
    h5f = h5py.File(dataSrc,'r')
    Xt = h5f['X{:>d}'.format(i)][...]
    dt = h5f['dt'][...]

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn_s, Xvld_s, Ctrn_s, Cvld_s, Dtrn_s, Dvld_s, Xtrn_t, Xvld_t, Dtrn_t, Dvld_t = train_test_split(Xs,cs,ds,Xt,dt,random_state=0,shuffle=False)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn_s,Ctrn_s,Dtrn_s,Xtrn_t,Dtrn_t)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld_s,Cvld_s,Dvld_s,Xvld_t,Dvld_t)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld_s,Cvld_s,Dvld_s,Xvld_t,Dvld_t)).batch(batchSize)
        )
