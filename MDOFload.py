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

def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def CreateData(**kwargs):
    CreateData.__globals__.update(kwargs)

    X1 = np.zeros((nX//3,nXchannels,Xsize),dtype=np.float32)
    i1 = 0

    for channel in idChannels:
        #load the measurements recorded by each single channel
        dataSrc = opj(dataroot_1,"undamaged_{:>s}_{:>s}_concat_dof_{:>d}.csv".format(avu,pb,channel))

        sdof=np.genfromtxt(dataSrc).astype(np.float32)

        i2=0

        for i3 in range(nX//3):
            X1[i3,i1,0:Xsize]=sdof[i2:(i2+Xsize)]
            i2=i2+Xsize

        i1 =i1+1

    X2 = np.zeros((nX//3,nXchannels,Xsize),dtype=np.float32)
    i1 = 0

    for channel in idChannels:
        #load the measurements recorded by each single channel
        dataSrc = opj(dataroot_2,"damaged_{:>s}_{:>s}_concat_dof_{:>d}.csv".format(avu,pb,channel))

        sdof=np.genfromtxt(dataSrc).astype(np.float32)

        i2=0

        for i3 in range(nX//3):
            X2[i3,i1,0:Xsize]=sdof[i2:(i2+Xsize)]
            i2=i2+Xsize
        i1 = i1+1

    X3 = np.zeros((nX//3,nXchannels,Xsize),dtype=np.float32)
    i1 = 0

    for channel in idChannels:
        #load the measurements recorded by each single channel
        dataSrc = opj(dataroot_3,"damaged_{:>s}_{:>s}_concat_dof_{:>d}.csv".format(avu,pb,channel))

        sdof=np.genfromtxt(dataSrc).astype(np.float32)

        i2=0

        for i3 in range(nX//3):
            X3[i3,i1,0:Xsize]=sdof[i2:(i2+Xsize)]
            i2=i2+Xsize
        i1 = i1+1

    c = np.zeros((nX,latentCdim),dtype=np.float32)
    X = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)

    y = 0
    for i in range(nX//3):
        pga_1 = np.max(np.absolute(X1[i,:,:]))
        pga_2 = np.max(np.absolute(X2[i,:,:]))
        pga_3 = np.max(np.absolute(X3[i,:,:]))
        pga = max(pga_1,pga_2,pga_3)
        for j in range(nXchannels):
            for k in range(Xsize):
                X[y,j,k] = X1[i,j,k]/pga
                X[y+1,j,k] = X2[i,j,k]/pga
                X[y+2,j,k] = X3[i,j,k]/pga
                c[y,0] = 1.0
                c[y+1,1] = 1.0
                c[y+2,2] = 1.0

                X1[i,j,k] = X1[i,j,k]/pga
                X2[i,j,k] = X2[i,j,k]/pga
                X3[i,j,k] = X3[i,j,k]/pga
        y = y+3   

            
    X = np.pad(X,((0,0),(0,0),(0,nxtpow2(X.shape[-1])-X.shape[-1])))
    X = np.swapaxes(X,1,2)

    X1 = np.pad(X1,((0,0),(0,0),(0,nxtpow2(X1.shape[-1])-X1.shape[-1])))
    X1 = np.swapaxes(X1,1,2)

    X2 = np.pad(X2,((0,0),(0,0),(0,nxtpow2(X2.shape[-1])-X2.shape[-1])))
    X2 = np.swapaxes(X2,1,2)

    X3 = np.pad(X3,((0,0),(0,0),(0,nxtpow2(X3.shape[-1])-X3.shape[-1])))
    X3 = np.swapaxes(X3,1,2)


    c1 = np.zeros((nX//3,latentCdim),dtype=np.float32)
    c1[:,0] = 1.0

    d1 = np.zeros((nX//3,domain),dtype=np.float32)
    d1[:,0] = 1.0

    h5f = h5py.File("Undamaged.h5",'w')
    h5f.create_dataset('X1', data=X1)
    h5f.create_dataset('c1', data=c1)
    h5f.create_dataset('d1', data=d1)
    h5f.close()


    c2 = np.zeros((nX//3,latentCdim),dtype=np.float32)
    c2[:,1] = 1.0

    d2 = np.zeros((nX//3,domain),dtype=np.float32)
    d2[:,0] = 1.0

    h5f = h5py.File("Damaged_1.h5",'w')
    h5f.create_dataset('X2', data=X2)
    h5f.create_dataset('c2', data=c2)
    h5f.create_dataset('d2', data=d2)
    h5f.close()

    c3 = np.zeros((nX//3,latentCdim),dtype=np.float32) 
    c3[:,2] = 1.0

    d3 = np.zeros((nX//3,domain),dtype=np.float32)
    d3[:,0] = 1.0

    h5f = h5py.File("Damaged_2.h5",'w')
    h5f.create_dataset('X3', data=X3)
    h5f.create_dataset('c3', data=c3)
    h5f.create_dataset('d3', data=d3)
    h5f.close()

    
    
    c = np.vstack((c1,c2,c3))

    X, c = shuffle(X, c, random_state=0)

    d = np.zeros((nX,domain),dtype=np.float32)
    d[:,0] = 1.0

    #s = np.random.normal(loc=0.0,scale=scaleS,size=[nX,latentSdim]).astype('float32')

    h5f = h5py.File("Data.h5",'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('c', data=c)
    h5f.create_dataset('d', data=d)
    h5f.close()

    
    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Dtrn, Dvld = train_test_split(X,c,d,random_state=0)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn,Dtrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Dvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Dvld)).batch(batchSize)
        )
    


def LoadData(**kwargs):
    LoadData.__globals__.update(kwargs)

    dataSrc = opj("Data.h5")
    
    h5f = h5py.File(dataSrc,'r')
    X = h5f['X'][...]
    c = h5f['c'][...]
    d = h5f['d'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Dtrn, Dvld = train_test_split(X,c,d,random_state=0,shuffle=True)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn,Dtrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Dvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Dvld)).batch(batchSize),
    )


def Load_Un_Damaged(**kwargs):
    Load_Un_Damaged.__globals__.update(kwargs)
    
    dataSrc = opj("Undamaged.h5")
    h5f = h5py.File(dataSrc,'r')
    X1 = h5f['X1'][...]
    c1 = h5f['c1'][...]
    c1 = h5f['c1'][...]
    h5f.close()

    dataSrc = opj("Damaged_1.h5")
    h5f = h5py.File(dataSrc,'r')
    X2 = h5f['X2'][...]
    c2 = h5f['c2'][...]
    c2 = h5f['c2'][...]
    h5f.close()

    dataSrc = opj("Damaged_2.h5")
    h5f = h5py.File(dataSrc,'r')
    X3 = h5f['X3'][...]
    c3 = h5f['c3'][...]
    c2 = h5f['c2'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn1, Xvld1, Ctrn1, Cvld1, Dtrn1, Dvld1, Xtrn2, Xvld2, Ctrn2, Cvld2, Dtrn2, Dvld2,
    Xtrn3, Xvld3, Ctrn3, Cvld3, Dtrn3, Dvld3 = train_test_split(X1,c1,d1,X2,c2,d2,X3,c3,d3,random_state=0,shuffle=True)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn1,Ctrn1,Dtrn1)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld1,Cvld1,Dvld1)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld1,Cvld1,Dvld1)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xtrn2,Ctrn2,Dtrn2)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld2,Cvld2,Dvld2)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld2,Cvld2,Dvld2)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xtrn3,Ctrn3,Dtrn3)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld3,Cvld3,Dvld3)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld3,Cvld3,Dvld3)).batch(batchSize),
    )
