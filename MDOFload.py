# MDOFload

# load data function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import tensorflow as tf
import numpy as np
import pandas as pd
from os.path import join as opj
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dask
import dask.dataframe as dd
import dask.array as da
import h5py

def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def CreateData(**kwargs):
    CreateData.__globals__.update(kwargs)


    dfParam = dd.read_csv(opj(dataroot_1,"parameters_model.csv")).astype('float32')


    dfParam["mean"] = dfParam.mean(axis=1)
    dfParam["std"] = dfParam.std(axis=1)

    
    
    # # Initialize wtdof_v tensor
    # for i1 in range(len(wtdof)):
    #     if i1==0:
    #         wtdof_v=[wtdof[i1]]
    #         wtdof_v=np.array(wtdof_v)
    #         np.expand_dims(wtdof_v, axis=0)
    #     else:
    #         i2=wtdof_v[i1-1]+wtdof[i1]
    #         np.concatenate((wtdof_v,i2), axis=1)

    # for i1 in range(nXchannels):
    #     #load the measurements recorded by each single channel
    #     if len(avu) == 0:
    #         dataSrc = opj(dataroot,"{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,case,idChannels[i1]))
    #     else:
    #         dataSrc = opj(dataroot,"{:>s}_{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,avu,case,idChannels[i1]))

    #     sdof=np.genfromtxt(dataSrc).astype(np.float32)

        
    #     #initialise X
    #     if i1==0:
    #         X = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    #     i2=0
    #     for i3 in range(nX):
    #         X[i3,i1,0:ntm]=sdof[i2:(i2+ntm)]
    #         i2=i2+ntm

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # X = np.pad(X,((0,0),(0,0),(0,nxtpow2(X.shape[-1])-X.shape[-1])))
    # X = np.swapaxes(X,1,2)

    # Load data - Pirellone
    X1 = []
    
    for channel in idChannels:
        if len(avu) == 0:
            dataSrc = opj(dataroot_1,"{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,case,channel))
        else:
            dataSrc = opj(dataroot_1,"{:>s}_{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,avu,case,channel))
    
        
        X1.append(np.genfromtxt(dataSrc).astype(np.float32).reshape(nX,1,-1))

    X1 = np.concatenate(X1,axis=1).reshape(-1,X1[0].shape[-1])
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1).reshape((nX,len(idChannels),-1))
    X1 = np.pad(X1,((0,0),(0,0),(0,nxtpow2(X1.shape[-1])-X1.shape[-1])))
    X1 = np.swapaxes(X1,1,2)



    # Load data - Edificio a taglio
    X2 = []
    for channel in idChannels:
        if len(avu) == 0:
            dataSrc = opj(dataroot_2,"{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,case,channel))
        else:
            dataSrc = opj(dataroot_2,"{:>s}_{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,avu,case,channel))
    
        
        X2.append(np.genfromtxt(dataSrc).astype(np.float32).reshape(nX,1,-1))

    X2 = np.concatenate(X2,axis=1).reshape(-1,X2[0].shape[-1])
    scaler = StandardScaler()
    X2 = scaler.fit_transform(X2).reshape((nX,len(idChannels),-1))
    X2 = np.pad(X2,((0,0),(0,0),(0,nxtpow2(X2.shape[-1])-X2.shape[-1])))
    X2 = np.swapaxes(X2,1,2)

    X = []

    idx = np.random.choice(4, 2, replace=False)
    idx.sort()

    if idx[0] == 0 and idx[1] == 1:
            X = X2[:,:,:]

    if idx[0] == 2 and idx[1] == 3:
            X = X1[:,:,:]

    if idx[0] == 0 and idx[1] != 1:
        X1 = X1[:,:,1:]
        if idx[1] == 2:
            X2 = X2[:,:,1:]
        else:
            X2 = X2[:,:,:-1]
        X = np.concatenate([X1, X2], axis=-1)

    if idx[0] == 1:
        X1 = X1[:,:,:-1]
        if idx[1] == 2:
            X2 = X2[:,:,1:]
        else:
            X2 = X2[:,:,:-1]
        X = np.concatenate([X1, X2], axis=-1)

   

    src_metadata = opj(dataroot_2,"{:>s}_labels_{:>s}.csv".format(pb,case))

    labels = np.genfromtxt(src_metadata)
    labels = labels.astype(int)

    c = np.zeros((nX,latentCdim),dtype=np.float32)

    for i1 in range(nX):
        c[i1,labels[i1]] = 1.0

 
    h5f = h5py.File("{:>s}_gdl.h5".format(dataSrc.split('_gdl_')[0]),'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('c', data=c)
    # h5f.create_dataset('s', data=s)
    # h5f.create_dataset('n', data=n)
    h5f.close()

    
    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld = train_test_split(X,c,random_state=5)

    # # Split between train and validation set of continuous variables s
    # s_trn, s_vld = train_test_split(s,random_state=5)

    # # Split between train and validation set of noise
    # n_trn, n_vld = train_test_split(n,random_state=5)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize)
        )
    


def LoadData(**kwargs):
    LoadData.__globals__.update(kwargs)
    if len(avu) == 0:
        dataSrc = opj(dataroot,"{:>s}_damaged_concat_{:>s}_gdl.h5".format(pb,case))
    else:
        dataSrc = opj(dataroot,"{:>s}_{:>s}_damaged_concat_{:>s}_gdl.h5".format(pb,avu,case))
    

    h5f = h5py.File(dataSrc,'r')
    X = h5f['X'][...]
    c = h5f['c'][...]
    # s = h5f['s'][...]
    # n = h5f['n'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld = train_test_split(X,c,random_state=5)

    # # Split between train and validation set of continuous variables s
    # s_trn, s_vld = train_test_split(s,random_state=5)

    # # Split between train and validation set of noise
    # n_trn, n_vld = train_test_split(n,random_state=5)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
    )

# def LoadNumpyData(**kwargs):
#     LoadData.__globals__.update(kwargs)
#     if len(avu) == 0:
#         dataSrc = opj(dataroot,"{:>s}_damaged_concat_{:>s}_gdl.h5".format(pb,case))
#     else:
#         dataSrc = opj(dataroot,"{:>s}_{:>s}_damaged_concat_{:>s}_gdl.h5".format(pb,avu,case))
    

#     h5f = h5py.File(dataSrc,'r')
#     X = h5f['X'][...]
#     c = h5f['c'][...]
#     # s = h5f['s'][...]
#     # n = h5f['n'][...]
#     h5f.close()

#     # Split between train and validation set (time series and parameters are splitted in the same way)
#     Xtrn, Xvld, Ctrn, Cvld = train_test_split(X,c,random_state=5)
    
#     return (Xtrn,Ctrn),(Xvld,Cvld),(Xvld,Cvld)