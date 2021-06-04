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


    dfParam = dd.read_csv(opj(dataroot,"parameters_model.csv")).astype('float32')


    dfParam["mean"] = dfParam.mean(axis=1)
    dfParam["std"] = dfParam.std(axis=1)

    
    X = []
    for channel in idChannels:
        if len(avu) == 0:
            dataSrc = opj(dataroot,"{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,case,channel))
        else:
            dataSrc = opj(dataroot,"{:>s}_{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,avu,case,channel))
    
        X.append(np.genfromtxt(dataSrc).astype(np.float32).reshape(nX,1,-1))

    X = np.concatenate(X,axis=1).reshape(-1,X[0].shape[-1])
    scaler = StandardScaler()
    X = scaler.fit_transform(X).reshape((nX,len(idChannels),-1))
    X = np.pad(X,((0,0),(0,0),(0,nxtpow2(X.shape[-1])-X.shape[-1])))
    X = np.swapaxes(X,1,2)
    
    # Sampling of categorical variables c
    c = np.zeros((latentCdim,X.shape[0]),dtype=np.float32)
    c[0,:] = 1.0
    np.random.shuffle(c)

    c = c.T
    # c = np.reshape(c,(*c.shape,1))
    # c = np.stack([c for _ in idChannels],axis=-1)

    # Sampling of continuous variables s
    # s = np.random.uniform(low=-1.0,high=1.0000001,size=(X.shape[0],latentSdim)).astype(np.float32)
    #Sampling of noise n
    # n = np.random.normal(loc=0.0,scale=1.0,size=(X.shape[0],latentNdim)).astype(np.float32)

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
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
        dfParam["mean"],dfParam["std"],)
    


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
