# MDOFload

# load data function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import tensorflow as tf
import numpy as np
import pandas as pd
from os.path import join as opj
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.utils import shuffle
import dask
import dask.dataframe as dd
import dask.array as da
import h5py
from matplotlib import pyplot as plt

def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def CreateData(**kwargs):
    CreateData.__globals__.update(kwargs)


    # dfParam1 = dd.read_csv(opj(dataroot_1,"parameters_model.csv")).astype('float32')


    # dfParam1["mean"] = dfParam1.mean(axis=1)
    # dfParam1["std"] = dfParam1.std(axis=1)

    # dfParam2 = dd.read_csv(opj(dataroot_2,"parameters_model.csv")).astype('float32')


    # dfParam2["mean"] = dfParam2.mean(axis=1)
    # dfParam2["std"] = dfParam2.std(axis=1)

    
    
    # # Initialize wtdof_v tensor
    # for i1 in range(len(wtdof)):
    #     if i1==0:
    #         wtdof_v=[wtdof[i1]]
    #         wtdof_v=np.array(wtdof_v)
    #         np.expand_dims(wtdof_v, axis=0)
    #     else:
    #         i2=wtdof_v[i1-1]+wtdof[i1]
    #         np.concatenate((wtdof_v,i2), axis=1)

    

    # # Load data - Undamaged

    # for i1 in range(nXchannels):
    #     #load the measurements recorded by each single channel
    #     dataSrc = opj(dataroot_1,"undamaged_{:>s}_{:>s}_concat_dof_{:>d}.csv".format(avu,pb,idChannels[i1]))

    #     sdof=np.genfromtxt(dataSrc).astype(np.float32)
        
    #     #initialise X
    #     if i1==0:
    #         X1 = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    #     #i2=0
    #     for i3 in range(nX):
    #         for j in range(Xsize):
    #             X1[i3,i1,j]=sdof[i3+j*nX]
    #             #i2=i2+ntm

    
    # scaler = StandardScaler()
    # for i1 in range(nXchannels):
    #     X1[:,i1,:] = scaler.fit_transform(X1[:,i1,:])
 

    # X1 = np.pad(X1,((0,0),(0,0),(0,nxtpow2(X1.shape[-1])-X1.shape[-1])))
    # X1 = np.swapaxes(X1,1,2)
    # signal = np.zeros((nX,Xsize),dtype=np.float32)
    # signal[:,:] = X1[:,:,1]
    # np.savetxt("prova.csv", signal, delimiter=",")


    # src_metadata = opj(dataroot_1,"undamaged_{:>s}_labels.csv".format(pb))

    # labels = np.genfromtxt(src_metadata)
    # labels = labels.astype(int)

    # c1 = np.zeros((nX,latentCdim),dtype=np.float32)

    # for i1 in range(nX):
    #     c1[i1,labels[i1]] = 1.0

    # h5f = h5py.File("Undamaged.h5",'w')
    # h5f.create_dataset('X1', data=X1)
    # h5f.create_dataset('c1', data=c1)
    # h5f.close()

    # # Load data - Damaged

    # for i1 in range(nXchannels):
    #     #load the measurements recorded by each single channel
    #     dataSrc = opj(dataroot_2,"damaged_{:>s}_{:>s}_concat_dof_{:>d}.csv".format(avu,pb,idChannels[i1]))

    #     sdof=np.genfromtxt(dataSrc).astype(np.float32)

        
    #     #initialise X
    #     if i1==0:
    #         X2 = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    #     i2=0
    #     for i3 in range(nX):
    #         X2[i3,i1,0:ntm]=sdof[i2:(i2+ntm)]
    #         i2=i2+ntm

    # scaler = MinMaxScaler(feature_range=(-1,1))
    # for i1 in range(nXchannels):
    #     X2[:,i1,:] = scaler.fit_transform(X2[:,i1,:])


    # X2 = np.pad(X2,((0,0),(0,0),(0,nxtpow2(X2.shape[-1])-X2.shape[-1])))
    # X2 = np.swapaxes(X2,1,2)

    # src_metadata = opj(dataroot_2,"damaged_{:>s}_labels.csv".format(pb))

    # labels = np.genfromtxt(src_metadata)
    # labels = labels.astype(int)

    # c2 = np.zeros((nX,latentCdim),dtype=np.float32)

    # for i1 in range(nX):
    #     c2[i1,labels[i1]] = 1.0

    # h5f = h5py.File("Damaged.h5",'w')
    # h5f.create_dataset('X2', data=X2)
    # h5f.create_dataset('c2', data=c2)
    # h5f.close()





    # Load data - Undamaged
    X1 = []
    
    for channel in idChannels:
        dataSrc = opj(dataroot_1,"undamaged_{:>s}_{:>s}_concat_dof_{:>d}.csv".format(avu,pb,channel))
        # if len(avu) == 0:
        #     dataSrc = opj(dataroot_1,"{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,case,channel))
        # else:
        #     dataSrc = opj(dataroot_1,"{:>s}_{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,avu,case,channel))
    
        
        X1.append(np.genfromtxt(dataSrc).astype(np.float32).reshape(nX//2,1,-1))


    X1 = np.concatenate(X1,axis=1).reshape(-1,X1[0].shape[-1])
    #scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1).reshape((nX//2,len(idChannels),-1))
    # X1 = X1.reshape((nX,len(idChannels),-1))
    # for i1 in range(nXchannels):
    #     X1[:,i1,:] = scaler.fit_transform(X1[:,i1,:])
    X1 = np.pad(X1,((0,0),(0,0),(0,nxtpow2(X1.shape[-1])-X1.shape[-1])))
    X1 = np.swapaxes(X1,1,2)

    #src_metadata = opj(dataroot_1,"undamaged_{:>s}_labels.csv".format(pb))
    #src_metadata = opj(dataroot_1,"{:>s}_labels_{:>s}.csv".format(pb,case))

    #labels = np.genfromtxt(src_metadata)
    #labels = labels.astype(int)

    c1 = np.zeros((nX//2,latentCdim),dtype=np.float32)

    # for i1 in range(nX):
    #     c1[i1,labels[i1]] = 1.0
    c1[:,0] = 1.0

    h5f = h5py.File("Undamaged.h5",'w')
    h5f.create_dataset('X1', data=X1)
    h5f.create_dataset('c1', data=c1)
    h5f.close()


    # Load data - Damaged
    X2 = []
    for channel in idChannels:
        dataSrc = opj(dataroot_2,"damaged_{:>s}_{:>s}_concat_dof_{:>d}.csv".format(avu,pb,channel))
        # if len(avu) == 0:
        #     dataSrc = opj(dataroot_2,"{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,case,channel))
        # else:
        #     dataSrc = opj(dataroot_2,"{:>s}_{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(pb,avu,case,channel))
    
        
        X2.append(np.genfromtxt(dataSrc).astype(np.float32).reshape(nX//2,1,-1))

    X2 = np.concatenate(X2,axis=1).reshape(-1,X2[0].shape[-1])
    #scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = StandardScaler()
    X2 = scaler.fit_transform(X2).reshape((nX//2,len(idChannels),-1))
    # X2 = X2.reshape((nX,len(idChannels),-1))
    # for i1 in range(nXchannels):
    #     X2[:,i1,:] = scaler.fit_transform(X2[:,i1,:])
    X2 = np.pad(X2,((0,0),(0,0),(0,nxtpow2(X2.shape[-1])-X2.shape[-1])))
    X2 = np.swapaxes(X2,1,2)

    #src_metadata = opj(dataroot_2,"damaged_{:>s}_labels.csv".format(pb))
    #src_metadata = opj(dataroot_2,"{:>s}_labels_{:>s}.csv".format(pb,case))

    #labels = np.genfromtxt(src_metadata)
    #labels = labels.astype(int)

    c2 = np.zeros((nX//2,latentCdim),dtype=np.float32)

    # for i1 in range(nX):
    #     c2[i1,labels[i1]] = 1.0
    c2[:,1] = 1.0

    h5f = h5py.File("Damaged.h5",'w')
    h5f.create_dataset('X2', data=X2)
    h5f.create_dataset('c2', data=c2)
    h5f.close()

    #X = np.zeros_like(X1)
    #c = np.zeros_like(c1)

    # idx = np.random.choice(4, 2, replace=False)
    # idx.sort()

    # if idx[0] == 0 and idx[1] == 1:
    #         X = X2[:,:,:]

    # if idx[0] == 2 and idx[1] == 3:
    #         X = X1[:,:,:]

    # if idx[0] == 0 and idx[1] != 1:
    #     X1 = X1[:,:,1:]
    #     if idx[1] == 2:
    #         X2 = X2[:,:,1:]
    #     else:
    #         X2 = X2[:,:,:-1]
    #     X = np.concatenate([X1, X2], axis=-1)

    # if idx[0] == 1:
    #     X1 = X1[:,:,:-1]
    #     if idx[1] == 2:
    #         X2 = X2[:,:,1:]
    #     else:
    #         X2 = X2[:,:,:-1]
    #     X = np.concatenate([X1, X2], axis=-1)


    # idx = np.random.choice(nX, nX//2, replace=False)
    # idx.sort()

    # for i in range (nX//2):
    #     X[i,:,:] = X1[idx[i],:,:]
    #     X[(i+nX//2),:,:] = X2[idx[i],:,:]
    #     c[i,:] = c1[idx[i],:]
    #     c[(i+nX//2),:] = c2[idx[i],:]

    X = np.vstack((X1,X2))
    c = np.vstack((c1,c2))

    #X, c = shuffle(X, c, random_state=0)



    h5f = h5py.File("{:>s}_gdl.h5".format(dataSrc.split('_gdl_')[0]),'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('c', data=c)
    h5f.close()

    
    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld = train_test_split(X,c,random_state=5)

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
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld = train_test_split(X,c,random_state=5)

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



def LoadUndamaged(**kwargs):
    LoadUndamaged.__globals__.update(kwargs)
    dataSrc = opj("Undamaged.h5")
    

    h5f = h5py.File(dataSrc,'r')
    X = h5f['X1'][...]
    c = h5f['c1'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld = train_test_split(X,c,random_state=5)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
    )

def LoadDamaged(**kwargs):
    LoadDamaged.__globals__.update(kwargs)
    dataSrc = opj("Damaged.h5")
    

    h5f = h5py.File(dataSrc,'r')
    X = h5f['X2'][...]
    c = h5f['c2'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld = train_test_split(X,c,random_state=5)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld)).batch(batchSize),
    )

    