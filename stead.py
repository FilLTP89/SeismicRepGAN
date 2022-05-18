import argparse
import math as mt

import tensorflow as tf
#tf.config.run_functions_eagerly(True)

import numpy as np
import pandas as pd
from os.path import join as osj
import h5py

from tensorflow import keras
import timeit
from numpy.random import randn
from numpy.random import randint
from ss_process import rsp
from scipy.signal import detrend, windows
import matplotlib
from matplotlib import pyplot as plt
import obspy
import seaborn as sn
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client



def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",type=str,default="/gpfs/workdir/invsem07/STEAD",help="Data root folder")
    parser.add_argument("--cuda",action='store_true',default=False,help='Use cuda powered GPU')
    parser.add_argument('--dtm',type=float,default=0.01,help='time-step [s]')
    parser.add_argument('--ntm',type=int,default=4096,help='Number of time steps')
    parser.add_argument('--imageSize', type=int, default=4096, help='the height / width of the input image to network')
    parser.add_argument('--nsy',type=int,default=1,help='number of synthetics [1]')
    parser.add_argument('--cutoff', type=float, default=1., help='cutoff frequency')
    options = parser.parse_args().__dict__


    return options

options = ParseOptions()

def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

md = {'dtm':0.01,'cutoff':options["cutoff"],'ntm':options["imageSize"]}
md['vTn'] = np.arange(0.0,3.05,0.05,dtype=np.float64)
md['nTn'] = md['vTn'].size

# vtm = md['dtm']*np.arange(0,md['ntm'])
# tar = np.zeros((options["nsy"],2))
# #acc  = -999.9*np.ones(shape=(options["nsy"],3,md['ntm']))
# acc  = np.zeros(shape=(options["nsy"],3,md['ntm']*2))

# # parse hdf5 database
# eqd = h5py.File('/gpfs/workdir/invsem07/STEAD/waveforms_11_13_19.hdf5','r')['earthquake']['local']
# eqm = pd.read_csv(osj(options["dataroot"],'metadata_11_13_19.csv'))
# eqm = eqm.loc[eqm['trace_category'] == 'earthquake_local']

# eqm = eqm.loc[(eqm['source_magnitude'] <= 5.0) & (eqm['source_magnitude'] >= 3.5)]
# eqm = eqm.sample(frac=options["nsy"]/len(eqm)).reset_index(drop=True)

# #w = windows.tukey(md['ntm'],5/100)

# for i in eqm.index:
#         tn = eqm.loc[i,'trace_name']
#         bi = int(eqd[tn].attrs['p_arrival_sample'])-200
#         for j in range(3):
#             acc[i,j,:(eqd[tn].shape[0]-bi)] = detrend(eqd[tn][bi:eqd[tn].shape[0],j])
#             #acc[i,j,:] = detrend(eqd[tn][bi:bi+options["ntm"],j])*w
#             #for k in range(eqd[tn].shape[0]):
#             #    acc[i,j,k] = signal[k]

# signal = tf.convert_to_tensor(acc)
# n = tf.cast(signal,dtype=float)

# np.savetxt("./stead/vtm.csv", vtm, delimiter=",")


# signal = np.zeros((options["nsy"],options["ntm"]*2),dtype=np.float32)
# for i in range(options["nsy"]):
#     for j in range(options["ntm"]*2):
#         signal[i,j] = n[i,1,j] # {0,1,2} = {x,y,z}

# np.savetxt("./stead/signal_100.csv", signal, delimiter=",")

# for i in range(options["nsy"]):
#     hfg = plt.figure(figsize=(12,6),tight_layout=True)
#     hax = hfg.add_subplot(111)
#     hax.plot(signal[i,:], color='black')
#     plt.savefig('./stead/signal_{:>d}.png'.format(i),bbox_inches = 'tight')
#     plt.close()

def make_stream(dataset):
    '''
    input: hdf5 dataset
    output: obspy stream
    
    '''
    data = np.array(dataset)
              
    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type']+'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']
    
    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type']+'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']
    
    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type']+'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])
    
    return stream


csv_file        = '/gpfs/workdir/invsem07/STEAD/metadata_11_13_19.csv'
file_name       = '/gpfs/workdir/invsem07/STEAD/waveforms_11_13_19.hdf5'


def data_selector(csv_file,file_name):

    # reading the csv file into a dataframe:
    df = pd.read_csv(csv_file)
    #print(f'total events in csv file: {len(df)}')
    # filterering the dataframe
    df = df[(df.trace_category == 'earthquake_local') & (7.0 <= df.source_magnitude)] #(7.0 <= df.source_magnitude) (df.network_code == 'TA')
    df = df.sample(frac=options["nsy"]/len(df)).reset_index(drop=True)
    #print(f'total events selected: {len(df)}')

    # making a list of trace names for the selected data
    ev_list = df['trace_name'].to_list()

    # retrieving selected waveforms from the hdf5 file: 
    dtfl = h5py.File(file_name, 'r')
    for c, evi in enumerate(ev_list):
        dataset = dtfl.get('earthquake/local/'+str(evi)) 
        # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel 
        data = np.array(dataset)

        # convert waveforms part ##################################################################################################################
        # downloading the instrument response of the station from IRIS
        client = Client("IRIS")
        inventory = client.get_stations(network=dataset.attrs['network_code'],
            station=dataset.attrs['receiver_code'],starttime=UTCDateTime(dataset.attrs['trace_start_time']),
            endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 60,loc="*",channel="*",level="response")
        #inventory[0].plot_response(min_freq=1E-4)
                
        # converting into acceleration
        st = make_stream(dataset)
        st.remove_response(inventory=inventory, output="ACC", taper=True, taper_fraction=0.05, plot=False)
        
        acc_for_loading_0 = st[0].data
        acc_for_loading_0 = np.float32(acc_for_loading_0)

        acc_for_loading_1 = st[1].data
        acc_for_loading_1 = np.float32(acc_for_loading_1)

        acc_for_loading_2 = st[2].data
        acc_for_loading_2 = np.float32(acc_for_loading_2)

        acc_for_loading_0  = np.expand_dims(acc_for_loading_0, axis=1)
        acc_for_loading_1  = np.expand_dims(acc_for_loading_1, axis=1)
        acc_for_loading_2  = np.expand_dims(acc_for_loading_2, axis=1)

        if c == 0:
            acc_0 = acc_for_loading_0
            acc_1 = acc_for_loading_1
            acc_2 = acc_for_loading_2
        else:
            acc_0  = np.concatenate((acc_0, acc_for_loading_0), axis=1)
            acc_1  = np.concatenate((acc_1, acc_for_loading_1), axis=1)
            acc_2  = np.concatenate((acc_2, acc_for_loading_2), axis=1)

        # end convert waveforms part ##############################################################################################################
    
    return acc_0,acc_1,acc_2,df

time = np.zeros((options["ntm"]*2,1),dtype=np.float32)
for i in range(1,options["ntm"]*2):
    time[i,0]=i*options["dtm"]
acc_0,acc_1,acc_2,df = data_selector(csv_file,file_name)

#acc_0 = np.swapaxes(acc_0,0,1)

# signal = np.zeros((options["nsy"],options["ntm"]*2),dtype=np.float32)
# for i in range(options["nsy"]):
#     for j in range(acc_0.shape[1]):
#         signal[i,j] = acc_0[i,j]

# signal = np.zeros((options["nsy"],options["ntm"]*2),dtype=np.float32)
# for i in range(acc_0.shape[1]):
#     signal[:,i] = acc_0[:,i]

# signal = np.transpose(signal)
# Dataset per Salome-Meca
signal = np.zeros((options["ntm"]*2,options["nsy"]+1),dtype=np.float32)
for j in range(options["nsy"]):
    signal[:,0] = time[:,0]
    for i in range(acc_0.shape[0]):
        signal[i,j+1] = acc_0[i,j]


np.savetxt("./acc_x.txt", signal, delimiter=",")

signal = np.zeros((options["ntm"]*2,options["nsy"]+1),dtype=np.float32)
for j in range(options["nsy"]):
    signal[:,0] = time[:,0]
    for i in range(acc_1.shape[0]):
        signal[i,j+1] = acc_1[i,j]


np.savetxt("./acc_y.txt", signal, delimiter=",")

signal = np.zeros((options["ntm"]*2,options["nsy"]+1),dtype=np.float32)
for j in range(options["nsy"]):
    signal[:,0] = time[:,0]
    for i in range(acc_2.shape[0]):
        signal[i,j+1] = acc_2[i,j]


np.savetxt("./acc_z.txt", signal, delimiter=",")

# acc_1 = np.swapaxes(acc_1,0,1)

# signal = np.zeros((options["nsy"],options["ntm"]*2),dtype=np.float32)
# for i in range(acc_1.shape[1]):
#     signal[:,i] = acc_1[:,i]


# np.savetxt("./Salome_y.txt", signal, delimiter=",")

# acc_2 = np.swapaxes(acc_2,0,1)

# signal = np.zeros((options["nsy"],options["ntm"]*2),dtype=np.float32)
# for i in range(acc_2.shape[1]):
#     signal[:,i] = acc_2[:,i]

# signal = np.transpose(signal)

# np.savetxt("./Salome_z.txt", signal, delimiter=",")

# vtm = md['dtm']*np.arange(0,md['ntm']*2)
# np.savetxt("./stead_1000/vtm.csv", vtm, delimiter=",")


# t = np.zeros(signal.shape[1])
# for k in range(signal.shape[1]-1):
#     t[k+1] = (k+1)*0.01

# for i in range(signal.shape[0]):
#     hfg = plt.figure(figsize=(12,6),tight_layout=True)
#     hax = hfg.add_subplot(111)
#     hax.plot(t,signal[i,:], color='black')
#     hax.set_ylabel(r'$Acceleration \hspace{0.5} [m / s^{2}]$', fontsize=26,fontweight='bold')
#     hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
#     hax.tick_params(axis='both', labelsize=20)
#     hax.yaxis.offsetText.set_fontsize(20)
#     plt.savefig('./acceleration/signal_{:>d}.png'.format(i),bbox_inches = 'tight')
#     plt.close()

# pga = np.zeros(signal.shape[0])

# for i in range(signal.shape[0]):
#         pga[i] = np.max(np.absolute(signal[i,:]))

# d = {r"$PGA$": pga}
# df = pd.DataFrame(data=d)
# sn.displot(data=d,kind="kde",legend=False)


# plt.xticks(fontsize=16,rotation=-45)
# plt.yticks(fontsize=16)
# #plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# plt.xlim(-0.02, 0.08)
# #plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# #plt.rc('font', size=16)
# plt.xlabel(r"$Peak \hspace{0.5} Ground \hspace{0.5} Acceleration \hspace{0.5} [m / s^{2}]$", fontsize=16)
# plt.ylabel(r"$Probability \hspace{0.5} Density \hspace{0.5} Function \hspace{0.5} [1]$", fontsize=16)
# plt.savefig('./acceleration/pga.png',bbox_inches = 'tight')
# #plt.savefig('./shear_building/pgd_undamaged.eps',bbox_inches = 'tight',dpi=200)
# plt.close()

   





