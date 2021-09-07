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
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client



def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",type=str,default="/gpfs/workdir/invsem07/STEAD",help="Data root folder")
    parser.add_argument("--cuda",action='store_true',default=False,help='Use cuda powered GPU')
    parser.add_argument('--dtm',type=float,default=0.01,help='time-step [s]')
    parser.add_argument('--ntm',type=int,default=4096,help='Number of time steps')
    parser.add_argument('--imageSize', type=int, default=4096, help='the height / width of the input image to network')
    parser.add_argument('--nsy',type=int,default=256,help='number of synthetics [1]')
    parser.add_argument('--cutoff', type=float, default=1., help='cutoff frequency')
    options = parser.parse_args().__dict__


    return options

options = ParseOptions()

def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

# md = {'dtm':0.01,'cutoff':options["cutoff"],'ntm':options["imageSize"]}
# md['vTn'] = np.arange(0.0,3.05,0.05,dtype=np.float64)
# md['nTn'] = md['vTn'].size

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

# np.savetxt("/gpfs/workdir/invsem07/GiorgiaGAN/stead/vtm.csv", vtm, delimiter=",")


# signal = np.zeros((options["nsy"],options["ntm"]*2),dtype=np.float32)
# for i in range(options["nsy"]):
#     for j in range(options["ntm"]*2):
#         signal[i,j] = n[i,1,j] # {0,1,2} = {x,y,z}

# np.savetxt("/gpfs/workdir/invsem07/GiorgiaGAN/stead/signal_100.csv", signal, delimiter=",")

# for i in range(options["nsy"]):
#     hfg = plt.figure(figsize=(12,6),tight_layout=True)
#     hax = hfg.add_subplot(111)
#     hax.plot(signal[i,:], color='black')
#     plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/stead/signal_{:>d}.png'.format(i),bbox_inches = 'tight')
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
    df = df[(df.trace_category == 'earthquake_local') & (df.network_code == 'TA') & (df.source_magnitude <= 5.0) & (3.5 <= df.source_magnitude)]
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
        acc_for_loading_1 = st[2].data
        acc_for_loading_1 = np.float32(acc_for_loading_1)

        acc_for_loading_1  = np.expand_dims(acc_for_loading_1, axis=1)

        if c == 0:
            acc_for_loading  = acc_for_loading_1
        else:
            acc_for_loading  = np.concatenate((acc_for_loading, acc_for_loading_1), axis=1)

        # end convert waveforms part ##############################################################################################################
    
    return acc_for_loading, df

acc, df = data_selector(csv_file,file_name)
acc = np.swapaxes(acc,0,1)

signal = np.zeros((options["nsy"],options["ntm"]*2),dtype=np.float32)
for i in range(acc.shape[1]):
    signal[:,i] = acc[:,i]

np.savetxt("/gpfs/workdir/invsem07/GiorgiaGAN/stead_256/signal_256.csv", signal, delimiter=",")

for i in range(options["nsy"]):
    hfg = plt.figure(figsize=(12,6),tight_layout=True)
    hax = hfg.add_subplot(111)
    hax.plot(signal[i,:], color='black')
    hax.set_title('Ground motion acceleration', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$[m / s^{2}]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$time \hspace{0.5} [s]$', fontsize=20,fontweight='bold')
    hax.tick_params(axis='both', labelsize=14)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/stead_256/signal_{:>d}.png'.format(i),bbox_inches = 'tight')
    plt.close()

   





