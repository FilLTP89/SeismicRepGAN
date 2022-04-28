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
import argparse

import openturns as ot
import openturns.viewer as viewer
import pickle
import control

import scipy
from scipy.fft import rfft, rfftfreq, irfft
from scipy.fft import fft, fftfreq, ifft

from scipy.signal import filtfilt, butter, lfilter
from mtcross_2 import MTCross
import utils_2 as utils

def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=2000,help='Number of epochs')
    parser.add_argument("--Xsize",type=int,default=2048,help='Data space size')
    parser.add_argument("--nX",type=int,default=4000,help='Number of signals')
    parser.add_argument("--signal",type=int,default=2,help="Types of signals")
    parser.add_argument("--N",type=int,default=2,help="Number of experiments")
    parser.add_argument("--nXchannels",type=int,default=4,help="Number of data channels")
    parser.add_argument("--nAElayers",type=int,default=3,help='Number of AE CNN layers')
    parser.add_argument("--nDlayers",type=int,default=10,help='Number of D CNN layers')
    parser.add_argument("--kernel",type=int,default=3,help='CNN kernel size')
    parser.add_argument("--stride",type=int,default=2,help='CNN stride')
    parser.add_argument("--nZfirst",type=int,default=8,help="Initial number of channels")
    parser.add_argument("--branching",type=str,default='conv',help='conv or dens')
    parser.add_argument("--latentSdim",type=int,default=2,help="Latent space s dimension")
    parser.add_argument("--latentCdim",type=int,default=2,help="Number of classes")
    parser.add_argument("--latentNdim",type=int,default=2,help="Latent space n dimension")
    parser.add_argument("--nSlayers",type=int,default=3,help='Number of S-branch CNN layers')
    parser.add_argument("--nClayers",type=int,default=3,help='Number of C-branch CNN layers')
    parser.add_argument("--nNlayers",type=int,default=3,help='Number of N-branch CNN layers')
    parser.add_argument("--Skernel",type=int,default=3,help='CNN kernel of S-branch branch')
    parser.add_argument("--Ckernel",type=int,default=3,help='CNN kernel of C-branch branch')
    parser.add_argument("--Nkernel",type=int,default=3,help='CNN kernel of N-branch branch')
    parser.add_argument("--Sstride",type=int,default=2,help='CNN stride of S-branch branch')
    parser.add_argument("--Cstride",type=int,default=2,help='CNN stride of C-branch branch')
    parser.add_argument("--Nstride",type=int,default=2,help='CNN stride of N-branch branch')
    parser.add_argument("--batchSize",type=int,default=50,help='input batch size')    
    parser.add_argument("--nCritic",type=int,default=1,help='number of discriminator training steps')
    parser.add_argument("--nGenerator",type=int,default=5,help='number of generator training steps')
    parser.add_argument("--clipValue",type=float,default=0.01,help='clip weight for WGAN')
    parser.add_argument("--dataroot", nargs="+", default=["/gpfs/workdir/invsem07/GiorgiaGAN/PortiqueElasPlas_N_2000_index",
                        "/gpfs/workdir/invsem07/GiorgiaGAN/PortiqueElasPlas_E_2000_index"],help="Data root folder") 
    # parser.add_argument("--dataroot", nargs="+", default=["/gpfs/workdir/invsem07/stead_1_9U","/gpfs/workdir/invsem07/stead_1_9D",
    #                     "/gpfs/workdir/invsem07/stead_1_10D"],help="Data root folder") 
    parser.add_argument("--idChannels",type=int,nargs='+',default=[1,2,3,4],help="Channel 1")
    parser.add_argument("--nParams",type=str,default=2,help="Number of parameters")
    parser.add_argument("--case",type=str,default="train_model",help="case")
    parser.add_argument("--avu",type=str,nargs='+',default="U",help="case avu")
    parser.add_argument("--pb",type=str,default="DC",help="case pb")#DC
    parser.add_argument("--CreateData",action='store_true',default=True,help='Create data flag')
    parser.add_argument("--cuda",action='store_true',default=False,help='Use cuda powered GPU')
    parser.add_argument('--dtm',type=float,default=0.04,help='time-step [s]')
    options = parser.parse_args().__dict__

    # assert options['nSchannels'] >= 1
    # assert options['Ssize'] >= options['Zsize']//(options['stride']**options['nSlayers'])

    return options

def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def cross_2d(current,ref,dt,nw,kspec,fmin,fmax,tmin,tmax):
    
    fnyq = 0.5/dt
    wn   = [fmin/fnyq,fmax/fnyq]
    b, a = butter(6, wn,'bandpass',output='ba')

    Pxy  = MTCross(current,ref,nw,kspec,dt,iadapt=2,wl=0.05)
    xcorr, dcohe, dconv  = Pxy.mt_corr()
    dconv = filtfilt(b, a, dcohe[:,0])
    k    = np.linspace(-Pxy.npts,Pxy.npts,len(xcorr),dtype=int)
    t2   = k*dt
    tloc = np.where((t2>=tmin) & (t2<=tmax))[0]
    irf  = np.zeros(len(tloc))
    irf = dconv[tloc]
    t = t2[tloc]
    
    return [irf,t]

def CreateData(**kwargs):
    CreateData.__globals__.update(kwargs)

    data = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    disp = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)

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
                data[i1,channel-1,i2] = np.array([row])
                i2 = i2+1
                if i2 == Xsize:
                    i2 = 0
                    i1 = i1+1

    for i in range(signal):
        
        for channel in idChannels:
            #load the measurements recorded by each single channel
            dataSrc = open(os.path.join(dataroot[i],"Disp_{:>d}.csv".format(channel)))
            file = csv.reader(dataSrc)


            i1 = int(nX/signal)*i
            i2 = 0
            for row in file:
                disp[i1,channel-1,i2] = np.array([row])
                i2 = i2+1
                if i2 == Xsize:
                    i2 = 0
                    i1 = i1+1

    labels = ['1']
    for i in range(2,data.shape[1]+1):
        labels.append('%d'% i)

    load = np.loadtxt('/gpfs/workdir/invsem07/GiorgiaGAN/acc_x.txt')
    load1 = np.loadtxt('/gpfs/workdir/invsem07/GiorgiaGAN/NDOF_code/noise_x.txt')
    acc = np.zeros((int(load.shape[0]/4),load.shape[1]-1))
    acc1 = np.zeros((int(load1.shape[0]/4),load.shape[1]-1))
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            acc[i,j] = load[i*4,j+1]
            acc1[i,j] = load1[i*4,j+1]
    
    nw      = 3.5
    kspec   = 5 
    fmin    = 1.0
    fmax    = 10.0
    tmin    = -2.0
    tmax    = +2.0

    St_a = np.zeros((nX,nXchannels,int((-tmin+tmax)/dtm+1)))
    St_d = np.zeros((nX,nXchannels,int((-tmin+tmax)/dtm+1)))

    # for k in range(N):
    #     for i in range(int(nX/signal/N)):
    #         for j in range(nXchannels):
    #             St[i+int(nX/signal/N)*k,j,:],tirf = cross_2d(data[i+int(nX/signal/N)*k,j,:],acc1[:,i],dtm,nw,kspec,fmin,fmax,tmin,tmax)
    #             St[i+int(nX/signal/N)*k+int(nX/signal),j,:],tirf = cross_2d(data[i+int(nX/signal/N)*k+int(nX/signal),j,:],acc[:,i],dtm,nw,kspec,fmin,fmax,tmin,tmax)

    # Sw_a = rfft(St_a)
    # Sw_d = rfft(St_d)

    # fig, axs = plt.subplots(data.shape[1], 2, sharex=False, figsize=(24,24))
    # fig.subplots_adjust(hspace=0)
    # for k in range(data.shape[1]-1,-1,-1):
    #     axs[data.shape[1]-1-k,0].plot(tirf, St_a[0,k,:],color='darkblue')
    #     axs[data.shape[1]-1-k,0].set_yticks([St_a[0,k,0]])
    #     axs[data.shape[1]-1-k,0].set_yticklabels([labels[k]],fontsize=20)
    #     axs[data.shape[1]-1-k,1].semilogx(Sw_a[0,k,:],color='firebrick')
    #     axs[data.shape[1]-1-k,1].set_yticks([Sw_a[0,k,0]])
    #     axs[data.shape[1]-1-k,1].set_yticklabels([labels[k]],fontsize=20)
    #     if k==(data.shape[1]-1):
    #         for spine in ['bottom']:
    #             axs[data.shape[1]-1-k,0].spines[spine].set_visible(False)
    #             axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
    #         axs[data.shape[1]-1-k,0].set_title('Deconvolved signal - Time domain',fontsize=26)
    #         axs[data.shape[1]-1-k,1].set_title('Deconvolved signal - Frequency domain',fontsize=26)
    #     if k==(0):
    #         for spine in ['top']:
    #             axs[data.shape[1]-1-k,0].spines[spine].set_visible(False)
    #             axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
    #         axs[data.shape[1]-1-k,0].set_xlabel('Time [s]',fontsize=23)
    #         axs[data.shape[1]-1-k,0].tick_params(axis='x', labelsize=20)
    #         axs[data.shape[1]-1-k,1].set_xlabel('Frequency [Hz]',fontsize=23)
    #         axs[data.shape[1]-1-k,1].tick_params(axis='x', labelsize=20)
    #     if k!=0 and k!=(data.shape[1]-1):
    #         for spine in ['bottom','top']:
    #             axs[data.shape[1]-1-k,0].spines[spine].set_visible(False)
    #             axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
    # fig.text(0.08,0.5,'Floor level', ha='center', va='center', rotation='vertical',fontsize=23)
    # plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/Deconvolution_acc.png',bbox_inches = 'tight')
    # plt.close()

    # fig, axs = plt.subplots(data.shape[1], 2, sharex=False, figsize=(24,24))
    # fig.subplots_adjust(hspace=0)
    # for k in range(data.shape[1]-1,-1,-1):
    #     axs[data.shape[1]-1-k,0].plot(tirf, St_d[0,k,:],color='darkblue')
    #     axs[data.shape[1]-1-k,0].set_yticks([St_d[0,k,0]])
    #     axs[data.shape[1]-1-k,0].set_yticklabels([labels[k]],fontsize=20)
    #     axs[data.shape[1]-1-k,1].semilogx(Sw_d[0,k,:],color='firebrick')
    #     axs[data.shape[1]-1-k,1].set_yticks([Sw_d[0,k,0]])
    #     axs[data.shape[1]-1-k,1].set_yticklabels([labels[k]],fontsize=20)
    #     if k==(data.shape[1]-1):
    #         for spine in ['bottom']:
    #             axs[data.shape[1]-1-k,0].spines[spine].set_visible(False)
    #             axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
    #         axs[data.shape[1]-1-k,0].set_title('Deconvolved signal - Time domain',fontsize=26)
    #         axs[data.shape[1]-1-k,1].set_title('Deconvolved signal - Frequency domain',fontsize=26)
    #     if k==(0):
    #         for spine in ['top']:
    #             axs[data.shape[1]-1-k,0].spines[spine].set_visible(False)
    #             axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
    #         axs[data.shape[1]-1-k,0].set_xlabel('Time [s]',fontsize=23)
    #         axs[data.shape[1]-1-k,0].tick_params(axis='x', labelsize=20)
    #         axs[data.shape[1]-1-k,1].set_xlabel('Frequency [Hz]',fontsize=23)
    #         axs[data.shape[1]-1-k,1].tick_params(axis='x', labelsize=20)
    #     if k!=0 and k!=(data.shape[1]-1):
    #         for spine in ['bottom','top']:
    #             axs[data.shape[1]-1-k,0].spines[spine].set_visible(False)
    #             axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
    # fig.text(0.08,0.5,'Floor level', ha='center', va='center', rotation='vertical',fontsize=23)
    # plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/Deconvolution_disp.png',bbox_inches = 'tight')
    # plt.close()
    
    t = np.zeros(data.shape[2])
    for k in range(data.shape[2]-1):
        t[k+1] = (k+1)*0.04

    u1 = np.zeros((nX,nXchannels,int(Xsize/2)+1),dtype=np.float32)
    u2 = np.zeros((nX,nXchannels,int(Xsize/2)+1),dtype=np.float32)
    u3 = np.zeros((nX,nXchannels,int(Xsize/2)+1),dtype=np.float32)
    
    xf = rfftfreq(Xsize,dtm)[:Xsize//2]

    for k in range(N):
        for i in range(int(nX/signal/N)):
            for j in range(nXchannels):
                u1[i+int(nX/signal/N)*k,j,:] = rfft(data[i+int(nX/signal/N)*k,j,:])
                u2[i+int(nX/signal/N)*k,j,:] = rfft(acc1[:,i])
                u3[i+int(nX/signal/N)*k,j,:] = rfft(disp[i+int(nX/signal/N)*k,j,:])
                u1[i+int(nX/signal/N)*k+int(nX/signal),j,:] = rfft(data[i+int(nX/signal/N)*k+int(nX/signal),j,:])
                u2[i+int(nX/signal/N)*k+int(nX/signal),j,:] = rfft(acc[:,i])
                u3[i+int(nX/signal/N)*k+int(nX/signal),j,:] = rfft(disp[i+int(nX/signal/N)*k+int(nX/signal),j,:])
                
   
    Xa = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    Xd = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    Xu1 = np.zeros((nX,nXchannels,int(Xsize/2)+1),dtype=np.float32)
    Xu3 = np.zeros((nX,nXchannels,int(Xsize/2)+1),dtype=np.float32)

    
    for i in range(data.shape[0]):
        pga = np.zeros((nXchannels))
        pgd = np.zeros((nXchannels))
        pga1 = np.zeros((nXchannels))
        pgd1 = np.zeros((nXchannels))
        for n in range(nXchannels):
            pga[n] = np.max(np.absolute(data[i,n,:]))
            pgd[n] = np.max(np.absolute(disp[i,n,:]))
            pga1[n] = np.max(np.absolute(u1[i,n,:]))
            pgd1[n] = np.max(np.absolute(u3[i,n,:]))
        pga_value = np.max(pga)
        pgd_value = np.max(pgd)
        pga_value1 = np.max(pga1)
        pgd_value1 = np.max(pgd1)
        if pga_value==0:
            print('pga',i)
        if pgd_value==0:
            print('pgd',i)
        Xa[i,:,:] = data[i,:,:]/pga_value
        Xd[i,:,:] = disp[i,:,:]/pgd_value
        Xu1[i,:,:] = u1[i,:,:]/pga_value1
        Xu3[i,:,:] = u3[i,:,:]/pgd_value1


    for i in range(1):
        fig, axs = plt.subplots(data.shape[1], 2, sharex=False, figsize=(24,24))
        fig.subplots_adjust(hspace=0)
        for k in range(data.shape[1]-1,-1,-1):
            axs[data.shape[1]-1-k,0].plot(t, Xa[i,k,:],color='darkblue')
            axs[data.shape[1]-1-k,0].set_yticks([Xa[i,k,0]])
            axs[data.shape[1]-1-k,0].set_yticklabels([labels[k]],fontsize=20)
            axs[data.shape[1]-1-k,1].semilogx(Xu1[i,k,:],color='firebrick')
            axs[data.shape[1]-1-k,1].set_yticks([Xu1[i,k,0]])
            axs[data.shape[1]-1-k,1].set_yticklabels([labels[k]],fontsize=20)
            if k==(data.shape[1]-1):
                for spine in ['bottom']:
                    axs[data.shape[1]-1-k,0].spines[spine].set_visible(False)
                    axs[data.shape[1]-1-k,0].legend([r'$[\frac{m}{s^2}]$'],frameon=False,loc='upper right',fontsize=20)
                    axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
                axs[data.shape[1]-1-k,0].set_title('Structural response - Time domain',fontsize=26)
                axs[data.shape[1]-1-k,1].set_title('Structural response - Frequency domain',fontsize=26)
            if k==(0):
                for spine in ['top']:
                    axs[data.shape[1]-1-k,0].spines[spine].set_visible(False)
                    axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
                axs[data.shape[1]-1-k,0].set_xlabel('Time [s]',fontsize=23)
                axs[data.shape[1]-1-k,0].tick_params(axis='x', labelsize=20)
                axs[data.shape[1]-1-k,1].set_xlabel('Frequency [Hz]',fontsize=23)
                axs[data.shape[1]-1-k,1].tick_params(axis='x', labelsize=20)
            if k!=0 and k!=(data.shape[1]-1):
                for spine in ['bottom','top']:
                    axs[data.shape[1]-1-k,0].spines[spine].set_visible(False)
                    axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
        fig.text(0.08,0.5,'Floor level', ha='center', va='center', rotation='vertical',fontsize=23)
        plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/Acceleration_{:>d}.png'.format(i),bbox_inches = 'tight')
        plt.close()

    for i in range(1):
        fig, axs = plt.subplots(data.shape[1], 2, sharex=False, figsize=(24,24))
        fig.subplots_adjust(hspace=0)
        for k in range(data.shape[1]-1,-1,-1):
            axs[data.shape[1]-1-k,0].plot(t, Xd[i,k,:],color='darkblue')
            axs[data.shape[1]-1-k,0].set_yticks([Xd[i,k,0]])
            axs[data.shape[1]-1-k,0].set_yticklabels([labels[k]],fontsize=23)
            axs[data.shape[1]-1-k,1].set_ylim([-np.abs(np.max(Xd[i,k,:])), np.abs(np.max(Xd[i,k,:]))])

            axs[data.shape[1]-1-k,1].plot(xf,2.0/Xsize*np.abs(Xu3[i,k,:Xsize//2]),color='firebrick')
            axs[data.shape[1]-1-k,1].set_yticks([2.0/Xsize*np.abs(Xu3[i,k,0])])
            axs[data.shape[1]-1-k,1].set_yticklabels([labels[k]],fontsize=23)
            axs[data.shape[1]-1-k,1].set_ylim([-np.max(2.0/Xsize*np.abs(Xu3[i,k,:])), np.max(2.0/Xsize*np.abs(Xu3[i,k,:]))])

            if k==(data.shape[1]-1):
                for spine in ['bottom']:
                    axs[data.shape[1]-1-k,0].spines[spine].set_visible(False)
                    axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
                    axs[data.shape[1]-1-k,0].legend([r'$[m]$'],frameon=False,loc='upper right',fontsize=20)
                axs[data.shape[1]-1-k,0].set_title('Structural response - Time domain',fontsize=26)
                axs[data.shape[1]-1-k,1].set_title('Structural response - Frequency domain',fontsize=26)
            if k==(0):
                for spine in ['top']:
                    axs[data.shape[1]-1-k,0].spines[spine].set_visible(False)
                    axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
                axs[data.shape[1]-1-k,0].set_xlabel('Time [s]',fontsize=20)
                axs[data.shape[1]-1-k,0].tick_params(axis='x', labelsize=20)
                axs[data.shape[1]-1-k,1].set_xlabel('Frequency [Hz]',fontsize=20)
                axs[data.shape[1]-1-k,1].tick_params(axis='x', labelsize=20)
            if k!=0 and k!=(data.shape[1]-1):
                for spine in ['bottom','top']:
                    axs[data.shape[1]-1-k,0].spines[spine].set_visible(False) 
                    axs[data.shape[1]-1-k,1].spines[spine].set_visible(False)
        fig.text(0.08,0.5,'Floor level', ha='center', va='center', rotation='vertical',fontsize=23)
        plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/Displacement_{:>d}.png'.format(i),bbox_inches = 'tight')
        plt.close()

    #irf = np.zeros((nX,nXchannels,Xsize),dtype=np.float32)
    
    # for k in range(N):
    #     for i in range(int(nX/signal/N)):
    #         for j in range(nXchannels):
    #             irf[i+int(nX/signal/N)*k,j,:] = np.fft.irfft(np.fft.rfft(data[i+int(nX/signal/N)*k,j,:])/np.fft.rfft(acc1[:,i]+0.0001*np.mean(acc1[:,i])))
    #             irf[i+int(nX/signal/N)*k+int(nX/signal),j,:] = np.fft.irfft(np.fft.rfft(data[i+int(nX/signal/N)*k+int(nX/signal),j,:])/np.fft.rfft(acc[:,i]+0.0001*np.mean(acc[:,i])))
    
    

    
    # for j in range(data.shape[1]):
    #     for k in range(10):
    #         i = randint(0, int(data.shape[0]/2))
    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(t,X[i,j,:], color='black')
    #         #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/noise_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         N = data.shape[2]
    #         SAMPLE_RATE = 25
    #         yf_real = rfft(X[i,j,:])
    #         xf_real = rfftfreq(N, 1 / SAMPLE_RATE)
    #         hax.semilogy(xf_real, np.abs(yf_real), color='black')
    #         hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/fft_noise_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()

    #         i = randint(int(data.shape[0]/2),data.shape[0]-1)
    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(t,X[i,j,:], color='black')
    #         #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/earthquake_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         N = data.shape[2]
    #         SAMPLE_RATE = 25
    #         yf_real = rfft(X[i,j,:])
    #         xf_real = rfftfreq(N, 1 / SAMPLE_RATE)
    #         hax.semilogy(xf_real, np.abs(yf_real), color='black')
    #         hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/fft_earthquake_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()
    
    # X = np.pad(X,((0,0),(0,0),(0,nxtpow2(X.shape[-1])-X.shape[-1])))
    # X = np.swapaxes(X,1,2)
    

                
    # for i in range(X.shape[0]):
    #     pga = np.zeros((nXchannels))
    #     for n in range(nXchannels):
    #         pga[n] = np.max(np.absolute(X[i,n,:]))
    #     pga_value = np.max(pga)
    #     if pga_value==0:
    #         print(pga,i)
    # for j in range(irf.shape[1]):
    #     hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #     hax = hfg.add_subplot(111)
    #     hax.plot(tirf, (irf[0,j,:]), color='black')
    #     hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     hax.tick_params(axis='both', labelsize=18)
    #     plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/H_n_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #     plt.close()

    #     hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #     hax = hfg.add_subplot(111)
    #     hax.plot(tirf, (irf[1,j,:]), color='black')
    #     hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     hax.tick_params(axis='both', labelsize=18)
    #     plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/H_e_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #     plt.close()
    

    
    # for k in range(10):
    #     i = randint(0, 2000)
    #     i1 = randint(2000, 4000)
    #     for j in range(irf.shape[1]):
    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(tirf, (irf[i,j,:]), color='black')
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/H_n_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(t, (X[i,:,j]), color='black')
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/X_n_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(tirf, (irf[i1,j,:]), color='black')
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/H_e_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(t, (X[i1,:,j]), color='black')
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/figures/X_e_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.close()

    return

options = ParseOptions()

CreateData(**options)