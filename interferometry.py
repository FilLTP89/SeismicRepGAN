from locale import D_T_FMT
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from os.path import join as opj

#from mtspec import dpss
import matplotlib.pyplot as plt

from scipy.signal import windows, detrend, filtfilt, butter, lfilter, resample
from scipy import interpolate
from mtcross_2 import MTCross
from interferometry_utils import *
from random import randint


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def cross_2d(data,i0,dtm,nw,kspec,fmin,fmax,tmin,tmax):
    
    fnyq = 0.5/dtm
    wn   = [fmin/fnyq,fmax/fnyq]
    b, a = butter(6, wn,'bandpass',output='ba')

    ntr = data.shape[1] 
    x   = data[:,i0]
    for i in range(ntr):
        y = data[:,i]
        Pxy  = MTCross(y,x,nw,kspec,dtm,iadapt=2,wl=0.0)
        xcorr, dcohe, dconv  = Pxy.mt_corr()
        dconv = filtfilt(b, a, dcohe[:,0])
        if (i==0):
            k    = np.linspace(-Pxy.npts,Pxy.npts,len(xcorr),dtype=int)
            t2   = k*dtm
            tloc = np.where((t2>=tmin) & (t2<=tmax))[0]
            irf  = np.zeros((len(tloc),ntr))
        irf[:,i] = dconv[tloc]
        t        = t2[tloc]
    
    return [irf,t]

def cross_2d_dam(und,dam,i0,dtm,nw,kspec,fmin,fmax,tmin,tmax):
    
    fnyq = 0.5/dtm
    wn   = [fmin/fnyq,fmax/fnyq]
    b, a = butter(6, wn,'bandpass',output='ba')

    ntr = und.shape[1] 
    x   = und[:,i0]
    for i in range(ntr):
        y = dam[:,i]
        Pxy  = MTCross(y,x,nw,kspec,dtm,iadapt=2,wl=0.0)
        xcorr, dcohe, dconv  = Pxy.mt_corr()
        dconv = filtfilt(b, a, dcohe[:,0])
        if (i==0):
            k    = np.linspace(-Pxy.npts,Pxy.npts,len(xcorr),dtype=int)
            t2   = k*dtm
            tloc = np.where((t2>=tmin) & (t2<=tmax))[0]
            irf  = np.zeros((len(tloc),ntr))
        irf[:,i] = dconv[tloc]
        t        = t2[tloc]
    
    return [irf,t]

def Stretching_current(ref, cur, t, dvmin, dvmax, nbtrial, window, fmin, fmax, tmin, tmax):
    """
    Stretching function: 
    This function compares the reference waveform to stretched/compressed current waveform to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the reference waveform and the current waveform.
    INPUTS:
        - ref = Reference waveform (np.ndarray, size N)
        - cur = Current waveform (np.ndarray, size N)
        - t = time vector, common to both ref and cur (np.ndarray, size N)
        - dvmin = minimum bound for the velocity variation; example: dvmin=-0.03 for -3% of relative velocity change ('float')
        - dvmax = maximum bound for the velocity variation; example: dvmax=0.03 for 3% of relative velocity change ('float')
        - nbtrial = number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
        - window = vector of the indices of the cur and ref windows on which you want to do the measurements (np.ndarray, size tmin*delta:tmax*delta)
        For error computation:
            - fmin = minimum frequency of the data
            - fmax = maximum frequency of the data
            - tmin = minimum time window where the dv/v is computed 
            - tmax = maximum time window where the dv/v is computed 
    OUTPUTS:
        - dv = Relative velocity change dv/v (in %)
        - cc = correlation coefficient between the reference waveform and the best stretched/compressed current waveform
        - cdp = correlation coefficient between the reference waveform and the initial current waveform
        - Eps = Vector of Epsilon values (Epsilon =-dtm/t = dv/v)
        - error = Errors in the dv/v measurements based on Weaver, R., C. Hadziioannou, E. Larose, and M. Camnpillo (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3), 1384?1392
        - C = vector of the correlation coefficient between the reference waveform and every stretched/compressed current waveforms
    The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "nbtrial" values. 
    A refined analysis is then performed around this value to obtain a more precise dv/v measurement .
    """ 
    Eps = np.asmatrix(np.linspace(dvmin, dvmax, nbtrial))
    L = 1 + Eps
    tt = np.matrix.transpose(np.asmatrix(t))
    tau = tt.dot(L)  # stretched/compressed time axis
    C = np.zeros((1, np.shape(Eps)[1]))

    # Set of stretched/compressed current waveforms
    for j in np.arange(np.shape(Eps)[1]):
        s = np.interp(x=np.ravel(tt), xp=np.ravel(tau[:, j]), fp=cur)
        waveform_ref = ref[window]
        waveform_cur = s[window]
        C[0, j] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]
    
    cdp = np.corrcoef(cur[window], ref[window])[0, 1] # correlation coefficient between the reference and initial current waveforms

    # find the maximum correlation coefficient
    imax = np.nanargmax(C)
    if imax >= np.shape(Eps)[1]-1:
        imax = imax - 2
    if imax <= 2:
        imax = imax + 2
    # Proceed to the second step to get a more precise dv/v measurement
    dtmfiner = np.linspace(Eps[0, imax-2], Eps[0, imax+1], 500)
    func = interpolate.interp1d(np.ravel(Eps[0, np.arange(imax-3, imax+2)]), np.ravel(C[0,np.arange(imax-3, imax+2)]), kind='cubic')
    CCfiner = func(dtmfiner)

    cc = np.max(CCfiner) # Find maximum correlation coefficient of the refined  analysis
    dv = 100. * dtmfiner[np.argmax(CCfiner)] # Multiply by 100 to convert to percentage (Epsilon = -dtm/t = dv/v)

    # Error computation based on Weaver, R., C. Hadziioannou, E. Larose, and M. Camnpillo (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3), 1384-1392
    T = 1 / (fmax - fmin)
    X = cc
    wc = np.pi * (fmin + fmax)
    t1 = np.min([tmin, tmax])
    t2 = np.max([tmin, tmax])
    error = 100*(np.sqrt(1-X**2)/(2*X)*np.sqrt((6* np.sqrt(np.pi/2)*T)/(wc**2*(t2**3-t1**3))))


    return dv, cc, error, cdp



def deconvolution(realX_u,realX_d,fakeX_new,d):

    nw      = 3.5
    kspec   = 5 
    fmin    = 0.5
    fmax    = 10
    tmin    = -1.0
    tmax    = +1.0
    dtm     = 0.04

    
    #IRF_mean_real = {}
    #IRF_mean_switch = {}
    IRF_mean_u = {}
    IRF_mean_s = {}
    #IRF_real = {}
    #IRF_switch = {}
    IRF_u = {}
    IRF_s = {}
    for i in range(realX_u.shape[2]): #IRF['IRF_%d' % i] = np.zeros((5,realX_u.shape[1],realX_u.shape[2]))
        #IRF_mean_real['IRF_mean_real_%d' % i] = np.zeros((int((-tmin+tmax)/dtm+1),realX_u.shape[2]))
        #IRF_mean_switch['IRF_mean_switch_%d' % i] = np.zeros((int((-tmin+tmax)/dtm+1),realX_u.shape[2]))
        #IRF_real['IRF_real_%d' % i] = np.zeros((realX_u.shape[0],int((-tmin+tmax)/dtm+1),realX_u.shape[2]))
        #IRF_switch['IRF_switch_%d' % i] = np.zeros((realX_u.shape[0],int((-tmin+tmax)/dtm+1),realX_u.shape[2]))
        IRF_mean_u['IRF_mean_u_%d' % i] = np.zeros((int((-tmin+tmax)/dtm+1),realX_u.shape[2]))
        IRF_mean_s['IRF_mean_s_%d' % i] = np.zeros((int((-tmin+tmax)/dtm+1),realX_u.shape[2]))
        IRF_u['IRF_u_%d' % i] = np.zeros((realX_u.shape[0],int((-tmin+tmax)/dtm+1),realX_u.shape[2]))
        IRF_s['IRF_s_%d' % i] = np.zeros((realX_u.shape[0],int((-tmin+tmax)/dtm+1),realX_u.shape[2]))

    
    for i in range(realX_u.shape[2]):
        for j in range(realX_u.shape[0]):#realX_u.shape[0]
            #IRF_real['IRF_real_%d' % i][j,:,:],tirf = cross_2d_dam(realX_u[j,:,:],realX_d[j,:,:],i,dtm,nw,kspec,fmin,fmax,tmin,tmax)
            #IRF_switch['IRF_switch_%d' % i][j,:,:],tirf = cross_2d_dam(realX_u[j,:,:],np.array(fakeX_new[j,:,:]),i,dtm,nw,kspec,fmin,fmax,tmin,tmax)
            IRF_u['IRF_u_%d' % i][j,:,:],tirf = cross_2d(realX_u[j,:,:],i,dtm,nw,kspec,fmin,fmax,tmin,tmax)
            IRF_s['IRF_s_%d' % i][j,:,:],tirf = cross_2d(np.array(fakeX_new[j,:,:]),i,dtm,nw,kspec,fmin,fmax,tmin,tmax)


    #Average IRF
    for i in range(realX_u.shape[2]):
        for k in range(realX_u.shape[2]):
            IRF_mean_u['IRF_mean_u_%d' % i][:,k] = np.mean(IRF_u['IRF_u_%d' % i][:,:,k],axis=0)
            IRF_mean_s['IRF_mean_s_%d' % i][:,k] = np.mean(IRF_s['IRF_s_%d' % i][:,:,k],axis=0)

    for i in range(realX_u.shape[2]):
        for k in range(realX_u.shape[2]):
            np.savetxt("/gpfs/workdir/colombergi/GiorgiaGAN/results/IRF_u{:>d}_floor{:>d}_{:>d}.csv".format(d,i,k), IRF_u['IRF_u_%d' % i][:,:,k], delimiter=",")
            np.savetxt("/gpfs/workdir/colombergi/GiorgiaGAN/results/IRF_s{:>d}_floor{:>d}_{:>d}.csv".format(d,i,k), IRF_s['IRF_s_%d' % i][:,:,k], delimiter=",")


    labels = ['1']
    for i in range(2,realX_u.shape[2]+1):
        labels.append('%d'% i)

    # for i in range(realX_u.shape[2]):
    #     fig, axs = plt.subplots(realX_u.shape[2], 1, sharex=True, figsize=(4,6))
    #     fig.subplots_adjust(hspace=0)
    #     for k in range(realX_u.shape[2]-1,-1,-1):
    #         for j in range(realX_u.shape[0]): #realX_u.shape[0]
    #             axs[realX_u.shape[2]-1-k].plot(tirf, IRF_real['IRF_real_%d' % i][j,:,k])
    #         axs[realX_u.shape[2]-1-k].plot(tirf, IRF_mean_real['IRF_mean_real_%d' % i][:,k],color='k')
    #         axs[realX_u.shape[2]-1-k].set_yticks([IRF_mean_real['IRF_mean_real_%d' % i][0,k]])
    #         axs[realX_u.shape[2]-1-k].set_yticklabels([labels[k]])
    #         axs[realX_u.shape[2]-1-k].set_xlim(tmin,tmax)
    #         if k==(realX_u.shape[2]-1):
    #             for spine in ['bottom']:
    #                 axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)
    #         if k==(0):
    #             for spine in ['top']:
    #                 axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)
    #         if k!=0 and k!=(realX_u.shape[2]-1):
    #             for spine in ['bottom','top']:
    #                 axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)     
    #     n = i+1
    #     fig.suptitle('IRF floor %d - Original signals' % n)
    #     fig.text(0.5,0.05,'Time [s]', ha='center', va='center')
    #     fig.text(0.05,0.5,'Floor level', ha='center', va='center', rotation='vertical')
    #     plt.savefig('/gpfs/workdir/colombergi/GiorgiaGAN/results/IRF{:>d} Floor_real_{:>d}.png'.format(d,i),bbox_inches = 'tight')
    #     plt.close()

    #     for i in range(realX_u.shape[2]):
    #         fig, axs = plt.subplots(realX_u.shape[2], 1, sharex=True, figsize=(4,6))
    #     fig.subplots_adjust(hspace=0)
    #     for k in range(realX_u.shape[2]-1,-1,-1):
    #         for j in range(realX_u.shape[0]): #realX_u.shape[0]
    #             axs[realX_u.shape[2]-1-k].plot(tirf, IRF_switch['IRF_switch_%d' % i][j,:,k])
    #         axs[realX_u.shape[2]-1-k].plot(tirf, IRF_mean_switch['IRF_mean_switch_%d' % i][:,k],color='k')
    #         axs[realX_u.shape[2]-1-k].set_yticks([IRF_mean_switch['IRF_mean_switch_%d' % i][0,k]])
    #         axs[realX_u.shape[2]-1-k].set_yticklabels([labels[k]])
    #         axs[realX_u.shape[2]-1-k].set_xlim(tmin,tmax)
    #         if k==(realX_u.shape[2]-1):
    #             for spine in ['bottom']:
    #                 axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)
    #         if k==(0):
    #             for spine in ['top']:
    #                 axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)
    #         if k!=0 and k!=(realX_u.shape[2]-1):
    #             for spine in ['bottom','top']:
    #                 axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)     
    #     n = i+1
    #     fig.suptitle('IRF floor %d - Switched signals' % n)
    #     fig.text(0.5,0.05,'Time [s]', ha='center', va='center')
    #     fig.text(0.05,0.5,'Floor level', ha='center', va='center', rotation='vertical')
    #     plt.savefig('/gpfs/workdir/colombergi/GiorgiaGAN/results/IRF{:>d} Floor_switch_{:>d}.png'.format(d,i),bbox_inches = 'tight')
    #     plt.close()
    
    for q in range(10):
        j = randint(0, realX_u.shape[0]-1)
        fig, axs = plt.subplots(realX_u.shape[2], 1, sharex=True, figsize=(4,6))
        fig.subplots_adjust(hspace=0)
        for k in range(realX_u.shape[2]-1,-1,-1):
            axs[realX_u.shape[2]-1-k].plot(tirf, IRF_u['IRF_u_%d' % 3][j,:,k])
            axs[realX_u.shape[2]-1-k].plot(tirf, IRF_mean_u['IRF_mean_u_%d' % 3][:,k],color='k')
            axs[realX_u.shape[2]-1-k].set_yticks([IRF_mean_u['IRF_mean_u_%d' % 3][0,k]])
            axs[realX_u.shape[2]-1-k].set_yticklabels([labels[k]])
            axs[realX_u.shape[2]-1-k].set_xlim(tmin,tmax)
            if k==(realX_u.shape[2]-1):
                for spine in ['bottom']:
                    axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)
            if k==(0):
                for spine in ['top']:
                    axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)
            if k!=0 and k!=(realX_u.shape[2]-1):
                for spine in ['bottom','top']:
                    axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)     
        fig.suptitle('IRF floor 4 - Original signals')
        fig.text(0.5,0.05,'Time [s]', ha='center', va='center')
        fig.text(0.05,0.5,'Floor level', ha='center', va='center', rotation='vertical')
        plt.savefig('/gpfs/workdir/colombergi/GiorgiaGAN/results/IRF{:>d}_undamaged_{:>d}.png'.format(d,j),bbox_inches = 'tight')
        plt.close()

        fig, axs = plt.subplots(realX_u.shape[2], 1, sharex=True, figsize=(4,6))
        fig.subplots_adjust(hspace=0)
        for k in range(realX_u.shape[2]-1,-1,-1):
            axs[realX_u.shape[2]-1-k].plot(tirf, IRF_s['IRF_s_%d' % 3][j,:,k])
            axs[realX_u.shape[2]-1-k].plot(tirf, IRF_mean_s['IRF_mean_s_%d' % 3][:,k],color='k')
            axs[realX_u.shape[2]-1-k].set_yticks([IRF_mean_s['IRF_mean_s_%d' % 3][0,k]])
            axs[realX_u.shape[2]-1-k].set_yticklabels([labels[k]])
            axs[realX_u.shape[2]-1-k].set_xlim(tmin,tmax)
            if k==(realX_u.shape[2]-1):
                for spine in ['bottom']:
                    axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)
            if k==(0):
                for spine in ['top']:
                    axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)
            if k!=0 and k!=(realX_u.shape[2]-1):
                for spine in ['bottom','top']:
                    axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)     
        fig.suptitle('IRF floor 4 - Switched signals')
        fig.text(0.5,0.05,'Time [s]', ha='center', va='center')
        fig.text(0.05,0.5,'Floor level', ha='center', va='center', rotation='vertical')
        plt.savefig('/gpfs/workdir/colombergi/GiorgiaGAN/results/IRF{:>d}_switched_{:>d}.png'.format(d,j),bbox_inches = 'tight')
        plt.close()

        fig, axs = plt.subplots(realX_u.shape[2], 1, sharex=True, figsize=(4,6))
        fig.subplots_adjust(hspace=0)
        for k in range(realX_u.shape[2]-1,-1,-1):
            axs[realX_u.shape[2]-1-k].plot(tirf, IRF_u['IRF_u_%d' % 3][j,:,k],color='r')
            axs[realX_u.shape[2]-1-k].plot(tirf, IRF_s['IRF_s_%d' % 3][j,:,k],color='b')
            axs[realX_u.shape[2]-1-k].set_yticks([IRF_mean_s['IRF_mean_s_%d' % 3][0,k]])
            axs[realX_u.shape[2]-1-k].set_yticklabels([labels[k]])
            axs[realX_u.shape[2]-1-k].set_xlim(tmin,tmax)
            if k==(realX_u.shape[2]-1):
                for spine in ['bottom']:
                    axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)
                    axs[realX_u.shape[2]-1-k].legend([r'$X_u$',r"$G_z(F_x(x_u))$"],frameon=False,loc='upper right')
            if k==(0):
                for spine in ['top']:
                    axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)
            if k!=0 and k!=(realX_u.shape[2]-1):
                for spine in ['bottom','top']:
                    axs[realX_u.shape[2]-1-k].spines[spine].set_visible(False)     
        fig.suptitle('IRF floor 4 - Comparison')
        #fig.legend([r'$Undamaged$', r"$ Reconstructed \hspace{0.5} Switched$"],frameon=False)
        fig.text(0.5,0.05,'Time [s]', ha='center', va='center')
        fig.text(0.05,0.5,'Floor level', ha='center', va='center', rotation='vertical')
        plt.savefig('/gpfs/workdir/colombergi/GiorgiaGAN/results/IRF{:>d}_comparison_{:>d}.png'.format(d,j),bbox_inches = 'tight')
        plt.close()

    

    dv = {}
    cc = {}
    error = {}
    cdp = {}

    t = np.linspace(tmin,tmax,int((-tmin+tmax)/dtm+1))


    for i in range(realX_u.shape[2]):
        dv['dv_%d' % i] = np.zeros((realX_u.shape[0],realX_u.shape[2]))
        cc['cc_%d' % i] = np.zeros((realX_u.shape[0],realX_u.shape[2]))
        error['error_%d' % i] = np.zeros((realX_u.shape[0],realX_u.shape[2]))
        cdp['cdp_%d' % i] = np.zeros((realX_u.shape[0],realX_u.shape[2]))
    
    dvmin = -0.2
    dvmax = 0.2
    nbtrial = 50

    zero_lag_ind = round(len(IRF_u['IRF_u_%d' % 0][0,:,0])/2)
    t_ini_d = 0
    t_length_d = tmax

    window = np.arange(int(zero_lag_ind), int((-tmin+tmax)/dtm+1),1)

    for i in range(realX_u.shape[2]):
        for j in range(realX_u.shape[0]):
            for k in range(realX_u.shape[2]):
                dv['dv_%d' % i][j,k], cc['cc_%d' % i][j,k], error['error_%d' % i][j,k], cdp['cdp_%d' % i][j,k] = Stretching_current(IRF_u['IRF_u_%d' % i][j,:,i],IRF_s['IRF_s_%d' % i][j,:,k],t,dvmin,dvmax,nbtrial,window,fmin,fmax,t_ini_d,t_length_d)

    for i in range(realX_u.shape[2]):
        for k in range(realX_u.shape[2]):
            np.savetxt("/gpfs/workdir/colombergi/GiorgiaGAN/results/dv_{:>d}_floor{:>d}_{:>d}.csv".format(d,i,k), dv['dv_%d' % i][:,k], delimiter=",")
            np.savetxt("/gpfs/workdir/colombergi/GiorgiaGAN/results/cc_{:>d}_floor{:>d}_{:>d}.csv".format(d,i,k), cc['cc_%d' % i][:,k], delimiter=",")
            np.savetxt("/gpfs/workdir/colombergi/GiorgiaGAN/results/error_{:>d}_floor{:>d}_{:>d}.csv".format(d,i,k), error['error_%d' % i][:,k], delimiter=",")

    for i in range(realX_u.shape[2]):
        fig = plt.figure(figsize=(6, 9))
        # Plot
        n = i+1
        data = []
        # Plot Correlation coefficient
        ax1 = fig.add_subplot(311)
        ax1.set_ylabel('Correlation coefficient')
        ax1.set_title('Undamaged VS Reconstructed Switched'+'  (Floor %d' % n + ')\n' +
                                'Stretching between '+ str(t_ini_d) + ' and '+ str( t_length_d) + ' s', fontsize = 12)              
        #ax1.set_xlim(labels[0], labels[-1])
        ax1.set_ylim(0.0, 1.1)
        ax1.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax1.grid(which='major', linewidth=1, alpha=0.5)
        #ax1.set_xticks(labels)

        for k in range(realX_u.shape[2]):
            x = []
            for j in range(realX_u.shape[0]):
                x.append(labels[k])
            ax1.scatter(x = x, y = cdp['cdp_%d' % i][:,k], s = 45, c = np.abs(cdp['cdp_%d' % i][:,k]), cmap = 'winter', edgecolors = 'k')
        
        ax2 = fig.add_subplot(312)
        ax2.set_ylabel('dv/v [%]')
        ax2.set_ylim(dvmin*100-5, dvmax*100+5)
        ax2.set_yticks([dvmin*100, dvmin*100/2, 0.0, dvmax*100/2, dvmax*100])
        ax2.grid(which='major', linewidth = 1, alpha=0.5)
        #ax2.set_xlim(labels[0], labels[-1])
        #ax2.set_xticks(labels)
        for k in range(realX_u.shape[2]):
            x = []
            for j in range(realX_u.shape[0]):
                x.append(labels[k])
            ax2.scatter(x = x, y = dv['dv_%d' % i][:,k], c = np.abs(dv['dv_%d' % i][:,k]), s = 45, cmap = 'autumn', edgecolors = 'k')
        
        ax3 = fig.add_subplot(313)
        for k in range(realX_u.shape[2]):
            data.append(dv['dv_%d' % i][:,k])
        ax3.boxplot(data)
        ax3.set_ylabel('dv/v [%]')
        ax3.grid(which='major', linewidth = 1, alpha=0.5)
        ax3.set_xlabel('Current floor')

        plt.subplots_adjust(hspace=0.2)
        fig.savefig('/gpfs/workdir/colombergi/GiorgiaGAN/results/stretching_{:>d}_{:>d}.png'.format(d,i), dpi=100)
        plt.close(fig)
 
    return





    

        



    
