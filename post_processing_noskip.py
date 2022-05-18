import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
import statistics
import scipy
from scipy import integrate
from scipy import signal
from scipy.stats import norm,lognorm
from scipy.fft import rfft, rfftfreq
import obspy.signal
from obspy.signal.tf_misfit import plot_tf_gofs, eg, pg
import itertools
import matplotlib
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from RepGAN_noskip import RepGAN, ParseOptions, WassersteinDiscriminatorLoss, WassersteinGeneratorLoss, GaussianNLL
from tensorflow.keras.optimizers import Adam

from matplotlib.pyplot import *
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict
cmaps = OrderedDict()
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from statistics import mean
import plotly.graph_objects as go
import plotly.express as px

from numpy.lib.type_check import imag
import sklearn
from PIL import Image
import io
import numpy as np

from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import CustomJS, Slider, ColumnDataSource
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.io import curdoc,output_file, show
import bokeh
from bokeh.models import Text, Label
import panel as pn
pn.extension()

checkpoint_dir = "./checkpoint_skip/21_04"

from interferometry_noskip import *
import MDOFload as mdof

from sklearn.manifold import TSNE

from random import seed
from random import randint

import matplotlib.font_manager
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Helvetica']
#families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
#rcParams['text.usetex'] = True

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import time
import warnings


from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


def correlation_lags(in1_len, in2_len, mode='full'):
    r"""
    Calculates the lag / displacement indices array for 1D cross-correlation.
    Parameters
    ----------
    in1_size : int
        First input size.
    in2_size : int
        Second input size.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `correlate` for more information.
    See Also
    --------
    correlate : Compute the N-dimensional cross-correlation.
    Returns
    -------
    lags : array
        Returns an array containing cross-correlation lag/displacement indices.
        Indices can be indexed with the np.argmax of the correlation to return
        the lag/displacement.
    Notes
    -----
    Cross-correlation for continuous functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left ( \tau \right )
        \triangleq \int_{t_0}^{t_0 +T}
        \overline{f\left ( t \right )}g\left ( t+\tau \right )dt
    Where :math:`\tau` is defined as the displacement, also known as the lag.
    Cross correlation for discrete functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left [ n \right ]
        \triangleq \sum_{-\infty}^{\infty}
        \overline{f\left [ m \right ]}g\left [ m+n \right ]
    Where :math:`n` is the lag.
    Examples
    --------
    Cross-correlation of a signal with its time-delayed self.
    >>> from scipy import signal
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> x = rng.standard_normal(1000)
    >>> y = np.concatenate([rng.standard_normal(100), x])
    >>> correlation = signal.correlate(x, y, mode="full")
    >>> lags = signal.correlation_lags(x.size, y.size, mode="full")
    >>> lag = lags[np.argmax(correlation)]
    """

    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags

def arias_intensity(dtm,tha,pc=0.95,nf=9.81):
    aid = np.pi/2./nf*scipy.integrate.cumtrapz(tha**2, dx=dtm, axis=-1, initial = 0.)
    mai = np.max(aid,axis=-1)
    ait = np.empty_like(mai)
    idx = np.empty_like(mai)
    if mai.size>1:
        for i in range(mai.size):
            ths = np.where(aid[i,...]/mai[i]>=pc)[0][0]
            ait[i] = aid[i,ths]
            idx[i] = ths*dtm
    else:
        ths = np.where(aid/mai>=pc)[0][0]
        ait = aid[ths]
        idx = ths*dtm
    return aid,ait,idx

def PlotLoss(history):
    # Plot loss
    fig, (ax0, ax1) = plt.subplots(2, 1,figsize=(18,18))
    #hfg = plt.figure(figsize=(12,6))
    #hax = hfg.add_subplot(111)
    ax0.set_rasterized(True)
    ax1.set_rasterized(True)
    loss0 = {}
    loss1 = {}
    loss0[r"$AdvDlossX$"] = history.history['AdvDlossX']
    loss0[r"$AdvDlossC$"] = history.history['AdvDlossC']
    loss0[r"$AdvDlossS$"] = history.history['AdvDlossS']
    loss0[r"$AdvDlossN$"] = history.history['AdvDlossN']
    loss0[r"$AdvGlossX$"] = history.history['AdvGlossX']
    loss0[r"$AdvGlossC$"] = history.history['AdvGlossC']
    loss1[r"$AdvGlossS$"] = history.history['AdvGlossS']
    loss1[r"$AdvGlossN$"] = history.history['AdvGlossN']
    loss1[r"$RecGlossX$"] = history.history['RecGlossX']
    loss1[r"$RecGlossC$"] = history.history['RecGlossC']
    loss1[r"$RecGlossS$"] = history.history['RecGlossS']
    
    clr = sn.color_palette("bright",len(loss0.keys()))
    i=0
    for k,v in loss0.items():
        ax0.plot(range(len(v)),v,linewidth=2,label=r"{}".format(k),color=clr[i])
        i+=1

    clr = sn.color_palette("bright",len(loss1.keys()))
    i=0
    for k,v in loss1.items():
        ax1.plot(range(len(v)),v,linewidth=2,label=r"{}".format(k),color=clr[i])
        i+=1

    #hax.set_xlabel('Epoch', fontsize=14)
    labels_legend0= [r"$\mathcal{L}_{AdvX}$",r"$\mathcal{L}_{AdvC}$",r"$\mathcal{L}_{AdvS}$",r"$\mathcal{L}_{AdvN}$",
        r"$\mathcal{L}_{GenX}$",r"$\mathcal{L}_{GenC}$"]#,r"$\mathcal{L}_{AdvClass}$",r"$\mathcal{L}_{GenClass}$"
    ax0.legend(labels_legend0,fontsize=18,frameon=False,loc='upper right')
    ax0.tick_params(axis='both', labelsize=18)
    ax0.set_xlabel(r"$n_{epochs}$",fontsize=20,fontweight='bold')
    ax0.set_ylabel(r'$Loss \hspace{0.5} [1]$',fontsize=20,fontweight='bold')

    labels_legend1= [r"$\mathcal{L}_{GenS}$",r"$\mathcal{L}_{GenN}$",r"$\mathcal{L}_{RecX}$",
        r"$\mathcal{L}_{RecC}$",r"$\mathcal{L}_{RecS}$"] #r"$\mathcal{L}_{AdvAll}$",r"$\mathcal{L}_{GenAll}$"
    ax1.legend(labels_legend1,fontsize=18,frameon=False,loc='upper right')
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_xlabel(r"$n_{epochs}$",fontsize=20,fontweight='bold')
    ax1.set_ylabel(r'$Loss \hspace{0.5} [1]$',fontsize=20,fontweight='bold')

    plt.savefig('./results_skip/loss.png',format='png',bbox_inches = 'tight')
    #plt.savefig('./results_skip/loss.eps',format='eps',rasterized=True,bbox_inches = 'tight',dpi=200)
    plt.close()

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.set_rasterized(True)
    hax.plot(history.history['fakeX'],linewidth=2,color='r',marker='^', linestyle='', label=r'$D_x(G_z(s,c,n))$')
    hax.plot(history.history['realX'],linewidth=2,color='b',marker='s', linestyle='', label=r'$D_x(x)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_x$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_x \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=20)
    hax.tick_params(axis='both', labelsize=18)
    plt.savefig('./results_skip/D_x.png',bbox_inches = 'tight')
    #plt.savefig('./results_skip/D_x.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.plot(history.history['fakeC'],linewidth=2, color='r', marker='^', linestyle='', label=r'$D_c(F_x(x))$')
    hax.plot(history.history['realC'],linewidth=2, color='b', marker='s', linestyle='', label=r'$D_c(c)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_c$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_c \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=20)
    hax.tick_params(axis='both', labelsize=18)
    plt.savefig('./results_skip/D_c.png',bbox_inches = 'tight')
    #plt.savefig('./results_skip/D_c.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.plot(history.history['fakeS'],linewidth=2, color='r', marker='^', linestyle='', label=r'$D_s(F_x(x))$')
    hax.plot(history.history['realS'],linewidth=2, color='b', marker='s', linestyle='', label=r'$D_s(s)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_s$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_s \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=20)
    hax.tick_params(axis='both', labelsize=18)
    plt.savefig('./results_skip/D_s.png',bbox_inches = 'tight')
    #plt.savefig('./results_skip/D_s.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.plot(history.history['fakeN'],linewidth=2, color='r', marker='^', linestyle='', label=r'$D_n(F_x(x))$')
    hax.plot(history.history['realN'],linewidth=2, color='b', marker='s', linestyle='', label=r'$D_n(n)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_n$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_n \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=20)
    hax.tick_params(axis='both', labelsize=18)
    plt.savefig('./results_skip/D_n.png',bbox_inches = 'tight')
    #plt.savefig('./results_skip/D_n.eps',bbox_inches = 'tight',dpi=200)
    plt.close()



def PlotReconstructedTHs(model,realXC):
    # Plot reconstructed time-histories
    realX = np.concatenate([x for x, c in realXC], axis=0)
    realC = np.concatenate([c for x, c in realXC], axis=0)

    recX,fakeC,fakeS,fakeN,fakeX = model.plot(realX,realC)

    t = np.zeros(realX.shape[1])
    for k in range(realX.shape[1]-1):
        t[k+1] = (k+1)*0.04

    recX_fft = tf.make_ndarray(tf.make_tensor_proto(recX))

    # Print real vs reconstructed signal
    for j in range(realX.shape[2]):
        for k in range(10):
            i = randint(0, realX.shape[0]-1)
            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            hax.plot(t,realX[i,:,j], color='black')
            hax.plot(t,recX[i,:,j], color='orange',linestyle="--")
            #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
            hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
            hax.set_ylim([-1.0, 1.0])
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig('./results_skip/reconstruction_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
            #plt.savefig('./results_skip/reconstruction_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
            plt.close()


            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            N = realX.shape[1]
            SAMPLE_RATE = 25
            yf_real = rfft(realX[i,:,j])
            xf_real = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_real, np.abs(yf_real), color='black')
            yf_rec = rfft(recX_fft[i,:,j])
            xf_rec = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_rec, np.abs(yf_rec), color='orange',linestyle="--")
            hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig('./results_skip/fft_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
            #plt.savefig('./results_skip/fft_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

def cross_2d_dam(und,dam,i0,dt,nw,kspec,fmin,fmax,tmin,tmax):
    
    fnyq = 0.5/dt
    wn   = [fmin/fnyq,fmax/fnyq]
    b, a = butter(6, wn,'bandpass',output='ba')

    ntr = und.shape[1] 
    x   = und[:,i0]
    for i in range(ntr):
        y = dam[:,i]
        Pxy  = MTCross(y,x,nw,kspec,dt,iadapt=2,wl=0.0)
        xcorr, dcohe, dconv  = Pxy.mt_corr()
        dconv = filtfilt(b, a, dcohe[:,0])
        if (i==0):
            k    = np.linspace(-Pxy.npts,Pxy.npts,len(xcorr),dtype=int)
            t2   = k*dt
            tloc = np.where((t2>=tmin) & (t2<=tmax))[0]
            irf  = np.zeros((len(tloc),ntr))
        irf[:,i] = dconv[tloc]
        t        = t2[tloc]
    
    return [irf,t]


def PlotSwitchedTHs(model,real_u,real_d,d):
    # Plot reconstructed time-histories
    
    realX_u = np.concatenate([x for x, c, in real_u], axis=0)
    realC_u = np.concatenate([c for x, c, in real_u], axis=0)

    recX_u,_,_,_ = model.predict(realX_u)

    realX_d = np.concatenate([x for x, c, in real_d], axis=0)
    realC_d = np.concatenate([c for x, c, in real_d], axis=0)

    recX_d,_,_,_ = model.predict(realX_d)

    t = np.zeros(realX_u.shape[1])
    for m in range(realX_u.shape[1]-1):
        t[m+1] = (m+1)*0.04

    if d==1:

        recX_fft = tf.make_ndarray(tf.make_tensor_proto(recX_u))

        # Print real vs reconstructed signal
        for j in range(realX_u.shape[2]):
            for k in range(10):
                i = randint(0, realX_u.shape[0]-1)
                hfg = plt.figure(figsize=(12,6),tight_layout=True)
                hax = hfg.add_subplot(111)
                hax.plot(t,realX_u[i,:,j], color='black')
                hax.plot(t,recX_u[i,:,j], color='orange',linestyle="--")
                #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
                hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
                hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
                hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
                hax.set_ylim([-1.0, 1.0])
                hax.tick_params(axis='both', labelsize=18)
                plt.savefig('./results_skip/reconstruction0_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
                #plt.savefig('./results_skip/reconstruction0_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
                plt.close()


                hfg = plt.figure(figsize=(12,6),tight_layout=True)
                hax = hfg.add_subplot(111)
                N = realX_u.shape[1]
                SAMPLE_RATE = 25
                yf_real = rfft(realX_u[i,:,j])
                xf_real = rfftfreq(N, 1 / SAMPLE_RATE)
                hax.plot(xf_real, np.abs(yf_real), color='black')
                yf_rec = rfft(recX_fft[i,:,j])
                xf_rec = rfftfreq(N, 1 / SAMPLE_RATE)
                hax.plot(xf_rec, np.abs(yf_rec), color='orange',linestyle="--")
                hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
                hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
                hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
                hax.tick_params(axis='both', labelsize=18)
                plt.savefig('./results_skip/fft0_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
                #plt.savefig('./results_skip/fft0_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
                plt.close()

    recX_fft = tf.make_ndarray(tf.make_tensor_proto(recX_d))

    # Print real vs reconstructed signal
    for j in range(realX_d.shape[2]):
        for k in range(10):
            i = randint(0, realX_d.shape[0]-1)
            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            hax.plot(t,realX_d[i,:,j], color='black')
            hax.plot(t,recX_d[i,:,j], color='orange',linestyle="--")
            #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
            hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
            hax.set_ylim([-1.0, 1.0])
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig('./results_skip/reconstruction{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results_skip/reconstruction{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()


            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            N = realX_d.shape[1]
            SAMPLE_RATE = 25
            yf_real = rfft(realX_d[i,:,j])
            xf_real = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_real, np.abs(yf_real), color='black')
            yf_rec = rfft(recX_fft[i,:,j])
            xf_rec = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_rec, np.abs(yf_rec), color='orange',linestyle="--")
            hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig('./results_skip/fft{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results_skip/fft{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

        
    fakeC_new = np.zeros_like((realC_d))
    fakeC_new[:,d] = 1.0
    fakeX_new = model.generate(realX_u,fakeC_new)
    fakeX_new_fft = tf.make_ndarray(tf.make_tensor_proto(fakeX_new))

    corr_real = np.zeros((realX_u.shape[0],realX_u.shape[1]*2-1,realX_u.shape[2]))
    corr_switch = np.zeros((realX_u.shape[0],realX_u.shape[1]*2-1,realX_u.shape[2]))
    lags_real = np.zeros((realX_u.shape[0],realX_u.shape[1]*2-1,realX_u.shape[2]))
    lags_switch = np.zeros((realX_u.shape[0],realX_u.shape[1]*2-1,realX_u.shape[2]))

    for j in range(realX_u.shape[0]):
        for i in range(realX_u.shape[2]):
            corr_real[j,:,i] = signal.correlate(realX_d[j,:,i], realX_u[j,:,i])
            lags_real[j,:,i] = correlation_lags(len(realX_d[j,:,i]), len(realX_u[j,:,i]))
            corr_real[j,:,i] /= np.max(corr_real[j,:,i])

            corr_switch[j,:,i] = signal.correlate(fakeX_new[j,:,i], realX_u[j,:,i])
            lags_switch[j,:,i] = correlation_lags(len(fakeX_new[j,:,i]), len(realX_u[j,:,i]))
            corr_switch[j,:,i] /= np.max(corr_switch[j,:,i])

    
    t = np.zeros(realX_u.shape[1])
    for m in range(realX_u.shape[1]-1):
        t[m+1] = (m+1)*0.04

    for j in range(realX_u.shape[2]):
        for k in range(10):
            i = randint(0, realX_u.shape[0]-1)
            fig, (ax0, ax1, ax2) = plt.subplots(3, 1,figsize=(12,18))
            ax0.plot(t,realX_u[i,:,j], color='green')
            ax1.plot(t,realX_d[i,:,j], color='black')
            ax2.plot(t,fakeX_new[i,:,j], color='orange')
            #hfg = plt.subplots(3,1,figsize=(12,6),tight_layout=True)
            #hax = hfg.add_subplot(111)
            #hax.plot(t,realX_u[0,:,0],t,realX_d[0,:,0],t,fakeX_d[0,:,0])
            #hax.plot(t,fakeX_u[0,:,0], color='orange')
            ax0.set_ylim([-1.0, 1.0])
            ax1.set_ylim([-1.0, 1.0])
            ax2.set_ylim([-1.0, 1.0])
            #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
            ax0.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            ax0.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            ax0.legend([r'$X_u$'], loc='best',frameon=False,fontsize=20)
            ax0.tick_params(axis='both', labelsize=18)
            ax1.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            ax1.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            ax1.legend([r'$X_d$'], loc='best',frameon=False,fontsize=20)
            ax1.tick_params(axis='both', labelsize=18)
            ax2.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            ax2.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            ax2.legend([r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
            ax2.tick_params(axis='both', labelsize=18)
            plt.savefig('./results_skip/reconstruction_switch{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results_skip/reconstruction_switch{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            N = realX_u.shape[1]
            SAMPLE_RATE = 25
            yf_real_d = rfft(realX_d[i,:,j])
            xf_real_d = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_real_d, np.abs(yf_real_d), color='black')
            yf_switch = rfft(fakeX_new_fft[i,:,j])
            xf_switch = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_switch, np.abs(yf_switch), color='orange',linestyle="--")
            hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X_d$', r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig('./results_skip/fft_switch{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results_skip/fft_switch{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            hax.plot(t,realX_d[i,:,j], color='black')
            hax.plot(t,fakeX_new[i,:,j], color='orange',linestyle="--")
            hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X_d$',r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
            hax.tick_params(axis='both', labelsize=18)
            hax.set_ylim([-1.0, 1.0])           

            plt.savefig('./results_skip/switch{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results_skip/switch{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

            fig, axs = plt.subplots(2, 2, figsize=(24,12))
            axs[0,0].plot(t,realX_u[i,:,j], color='black')
            axs[0,0].plot(t,realX_d[i,:,j], color='orange',linestyle="--")
            axs[0,0].set_title('Signals', fontsize=30,fontweight='bold')
            axs[0,0].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            axs[0,0].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            axs[0,0].legend([r'$X_u$',r"$X_d$"], loc='best',frameon=False,fontsize=20)

            axs[1,0].plot(lags_real[i,:,j],corr_real[i,:,j])
            axs[1,0].set_title('Cross-correlated signal', fontsize=30,fontweight='bold')
            axs[1,0].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            axs[1,0].set_xlabel(r'$Lag$', fontsize=26,fontweight='bold')

            axs[0,1].plot(t,realX_u[i,:,j], color='black')
            axs[0,1].plot(t,fakeX_new[i,:,j], color='orange',linestyle="--")
            axs[0,1].set_title('Signals', fontsize=30,fontweight='bold')
            axs[0,1].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            axs[0,1].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            axs[0,1].legend([r'$X_u$',r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)

            axs[1,1].plot(lags_switch[i,:,j],corr_switch[i,:,j])
            axs[1,1].set_title('Cross-correlated signal', fontsize=30,fontweight='bold')
            axs[1,1].set_xlabel(r'$Lag$', fontsize=26,fontweight='bold')
            axs[1,1].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            

            fig.tight_layout()

            plt.savefig('./results_skip/cross-corr{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results_skip/cross-corr{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            hax.plot(lags_real[i,:,j],corr_real[i,:,j], color='black')
            hax.plot(lags_switch[i,:,j],corr_switch[i,:,j], color='orange',linestyle="--")
            hax.set_title('Cross-correlated signals - Comparison', fontsize=30,fontweight='bold')
            hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$Lag$', fontsize=26,fontweight='bold')
            hax.legend([r'$Original$',r"$Switch$"], loc='best',frameon=False,fontsize=20)

            plt.savefig('./results_skip/cross-corr-comparison{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results_skip/cross-corr-comparison{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

    deconvolution(realX_u,realX_d,fakeX_new,d)



def PlotTHSGoFs(model,realXC):
    # Plot reconstructed time-histories
    #realX, realC = realXC

    realX = np.concatenate([x for x, c, in realXC], axis=0)
    realC = np.concatenate([c for x, c, in realXC], axis=0)

    recX,fakeC,fakeS,fakeN = model.predict(realX)

    ## Print signal GoF
    for j in range(realX.shape[2]):
        for k in range(10):
            i = randint(0, realX.shape[0]-1)
            plot_tf_gofs(realX[i,:,j],recX[i,:,j],dt=0.04,fmin=0.1,fmax=30.0,t0=0.0,nf=100,w0=6,norm='global',st2_isref=True,
                a=10.,k=1.,left=0.1,bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2,w_1=0.2,w_2=0.6,w_cb=0.01, d_cb=0.0,show=False,
                plot_args=['k', 'r', 'b'],ylim=0., clim=0.)
            plt.savefig('./results_skip/gof_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
            #plt.savefig('./results_skip/gof_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

def colored_scatter(*args, **kwargs):
    plt.scatter(*args, **kwargs)
    return

def PlotEGPGgrid(col_x,col_y,col_k,i,df,k_is_color=False, scatter_alpha=.7):
    k=0
    for name, df_group in df.groupby(col_k):
        k+=1
    plt.figure(figsize=(10,6), dpi= 500)
    sn.color_palette("Paired", k)
    def colored_scatter(x, y, c=None, edgecolor='black', linewidth=0.8):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            kwargs['edgecolor']=edgecolor
            kwargs['linewidth']=linewidth
            plt.scatter(*args, **kwargs)

        return scatter
    g = sn.JointGrid(x=col_x,y=col_y,data=df,space=0.0)
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(colored_scatter(df_group[col_x],df_group[col_y],color),)
        hax=sn.distplot(df_group[col_x].values,ax=g.ax_marg_x,kde=False,color=color,norm_hist=True)
        hay=sn.distplot(df_group[col_y].values,ax=g.ax_marg_y,kde=False,color=color,norm_hist=True,vertical=True)
        hax.set_xticks(list(np.linspace(0,10,11)))
        hay.set_yticks(list(np.linspace(0,10,11)))
    ## Do also global Hist:
    g.ax_joint.set_xticks(list(np.linspace(0,10,11)))
    g.ax_joint.set_yticks(list(np.linspace(0,10,11)))
    g.ax_joint.spines['right'].set_visible(False)
    g.ax_joint.spines['left'].set_visible(True)
    g.ax_joint.spines['bottom'].set_visible(True)
    g.ax_joint.spines['top'].set_visible(False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('EG', fontsize=14)
    plt.ylabel('PG', fontsize=14)
    plt.legend(legends,frameon=False,fontsize=14)
    plt.savefig('./results_skip/Gz(Fx(X))_gofs_{:>d}.png'.format(i),bbox_inches = 'tight')
    #plt.savefig('./results_skip/Gz(Fx(X))_gofs_{:>d}.eps'.format(i),bbox_inches = 'tight',dpi=200)
    plt.close()


def PlotBatchGoFs(model,Xtrn,Xvld,i):
    # Plot GoFs on a batch

    realX_trn = np.concatenate([x for x, c in Xtrn], axis=0)
    realC_trn = np.concatenate([c for x, c in Xtrn], axis=0)

    fakeX_trn,_,_,_ = model.predict(realX_trn)

    realX_vld = np.concatenate([x for x, c in Xvld], axis=0)
    realC_vld = np.concatenate([c for x, c in Xvld], axis=0)

    fakeX_vld,_,_,_ = model.predict(realX_vld)

    egpg_trn = {}
    for j in range(realX_trn.shape[2]):
        egpg_trn['egpg_trn_%d' % j] = np.zeros((realX_trn.shape[0],2),dtype=np.float32)
        st1 = np.zeros((realX_trn.shape[0],realX_trn.shape[1]))
        st2 = np.zeros((realX_trn.shape[0],realX_trn.shape[1]))
        for k in range(realX_trn.shape[0]):
            st1 = realX_trn[k,:,j]
            st2 = fakeX_trn[k,:,j]
            egpg_trn['egpg_trn_%d' % j][k,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
            egpg_trn['egpg_trn_%d' % j][k,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

    egpg_vld = {}
    for j in range(realX_vld.shape[2]):
        egpg_vld['egpg_vld_%d' % j] = np.zeros((realX_vld.shape[0],2),dtype=np.float32)
        st1 = np.zeros((realX_vld.shape[0],realX_vld.shape[1]))
        st2 = np.zeros((realX_vld.shape[0],realX_vld.shape[1]))
        for k in range(realX_vld.shape[0]):
            st1 = realX_vld[k,:,j]
            st2 = fakeX_vld[k,:,j]
            egpg_vld['egpg_vld_%d' % j][k,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
            egpg_vld['egpg_vld_%d' % j][k,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)


    egpg_df_trn = {}
    for j in range(realX_trn.shape[2]):
        egpg_df_trn['egpg_df_trn_%d' % j] = pd.DataFrame(egpg_trn['egpg_trn_%d' % j],columns=['EG','PG'])
        egpg_df_trn['egpg_df_trn_%d' % j]['kind']=r"$G_z(F_x(x)) \hspace{0.5} train$"

    egpg_df_vld = {}
    for j in range(realX_vld.shape[2]):
        egpg_df_vld['egpg_df_vld_%d' % j] = pd.DataFrame(egpg_vld['egpg_vld_%d' % j],columns=['EG','PG'])
        egpg_df_vld['egpg_df_vld_%d' % j]['kind']=r"$G_z(F_x(x)) \hspace{0.5} validation$"

    egpg_data = []
    for j in range(realX_trn.shape[2]):
        egpg_data.append(egpg_df_trn['egpg_df_trn_%d' % j])
    for j in range(realX_vld.shape[2]):
        egpg_data.append(egpg_df_vld['egpg_df_vld_%d' % j])
    egpg_df = pd.concat(egpg_data)

    egpg_df.to_csv('./results_skip/EG_PG_{:>d}.csv'.format(i), index= True)
    PlotEGPGgrid('EG','PG','kind',i,df=egpg_df)

def PlotClassificationMetrics(model,realXC):
    # Plot classification metrics
    realX = np.concatenate([x for x, c in realXC], axis=0)
    realC = np.concatenate([c for x, c in realXC], axis=0)

    fakeC, recC = model.label_predictor(realX,realC)

    labels_fake = np.zeros((fakeC.shape[0]))
    for i in range(fakeC.shape[0]):
        labels_fake[i] = np.argmax(fakeC[i,:])

    labels_rec = np.zeros((recC.shape[0]))
    for i in range(recC.shape[0]):
        labels_rec[i] = np.argmax(recC[i,:])
    
    labels_real = np.zeros((realC.shape[0]))
    for i in range(realC.shape[0]):
        labels_real[i] = np.argmax(realC[i,:])

    labels_fake = labels_fake.astype(int)
    labels_rec = labels_rec.astype(int)
    labels_real = labels_real.astype(int)

    target_names = []
    for i in range(options['latentCdim']):
        target_names.append('damage class %d'% i) 

    fig, ax = plt.subplots()
    report = classification_report(y_true = labels_real, y_pred = labels_fake,
            target_names=target_names,output_dict=True,zero_division=1)
    df = pd.DataFrame(report).transpose()
    df.to_csv('./results_skip/Classification Report C.csv', index= True)
    cr = sn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, annot_kws={"size": 12})
    cr.tick_params(axis='both', labelsize=12)
    cr.set_yticklabels(cr.get_yticklabels(), rotation=0)
    plt.savefig('./results_skip/classification_report_fakeC.png',bbox_inches = 'tight')
    #plt.savefig('./results_skip/classification_report.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    conf_mat = confusion_matrix(labels_real, labels_fake)
    fig, ax = plt.subplots(figsize=(10,10),tight_layout=True)
    sn.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names,yticklabels=target_names, vmin=0, vmax=realC.shape[0],
        annot_kws={"size": 20})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(axis='both', labelsize=20)
    plt.ylabel("True class",fontsize=22,labelpad=10)
    plt.xlabel("Predicted class",fontsize=22,labelpad=10)
    plt.savefig('./results_skip/confusion_matrix_fakeC.png',bbox_inches = 'tight')
    #plt.savefig('./results_skip/confusion_matrixC.eps',bbox_inches = 'tight',dpi=200)
    plt.close()


    fig, ax = plt.subplots()
    report = classification_report(y_true = labels_real, y_pred = labels_rec,
            target_names=target_names,output_dict=True,zero_division=1)
    df = pd.DataFrame(report).transpose()
    df.to_csv('./results_skip/Classification Report recC.csv', index= True)
    cr = sn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, annot_kws={"size": 12})
    cr.tick_params(axis='both', labelsize=12)
    cr.set_yticklabels(cr.get_yticklabels(), rotation=0)
    plt.savefig('./results_skip/classification_report_recC.png',bbox_inches = 'tight')
    #plt.savefig('./results_skip/classification_reportrec.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    
    conf_mat = confusion_matrix(labels_real, labels_rec)
    fig, ax = plt.subplots(figsize=(10,10),tight_layout=True)
    sn.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names,yticklabels=target_names, vmin=0, vmax=realC.shape[0],
        annot_kws={"size": 20})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(axis='both', labelsize=20)
    plt.ylabel("True class",fontsize=22,labelpad=10)
    plt.xlabel("Predicted class",fontsize=22,labelpad=10)
    plt.savefig('./results_skip/confusion_matrix_recC.png',bbox_inches = 'tight')
    #plt.savefig('./results_skip/confusion_matrixrecC.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    return

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

def PlotLatentSpace(model,realXC):
    s_list = []
    n_list = []
    cq_list = []
    c_list = []

    iterator = iter(realXC)  # restart data iter
    for b in range(len(realXC)):
        data = iterator.get_next()
        realX = data[0]
        realC = data[1]
        # import pdb
        # pdb.set_trace()
        [_,C_set,S_set,N_set] = model.predict(realX,batch_size=1)
        s_list.append(S_set)
        n_list.append(N_set)
        cq_list.append(C_set)
        c_list.append(realC)

    s_np = tf.concat(s_list, axis=0).numpy().squeeze()
    n_np = tf.concat(n_list, axis=0).numpy().squeeze()
    cq_tensor = tf.concat(cq_list, axis=0)
    c_tensor = tf.concat(cq_list, axis=0)
    cq_np = np.argmax(cq_tensor.numpy().squeeze(), axis = -1)
    c_np = np.argmax(c_tensor.numpy().squeeze(), axis = -1)

    fig = plt.figure()

    #hist plot
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ax = fig.add_axes(rect_scatter)

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(s_np[:,0],s_np[:,1], ax, ax_histx, ax_histy)
    fig.savefig('./results_skip/s_all.png')

    for n_i in range(1):
        #per example
        fig, ax = plt.subplots()
        for i,n in enumerate(n_np):
            if cq_np[i] == 0 and c_np[i] == cq_np[i] :
                plt.scatter(n[n_i],n[n_i+1],c = 'b',marker='o', alpha=0.5,s=12)
            elif cq_np[i] == 0 and c_np[i] != cq_np[i] :
                plt.scatter(n[n_i],n[n_i+1],c = 'b',marker='x', alpha=0.5,s=12)
            elif cq_np[i] == 1 and c_np[i] == cq_np[i] :
                plt.scatter(n[n_i],n[n_i+1],c = 'r',marker='o', alpha=0.5,s=12)
            elif cq_np[i] == 1 and c_np[i] != cq_np[i] :
                plt.scatter(n[n_i],n[n_i+1],c = 'r',marker='x', alpha=0.5,s=12)

        plt.ylabel("N1",fontsize=12,labelpad=10)
        plt.xlabel("N0",fontsize=12,labelpad=10)
        plt.title("N variables",fontsize=16)
        plt.legend(["0","1"],frameon=False,fontsize=14)
        fig.savefig('./results_skip/n_{:>d}_{:>d}'.format(n_i,n_i+1),dpi=300,bbox_inches = 'tight')
        plt.clf()

    
    for s_i in range(1):
        #per example
        fig, ax = plt.subplots()
        for i,s in enumerate(s_np):
            if cq_np[i] == 0 and c_np[i] == cq_np[i] :
                plt.scatter(s[s_i],s[s_i+1],c = 'b',marker='o', alpha=0.5,s=12)
            elif cq_np[i] == 0 and c_np[i] != cq_np[i] :
                plt.scatter(s[s_i],s[s_i+1],c = 'b',marker='x', alpha=0.5,s=12)
            elif cq_np[i] == 1 and c_np[i] == cq_np[i] :
                plt.scatter(s[s_i],s[s_i+1],c = 'r',marker='o', alpha=0.5,s=12)
            elif cq_np[i] == 1 and c_np[i] != cq_np[i] :
                plt.scatter(s[s_i],s[s_i+1],c = 'r',marker='x', alpha=0.5,s=12)
            
        plt.ylabel("S1",fontsize=12,labelpad=10)
        plt.xlabel("S0",fontsize=12,labelpad=10)
        plt.title("S variables",fontsize=16)
        plt.legend(["0","1"],frameon=False,fontsize=14)

        fig.savefig('./results_skip/s_{:>d}_{:>d}'.format(s_i,s_i+1),dpi=300,bbox_inches = 'tight')
        plt.clf()

    return

def PlotTSNE(model,realXC):
    
    realX = np.concatenate([x for x, c, in realXC], axis=0)
    realC = np.concatenate([c for x, c, in realXC], axis=0)

    _,_,fakeS,fakeN = model.predict(realX)

    labels = np.zeros((realC.shape[0]))
    for i in range(realC.shape[0]):
        labels[i] = np.argmax(realC[i,:])

    
    transformerN = TSNE(n_components=2, verbose=1, random_state=123)
    n = transformerN.fit_transform(fakeN)

    dfN = pd.DataFrame()
    dfN["C"] = labels
    dfN["Dimension 1"] = n[:,0]
    dfN["Dimension 2"] = n[:,1]

    sn.scatterplot(x="Dimension 1", y="Dimension 2", hue=dfN.C.tolist(),
            palette=sn.color_palette("hls", 2),data=dfN).set(title="Variables N T-SNE projection")
    plt.savefig('./results_skip/tsne_N.png',bbox_inches = 'tight')
    plt.close()
    
    transformerS = TSNE(n_components=2, verbose=1, random_state=123)

    s = transformerS.fit_transform(fakeS)

    dfS = pd.DataFrame()
    dfS["C"] = labels
    dfS["Dimension 1"] = s[:,0]
    dfS["Dimension 2"] = s[:,1]

    sn.scatterplot(x="Dimension 1", y="Dimension 2", hue=dfS.C.tolist(),
                palette=sn.color_palette("hls", 2),data=dfS).set(title="Variables S T-SNE projection")
    plt.savefig('./results_skip/tsne_S.png',bbox_inches = 'tight')
    plt.close()

    return

def PlotChangeS(model,realXC):
    
    realX = np.concatenate([x for x, c, in realXC], axis=0)
    realC = np.concatenate([c for x, c, in realXC], axis=0)

    realS,realN,recS,recC,recN = model.cycling(realX,realC)

    labels_rec = np.zeros((recC.shape[0]))
    for i in range(recC.shape[0]):
        labels_rec[i] = np.argmax(recC[i,:])
    
    labels_real = np.zeros((realC.shape[0]))
    for i in range(realC.shape[0]):
        labels_real[i] = np.argmax(realC[i,:])

    labels_rec = labels_rec.astype(int)
    labels_real = labels_real.astype(int)

    target_names = []
    for i in range(options['latentCdim']):
        target_names.append('damage class %d'% i) 

    fig, ax = plt.subplots()
    report = classification_report(y_true = labels_real, y_pred = labels_rec,
            target_names=target_names,output_dict=True,zero_division=1)
    df = pd.DataFrame(report).transpose()
    df.to_csv('./results/ChangeS_ClassificationC.csv', index= True)
    cr = sn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, annot_kws={"size": 12})
    cr.tick_params(axis='both', labelsize=12)
    cr.set_yticklabels(cr.get_yticklabels(), rotation=0)
    plt.savefig('./results_skip/ChangeS_ClassificationC.png',bbox_inches = 'tight')
    plt.close()

    conf_mat = confusion_matrix(labels_real, labels_rec)
    fig, ax = plt.subplots(figsize=(10,10),tight_layout=True)
    sn.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names,yticklabels=target_names, vmin=0, vmax=realC.shape[0],
        annot_kws={"size": 20})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(axis='both', labelsize=20)
    plt.ylabel("True class",fontsize=22,labelpad=10)
    plt.xlabel("Predicted class",fontsize=22,labelpad=10)
    plt.savefig('./results_skip/ChangeS_ConfusionC.png',bbox_inches = 'tight')
    plt.close()

    transformerN1 = TSNE(n_components=2, verbose=1, random_state=123)
    n1 = transformerN1.fit_transform(realN)

    dfN1 = pd.DataFrame()
    dfN1["C"] = labels_real
    dfN1["Dimension 1"] = n1[:,0]
    dfN1["Dimension 2"] = n1[:,1]

    transformerN2 = TSNE(n_components=2, verbose=1, random_state=123)
    n2 = transformerN2.fit_transform(recN)

    dfN2 = pd.DataFrame()
    dfN2["C"] = labels_rec
    dfN2["Dimension 1"] = n2[:,0]
    dfN2["Dimension 2"] = n2[:,1]

    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].set_title("N: T-SNE projection")
    sn.scatterplot(ax=ax[0], x="Dimension 1", y="Dimension 2", hue=dfN1.C.tolist(),palette=sn.color_palette("hls", 2),data=dfN1)

    ax[1].set_title(r"$F_x(G_z(c,s,n))$: T-SNE projection")
    sn.scatterplot(ax=ax[1], x="Dimension 1", y="Dimension 2", hue=dfN2.C.tolist(),palette=sn.color_palette("hls", 2),data=dfN2)
    plt.savefig('./results_skip/ChangeS_tsne_N.png',bbox_inches = 'tight')
    plt.close()
    
    transformerS1 = TSNE(n_components=2, verbose=1, random_state=123)
    s1 = transformerS1.fit_transform(realS)

    dfS1 = pd.DataFrame()
    dfS1["C"] = labels_real
    dfS1["Dimension 1"] = s1[:,0]
    dfS1["Dimension 2"] = s1[:,1]

    transformerS2 = TSNE(n_components=2, verbose=1, random_state=123)
    s2 = transformerS2.fit_transform(recS)

    dfS2 = pd.DataFrame()
    dfS2["C"] = labels_rec
    dfS2["Dimension 1"] = s2[:,0]
    dfS2["Dimension 2"] = s2[:,1]

    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].set_title("S: T-SNE projection")
    sn.scatterplot(ax=ax[0], x="Dimension 1", y="Dimension 2", hue=dfS1.C.tolist(),palette=sn.color_palette("hls", 2),data=dfS1)

    ax[1].set_title(r"$F_x(G_z(c,s,n))$: T-SNE projection")
    sn.scatterplot(ax=ax[1], x="Dimension 1", y="Dimension 2", hue=dfS2.C.tolist(),palette=sn.color_palette("hls", 2),data=dfS2)
    plt.savefig('./results_skip/ChangeS_tsne_S.png',bbox_inches = 'tight')
    plt.close()

    return

def PlotDistributions(model,realXC):
    
    realX = np.concatenate([x for x, c in realXC], axis=0)
    realC = np.concatenate([c for x, c in realXC], axis=0)

    realS, realN, fakeS, fakeN, recS, recN = model.distribution(realX,realC)


    for i in range(realS.shape[1]):
        sn.distplot(realS[:,i], hist=True, kde=True, color = 'b', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, label='S')
        sn.distplot(fakeS[:,i], hist=True, kde=True, color = 'g', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, label=r"$F_x(x)$")
        sn.distplot(recS[:,i], hist=True, kde=True, color = 'r', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, label=r"$F_x(G_z(c,s,n))$")
        plt.legend()
        plt.title('Distribution S{:>d}'.format(i))
        plt.xlabel("Variable S")
        plt.ylabel("Density")
        plt.savefig('./results_skip/distribution_S_{:>d}.png'.format(i),bbox_inches = 'tight')
        plt.close()

    # for i in range(realN.shape[1]):
    #     sn.distplot(realN[:,i], hist=True, kde=True, color = 'b', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, label='N')
    #     sn.distplot(fakeN[:,i], hist=True, kde=True, color = 'g', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, label=r"$F_x(x)$")
    #     sn.distplot(recN[:,i], hist=True, kde=True, color = 'r', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, label=r"$F_x(G_z(c,s,n))$")
    #     plt.legend()
    #     plt.title('Distribution N{:>d}'.format(i))
    #     plt.xlabel("Variable N")
    #     plt.ylabel("Density")
    #     plt.savefig('./results_skip/distribution_N_{:>d}.png'.format(i),bbox_inches = 'tight')
    #     plt.close()

    return




options = ParseOptions()

# MODEL LOADING
optimizers = {}
optimizers['DxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['DcOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['DsOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['DnOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['FxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['QOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['GzOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['GqOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)

losses = {}
losses['AdvDlossWGAN'] = WassersteinDiscriminatorLoss
losses['AdvGlossWGAN'] = WassersteinGeneratorLoss
losses['AdvDlossGAN'] = tf.keras.losses.BinaryCrossentropy()
losses['AdvGlossGAN'] = tf.keras.losses.BinaryCrossentropy()
losses['RecSloss'] = GaussianNLL
losses['RecXloss'] = tf.keras.losses.MeanAbsoluteError()  # XLoss #
losses['RecCloss'] = tf.keras.losses.CategoricalCrossentropy()
losses['PenAdvXloss'] = 1.
losses['PenAdvCloss'] = 1.
losses['PenAdvSloss'] = 1.
losses['PenAdvNloss'] = 1.
losses['PenRecXloss'] = 1.
losses['PenRecCloss'] = 1.
losses['PenRecSloss'] = 1.

# Instantiate the RepGAN model.
GiorgiaGAN = RepGAN(options)

# Compile the RepGAN model.
GiorgiaGAN.compile(optimizers, losses)  # run_eagerly=True

Xtrn, Xvld, _ = mdof.LoadData(**options)

GiorgiaGAN.Fx = keras.models.load_model("./checkpoint_skip/21_04/Fx",compile=False)
GiorgiaGAN.Gz = keras.models.load_model("./checkpoint_skip/21_04/Gz",compile=False)
GiorgiaGAN.Dx = keras.models.load_model("./checkpoint_skip/21_04/Dx",compile=False)
GiorgiaGAN.Ds = keras.models.load_model("./checkpoint_skip/21_04/Ds",compile=False)
GiorgiaGAN.Dn = keras.models.load_model("./checkpoint_skip/21_04/Dn",compile=False)
GiorgiaGAN.Dc = keras.models.load_model("./checkpoint_skip/21_04/Dc",compile=False)
GiorgiaGAN.Q  = keras.models.load_model("./checkpoint_skip/21_04/Q",compile=False)
GiorgiaGAN.Gq = keras.models.load_model("./checkpoint_skip/21_04/Gq",compile=False)


GiorgiaGAN.build(input_shape=(options['batchSize'], options['Xsize'], options['nXchannels']))

#load_status = GiorgiaGAN.load_weights("ckpt")

#latest = tf.train.latest_checkpoint(checkpoint_dir)
#print('restoring model from ' + latest)
#GiorgiaGAN.load_weights(latest)
#initial_epoch = int(latest[len(checkpoint_dir) + 7:])
GiorgiaGAN.summary()

if options['CreateData']:
    # Create the dataset
    Xtrn,  Xvld, _ = mdof.CreateData(**options)
else:
    # Load the dataset
    Xtrn, Xvld, _ = mdof.LoadData(**options)

PlotReconstructedTHs(GiorgiaGAN,Xvld) # Plot reconstructed time-histories

PlotTHSGoFs(GiorgiaGAN,Xvld) # Plot reconstructed time-histories

PlotClassificationMetrics(GiorgiaGAN,Xvld) # Plot classification metrics

PlotLatentSpace(GiorgiaGAN,Xvld)

PlotTSNE(GiorgiaGAN,Xvld)

PlotChangeS(GiorgiaGAN,Xvld)

PlotDistributions(GiorgiaGAN,Xvld)

Xtrn = {}
Xvld = {}
for i in range(options['latentCdim']):
    Xtrn['Xtrn_%d' % i], Xvld['Xvld_%d' % i], _  = mdof.Load_Un_Damaged(i,**options)

for i in range(options['latentCdim']):
    PlotBatchGoFs(GiorgiaGAN,Xtrn['Xtrn_%d' % i],Xvld['Xvld_%d' % i],i)

for i in range(1,options['latentCdim']):
    PlotSwitchedTHs(GiorgiaGAN,Xvld['Xvld_%d' % 0],Xvld['Xvld_%d' % i],i) # Plot switched time-histories

