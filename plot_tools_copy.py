import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
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
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# # for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })
from matplotlib.pyplot import *
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
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
from bokeh.io import curdoc
import bokeh
from bokeh.models import Text, Label


from random import seed
from random import randint

import matplotlib.font_manager
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Helvetica']
#families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
#rcParams['text.usetex'] = True

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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
    loss1[r"$Domainloss$"] = history.history['Domainloss']
    
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

    
    #hax.plot(history.history['AdvDlossClass'], linewidth=2,label=r"$AdvDlossClass$", color = 'violet')
    #hax.plot(history.history['AdvGlossClass'], linewidth=2,label=r"$AdvGlossClass$", color = 'palegreen')
    #hax.plot(history.history['AdvDloss'], color='b')
    #hax.plot(history.history['AdvGloss'], color='g')
    # hax.plot(history.history['AdvDlossX'],linewidth=2) #color='r',linewidth=2)
    # hax.plot(history.history['AdvDlossC'],linewidth=2) #color='c',linewidth=2)
    # hax.plot(history.history['AdvDlossS'],linewidth=2)#color='m',linewidth=2)
    # hax.plot(history.history['AdvDlossN'],linewidth=2) #color='gold',linewidth=2)
    # hax.plot(history.history['AdvDlossPenGradX'])
    # hax.plot(history.history['AdvGlossX'],linewidth=2) #color = 'purple',linewidth=2)
    # hax.plot(history.history['AdvGlossC'],linewidth=2) #color = 'brown',linewidth=2)
    # hax.plot(history.history['AdvGlossS'],linewidth=2) #color = 'salmon',linewidth=2)
    # hax.plot(history.history['AdvGlossN'],linewidth=2) #color = 'lightblue',linewidth=2)
    # hax.plot(history.history['RecGlossX'],linewidth=2) #color='darkorange',linewidth=2)
    # hax.plot(history.history['RecGlossC'],linewidth=2) #color='lime',linewidth=2)
    # hax.plot(history.history['RecGlossS'],linewidth=2) #color='grey',linewidth=2)
    #hax.set_title('Model loss', fontsize=14)
    #hax.set_ylabel('Loss', fontsize=14)
    #hax.set_xlabel('Epoch', fontsize=14)
    labels_legend0= [r"$\mathcal{L}_{AdvX}$",r"$\mathcal{L}_{AdvC}$",r"$\mathcal{L}_{AdvS}$",r"$\mathcal{L}_{AdvN}$",
        r"$\mathcal{L}_{GenX}$",r"$\mathcal{L}_{GenC}$"]#,r"$\mathcal{L}_{AdvClass}$",r"$\mathcal{L}_{GenClass}$"
    ax0.legend(labels_legend0,fontsize=18,frameon=False,loc='upper right')
    ax0.tick_params(axis='both', labelsize=18)
    ax0.set_xlabel(r"$n_{epochs}$",fontsize=20,fontweight='bold')
    ax0.set_ylabel(r'$Loss \hspace{0.5} [1]$',fontsize=20,fontweight='bold')

    labels_legend1= [r"$\mathcal{L}_{GenS}$",r"$\mathcal{L}_{GenN}$",r"$\mathcal{L}_{RecX}$",
        r"$\mathcal{L}_{RecC}$",r"$\mathcal{L}_{RecS}$",r"$\mathcal{L}_{Domain}$"] #r"$\mathcal{L}_{AdvAll}$",r"$\mathcal{L}_{GenAll}$"
    ax1.legend(labels_legend1,fontsize=18,frameon=False,loc='upper right')
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_xlabel(r"$n_{epochs}$",fontsize=20,fontweight='bold')
    ax1.set_ylabel(r'$Loss \hspace{0.5} [1]$',fontsize=20,fontweight='bold')

    plt.savefig('./results/loss.png',format='png',bbox_inches = 'tight')
    #plt.savefig('./results/loss.eps',format='eps',rasterized=True,bbox_inches = 'tight',dpi=200)
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
    plt.savefig('./results/D_x.png',bbox_inches = 'tight')
    #plt.savefig('./results/D_x.eps',bbox_inches = 'tight',dpi=200)
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
    plt.savefig('./results/D_c.png',bbox_inches = 'tight')
    #plt.savefig('./results/D_c.eps',bbox_inches = 'tight',dpi=200)
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
    plt.savefig('./results/D_s.png',bbox_inches = 'tight')
    #plt.savefig('./results/D_s.eps',bbox_inches = 'tight',dpi=200)
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
    plt.savefig('./results/D_n.png',bbox_inches = 'tight')
    #plt.savefig('./results/D_n.eps',bbox_inches = 'tight',dpi=200)
    plt.close()



def PlotReconstructedTHs(model,realXC):
    # Plot reconstructed time-histories
    realX, realC = realXC
    realX = np.concatenate([x for x, c, d in realXC], axis=0)
    realC = np.concatenate([c for x, c, d in realXC], axis=0)
    realD = np.concatenate([d for x, c, d in realXC], axis=0)
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
            plt.savefig('./results/reconstruction_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
            #plt.savefig('./results/reconstruction_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
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
            plt.savefig('./results/fft_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
            #plt.savefig('./results/fft_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

    

def PlotSwitchedTHs(model,realSwitch):
    # Plot reconstructed time-histories
    realXC = {}
    realX = {}
    realC = {}
    realD = {}
    recX = {}
    fakeC_new = {}
    fakeX_new = {}

    for k in range(options['latentCdim']):
        realXC['realXC_%d' % k] = tf.unstack(realSwitch,axis=0)[k]
        realX['realX_%d' % k], realC['realC_%d' % k] = realXC['realXC_%d' % k]
        realX['realX_%d' % k] = np.concatenate([x for x, c, d in realXC['realXC_%d' % k]], axis=0)
        realC['realC_%d' % k] = np.concatenate([c for x, c, d in realXC['realXC_%d' % k]], axis=0)
        realD['realD_%d' % k] = np.concatenate([d for x, c, d in realXC['realXC_%d' % k]], axis=0)
        recX['recX_%d' % k],_,_,_,_ = model.predict(realX['realX_%d' % k])

        t = np.zeros(realX['realX_%d' % k].shape[1])
        for m in range(realX['realX_%d' % k].shape[1]-1):
            t[m+1] = (m+1)*0.04

        recX_fft = tf.make_ndarray(tf.make_tensor_proto(recX['recX_%d' % k]))

        # Print real vs reconstructed signal
        for j in range(realX['realX_%d' % k].shape[2]):
            for k in range(10):
                i = randint(0, realX['realX_%d' % k].shape[0]-1)
                hfg = plt.figure(figsize=(12,6),tight_layout=True)
                hax = hfg.add_subplot(111)
                hax.plot(t,realX['realX_%d' % k][i,:,j], color='black')
                hax.plot(t,recX['recX_%d' % k][i,:,j], color='orange',linestyle="--")
                #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
                hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
                hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
                hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
                hax.set_ylim([-1.0, 1.0])
                hax.tick_params(axis='both', labelsize=18)
                plt.savefig('./results/reconstruction{:>d}_{:>d}_{:>d}.png'.format(k,j,i),bbox_inches = 'tight')
                #plt.savefig('./results/reconstruction{:>d}_{:>d}_{:>d}.eps'.format(k,j,i),bbox_inches = 'tight',dpi=200)
                plt.close()


                hfg = plt.figure(figsize=(12,6),tight_layout=True)
                hax = hfg.add_subplot(111)
                N = realX['realX_%d' % k].shape[1]
                SAMPLE_RATE = 25
                yf_real = rfft(realX['realX_%d' % k][i,:,j])
                xf_real = rfftfreq(N, 1 / SAMPLE_RATE)
                hax.plot(xf_real, np.abs(yf_real), color='black')
                yf_rec = rfft(recX_fft[i,:,j])
                xf_rec = rfftfreq(N, 1 / SAMPLE_RATE)
                hax.plot(xf_rec, np.abs(yf_rec), color='orange',linestyle="--")
                hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
                hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
                hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
                hax.tick_params(axis='both', labelsize=18)
                plt.savefig('./results/fft{:>d}_{:>d}_{:>d}.png'.format(k,j,i),bbox_inches = 'tight')
                #plt.savefig('./results/fft{:>d}_{:>d}_{:>d}.eps'.format(k,j,i),bbox_inches = 'tight',dpi=200)
                plt.close()

    for k in range(1,options['latentCdim']):
        fakeC_new['fakeC_new_%d' % k] = np.zeros_like(realC['realC_%d' % k])
        fakeC_new['fakeC_new_%d' % k][:,k] = 1.0
        fakeX_new['fakeX_new_%d' % k] = model.generate(realX['realX_%d' % 0],fakeX_new['fakeC_new_%d' % k])
        fakeX_new_fft = tf.make_ndarray(tf.make_tensor_proto(fakeX_new['fakeX_new_%d' % k]))

        t = np.zeros(realX['realX_%d' % 0].shape[1])
        for m in range(realX['realX_%d' % 0].shape[1]-1):
            t[m+1] = (m+1)*0.04

        for j in range(realX['realX_%d' % 0].shape[2]):
            for i in range(realX['realX_%d' % 0].shape[0]):
                #i = randint(0, realX_u.shape[0]-1)
                fig, (ax0, ax1, ax2) = plt.subplots(3, 1,figsize=(12,18))
                ax0.plot(t,realX['realX_%d' % 0][i,:,j], color='green')
                ax1.plot(t,realX['realX_%d' % k][i,:,j], color='black')
                ax2.plot(t,fakeX_new['fakeX_new_%d' % k][i,:,j], color='orange')
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
                plt.savefig('./results/reconstruction_switch{:>d}_{:>d}_{:>d}.png'.format(k,j,i),bbox_inches = 'tight')
                plt.savefig('./results/reconstruction_switch{:>d}_{:>d}_{:>d}.eps'.format(k,j,i),bbox_inches = 'tight',dpi=200)
                plt.close()

                hfg = plt.figure(figsize=(12,6),tight_layout=True)
                hax = hfg.add_subplot(111)
                N = realX['realX_%d' % 0].shape[1]
                SAMPLE_RATE = 25
                yf_real_d = rfft(realX['realX_%d' % k][i,:,j])
                xf_real_d = rfftfreq(N, 1 / SAMPLE_RATE)
                hax.plot(xf_real_d, np.abs(yf_real_d), color='black')
                yf_switch = rfft(fakeX_new_fft[i,:,j])
                xf_switch = rfftfreq(N, 1 / SAMPLE_RATE)
                hax.plot(xf_switch, np.abs(yf_switch), color='orange',linestyle="--")
                hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
                hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
                hax.legend([r'$X_d$', r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
                hax.tick_params(axis='both', labelsize=18)
                plt.savefig('./results/fft_switch{:>d}_{:>d}_{:>d}.png'.format(k,j,i),bbox_inches = 'tight')
                plt.savefig('./results/fft_switch{:>d}_{:>d}_{:>d}.eps'.format(k,j,i),bbox_inches = 'tight',dpi=200)
                plt.close()

                hfg = plt.figure(figsize=(12,6),tight_layout=True)
                hax = hfg.add_subplot(111)
                hax.plot(t,realX['realX_%d' % k][i,:,j], color='black')
                hax.plot(t,fakeX_new['fakeX_new_%d' % k][i,:,j], color='orange',linestyle="--")
                hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
                hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
                hax.legend([r'$X_d$',r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
                hax.tick_params(axis='both', labelsize=18)
                hax.set_ylim([-1.0, 1.0])           

                plt.savefig('./results/switch{:>d}_{:>d}_{:>d}.png'.format(k,j,i),bbox_inches = 'tight')
                plt.savefig('./results/switch{:>d}_{:>d}_{:>d}.eps'.format(k,j,i),bbox_inches = 'tight',dpi=200)
                plt.close()

     
    # realX_u = np.concatenate([x for x, c in realXC], axis=0)
    # realC_u = np.concatenate([c for x, c in realXC], axis=0)
    # #fakeC_new = np.ones_like(realC_u)
    # #fakeC_new[:,1] = 1.0
    # #fakeX_new = model.generate(realX_u,fakeC_new)

    # fakeX_u,_,_,_ = model.predict(realX_u)

    # # hfg = plt.figure(figsize=(12,6),tight_layout=True)
    # # hax = hfg.add_subplot(111)
    # # hax.plot(t,realX_u[0,:,0], color='black')
    # # hax.plot(t,fakeX_u[0,:,0], color='orange')
    # # hax.set_ylim([-1.0, 1.0])
    # # #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    # # hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    # # hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    # # hax.legend([r'$X_u$', r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
    # # hax.tick_params(axis='both', labelsize=18)
    # # plt.savefig('./results/reconstruction_undamaged.png',bbox_inches = 'tight')
    # # plt.savefig('./results/reconstruction_undamaged.eps',bbox_inches = 'tight',dpi=200)
    # # plt.close()

    # realX_d1 = np.concatenate([x for x, c in realXC_d1], axis=0)
    # realC_d1 = np.concatenate([c for x, c in realXC_d1], axis=0)

    # fakeX_d1,_,_,_ = model.predict(realX_d1)

    # realX_d2 = np.concatenate([x for x, c in realXC_d2], axis=0)
    # realC_d2 = np.concatenate([c for x, c in realXC_d2], axis=0)

    # fakeX_d2,_,_,_ = model.predict(realX_d2)

    

    
    # for k in range(10):
    #     i = randint(0, realX_u.shape[0]-1)
    #     fig, axs = plt.subplots(2, 3,figsize=(40,15))
        
    #     axs[0, 0].plot(t,realX_u[i,:,0], color='black')
    #     axs[0, 0].plot(t,fakeX_u[i,:,0], color='green',linestyle="--")
    #     axs[1, 0].plot(t,realX_u[i,:,1], color='black')
    #     axs[1, 0].plot(t,fakeX_u[i,:,1], color='green',linestyle="--")
    #     # axs[2, 0].plot(t,realX_u[i,:,2], color='black')
    #     # axs[2, 0].plot(t,fakeX_u[i,:,2], color='green',linestyle="--")
    #     # axs[3, 0].plot(t,realX_u[i,:,3], color='black')
    #     # axs[3, 0].plot(t,fakeX_u[i,:,3], color='green',linestyle="--")
    #     axs[0, 1].plot(t,realX_d1[i,:,0], color='black')
    #     axs[0, 1].plot(t,fakeX_d1[i,:,0], color='red',linestyle="--")
    #     axs[1, 1].plot(t,realX_d1[i,:,1], color='black')
    #     axs[1, 1].plot(t,fakeX_d1[i,:,1], color='red',linestyle="--")
    #     # axs[2, 1].plot(t,realX_d1[i,:,2], color='black')
    #     # axs[2, 1].plot(t,fakeX_d1[i,:,2], color='red',linestyle="--")
    #     # axs[3, 1].plot(t,realX_d1[i,:,3], color='black')
    #     # axs[3, 1].plot(t,fakeX_d1[i,:,3], color='red',linestyle="--")
    #     axs[0, 2].plot(t,realX_d2[i,:,0], color='black')
    #     axs[0, 2].plot(t,fakeX_d2[i,:,0], color='orange',linestyle="--")
    #     axs[1, 2].plot(t,realX_d2[i,:,1], color='black')
    #     axs[1, 2].plot(t,fakeX_d2[i,:,1], color='orange',linestyle="--")
    #     # axs[2, 2].plot(t,realX_d2[i,:,2], color='black')
    #     # axs[2, 2].plot(t,fakeX_d2[i,:,2], color='orange',linestyle="--")
    #     # axs[3, 2].plot(t,realX_d2[i,:,3], color='black')
    #     # axs[3, 2].plot(t,fakeX_d2[i,:,3], color='orange',linestyle="--")
    #     axs[0, 0].set_ylim([-1.0, 1.0])
    #     axs[1, 0].set_ylim([-1.0, 1.0])
    #     # axs[2, 0].set_ylim([-1.0, 1.0])
    #     # axs[3, 0].set_ylim([-1.0, 1.0])
    #     axs[0, 1].set_ylim([-1.0, 1.0])
    #     axs[1, 1].set_ylim([-1.0, 1.0])
    #     # axs[2, 1].set_ylim([-1.0, 1.0])
    #     # axs[3, 1].set_ylim([-1.0, 1.0])
    #     axs[0, 2].set_ylim([-1.0, 1.0])
    #     axs[1, 2].set_ylim([-1.0, 1.0])
    #     # axs[2, 2].set_ylim([-1.0, 1.0])
    #     # axs[3, 2].set_ylim([-1.0, 1.0])
    #     #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    #     axs[0, 0].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     axs[0, 0].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     axs[0, 0].legend([r'$X_{u1}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     axs[0, 0].tick_params(axis='both', labelsize=18)
    #     axs[1, 0].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     axs[1, 0].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     axs[1, 0].legend([r'$X_{u2}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     axs[1, 0].tick_params(axis='both', labelsize=18)
    #     # axs[2, 0].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     # axs[2, 0].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     # axs[2, 0].legend([r'$X_{u3}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     # axs[2, 0].tick_params(axis='both', labelsize=18)
    #     # axs[3, 0].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     # axs[3, 0].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     # axs[3, 0].legend([r'$X_{u4}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     # axs[3, 0].tick_params(axis='both', labelsize=18)

    #     axs[0, 1].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     axs[0, 1].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     axs[0, 1].legend([r'$X_{d1}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     axs[0, 1].tick_params(axis='both', labelsize=18)
    #     axs[1, 1].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     axs[1, 1].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     axs[1, 1].legend([r'$X_{d2}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     axs[1, 1].tick_params(axis='both', labelsize=18)
    #     # axs[2, 1].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     # axs[2, 1].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     # axs[2, 1].legend([r'$X_{d3}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     # axs[2, 1].tick_params(axis='both', labelsize=18)
    #     # axs[3, 1].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     # axs[3, 1].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     # axs[3, 1].legend([r'$X_{d4}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     # axs[3, 1].tick_params(axis='both', labelsize=18)

    #     axs[0, 2].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     axs[0, 2].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     axs[0, 2].legend([r'$X_{d1}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     axs[0, 2].tick_params(axis='both', labelsize=18)
    #     axs[1, 2].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     axs[1, 2].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     axs[1, 2].legend([r'$X_{d2}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     axs[1, 2].tick_params(axis='both', labelsize=18)
    #     # axs[2, 2].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     # axs[2, 2].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     # axs[2, 2].legend([r'$X_{d3}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     # axs[2, 2].tick_params(axis='both', labelsize=18)
    #     # axs[3, 2].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #     # axs[3, 2].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #     # axs[3, 2].legend([r'$X_{d4}$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    #     # axs[3, 2].tick_params(axis='both', labelsize=18)

    #     plt.savefig('./results/signals_{:>d}.png'.format(i),bbox_inches = 'tight')
    #     #plt.savefig('./results/signals_{:>d}.eps'.format(i),bbox_inches = 'tight',dpi=200)
    #     plt.close()

    # fakeC_new1 = np.zeros_like(realC_u)
    # fakeC_new1[:,1] = 1.0
    # fakeX_new1 = model.generate(realX_u,fakeC_new1)
    # fakeX_new_fft1 = tf.make_ndarray(tf.make_tensor_proto(fakeX_new1))

    # fakeC_new2 = np.zeros_like(realC_u)
    # fakeC_new2[:,2] = 1.0
    # fakeX_new2 = model.generate(realX_u,fakeC_new2)
    # fakeX_new_fft2 = tf.make_ndarray(tf.make_tensor_proto(fakeX_new2))

    # t = np.zeros(realX_u.shape[1])
    # for k in range(realX_u.shape[1]-1):
    #     t[k+1] = (k+1)*0.04

    # for j in range(realX_u.shape[2]):
    #     for i in range(realX_u.shape[0]):
    #         #i = randint(0, realX_u.shape[0]-1)
    #         fig, (ax0, ax1, ax2) = plt.subplots(3, 1,figsize=(12,18))
    #         ax0.plot(t,realX_u[i,:,j], color='green')
    #         ax1.plot(t,realX_d1[i,:,j], color='black')
    #         ax2.plot(t,fakeX_new1[i,:,j], color='orange')
    #         #hfg = plt.subplots(3,1,figsize=(12,6),tight_layout=True)
    #         #hax = hfg.add_subplot(111)
    #         #hax.plot(t,realX_u[0,:,0],t,realX_d[0,:,0],t,fakeX_d[0,:,0])
    #         #hax.plot(t,fakeX_u[0,:,0], color='orange')
    #         ax0.set_ylim([-1.0, 1.0])
    #         ax1.set_ylim([-1.0, 1.0])
    #         ax2.set_ylim([-1.0, 1.0])
    #         #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    #         ax0.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         ax0.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         ax0.legend([r'$X_u$'], loc='best',frameon=False,fontsize=20)
    #         ax0.tick_params(axis='both', labelsize=18)
    #         ax1.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         ax1.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         ax1.legend([r'$X_d$'], loc='best',frameon=False,fontsize=20)
    #         ax1.tick_params(axis='both', labelsize=18)
    #         ax2.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         ax2.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         ax2.legend([r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
    #         ax2.tick_params(axis='both', labelsize=18)
    #         plt.savefig('./results/reconstruction_switch1_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.savefig('./results/reconstruction_switch1_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         N = realX_u.shape[1]
    #         SAMPLE_RATE = 25
    #         yf_real_d = rfft(realX_d1[i,:,j])
    #         xf_real_d = rfftfreq(N, 1 / SAMPLE_RATE)
    #         hax.plot(xf_real_d, np.abs(yf_real_d), color='black')
    #         yf_switch = rfft(fakeX_new_fft1[i,:,j])
    #         xf_switch = rfftfreq(N, 1 / SAMPLE_RATE)
    #         hax.plot(xf_switch, np.abs(yf_switch), color='orange',linestyle="--")
    #         hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
    #         hax.legend([r'$X_d$', r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('./results/fft_switch1_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.savefig('./results/fft_switch1_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(t,realX_d1[i,:,j], color='black')
    #         hax.plot(t,fakeX_new1[i,:,j], color='orange',linestyle="--")
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         hax.legend([r'$X_d$',r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
    #         hax.tick_params(axis='both', labelsize=18)
    #         hax.set_ylim([-1.0, 1.0])           

    #         plt.savefig('./results/switch_d1_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.savefig('./results/switch_d1_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(t,realX_u[i,:,j], color='black')
    #         hax.plot(t,fakeX_u[i,:,j], color='orange',linestyle="--")
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         hax.legend([r'$X_u$',r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
    #         hax.tick_params(axis='both', labelsize=18)
    #         hax.set_ylim([-1.0, 1.0])           

    #         plt.savefig('./results/switch_u_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.savefig('./results/switch_u_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()

    #         fig, (ax0, ax1, ax2) = plt.subplots(3, 1,figsize=(12,18))
    #         ax0.plot(t,realX_u[i,:,j], color='green')
    #         ax1.plot(t,realX_d2[i,:,j], color='black')
    #         ax2.plot(t,fakeX_new2[i,:,j], color='orange')
    #         #hfg = plt.subplots(3,1,figsize=(12,6),tight_layout=True)
    #         #hax = hfg.add_subplot(111)
    #         #hax.plot(t,realX_u[0,:,0],t,realX_d[0,:,0],t,fakeX_d[0,:,0])
    #         #hax.plot(t,fakeX_u[0,:,0], color='orange')
    #         ax0.set_ylim([-1.0, 1.0])
    #         ax1.set_ylim([-1.0, 1.0])
    #         ax2.set_ylim([-1.0, 1.0])
    #         #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    #         ax0.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         ax0.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         ax0.legend([r'$X_u$'], loc='best',frameon=False,fontsize=20)
    #         ax0.tick_params(axis='both', labelsize=18)
    #         ax1.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         ax1.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         ax1.legend([r'$X_d$'], loc='best',frameon=False,fontsize=20)
    #         ax1.tick_params(axis='both', labelsize=18)
    #         ax2.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         ax2.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         ax2.legend([r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
    #         ax2.tick_params(axis='both', labelsize=18)
    #         plt.savefig('./results/reconstruction_switch2_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.savefig('./results/reconstruction_switch2_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         N = realX_u.shape[1]
    #         SAMPLE_RATE = 25
    #         yf_real_d = rfft(realX_d2[i,:,j])
    #         xf_real_d = rfftfreq(N, 1 / SAMPLE_RATE)
    #         hax.plot(xf_real_d, np.abs(yf_real_d), color='black')
    #         yf_switch = rfft(fakeX_new_fft2[i,:,j])
    #         xf_switch = rfftfreq(N, 1 / SAMPLE_RATE)
    #         hax.plot(xf_switch, np.abs(yf_switch), color='orange',linestyle="--")
    #         hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
    #         hax.legend([r'$X_d$', r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
    #         hax.tick_params(axis='both', labelsize=18)
    #         plt.savefig('./results/fft_switch2_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.savefig('./results/fft_switch2_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(t,realX_d2[i,:,j], color='black')
    #         hax.plot(t,fakeX_new2[i,:,j], color='orange',linestyle="--")
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         hax.legend([r'$X_d$',r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
    #         hax.tick_params(axis='both', labelsize=18)
    #         hax.set_ylim([-1.0, 1.0])           

    #         plt.savefig('./results/switch_d2_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         plt.savefig('./results/switch_d2_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()

    





def PlotCorrelationS(model,realXC):
    # Plot s correlation
    #realX, realC = realXC
    realX = np.concatenate([x for x, c in realXC], axis=0)
    realC = np.concatenate([c for x, c in realXC], axis=0)
    fakeX,fakeC,fakeS,fakeN = model.predict(realX)

    # Print fakeS autocorrelation
    # S = np.reshape(fakeS, fakeS.size)
    # hfg = plt.figure(figsize=(12,6),tight_layout=True)
    # corr3 = hfg.add_subplot(111)
    # corr3.set_title(r"$Continuous \hspace{0.5} variables \hspace{0.5} S - Autocorrelation \hspace{0.5} Plot$", fontsize=22,fontweight='bold')
    # corr3.set_xlabel(r"$Lags \hspace{0.5} [1]$", fontsize=20,fontweight='bold')
    # corr3.set_ylabel(r"$Autocorrelation \hspace{0.5} [1]$", fontsize=20,fontweight='bold')
    # corr3.tick_params(axis='both', labelsize=14)
    # plt.acorr(S, maxlags = 20)
    # plt.savefig('./results/autocorrelation_fakeS.png',bbox_inches = 'tight')
    # #plt.savefig('./results/autocorrelation_fakeS.eps',bbox_inches = 'tight',dpi=200)
    # plt.close()

    # Print fakeS correlation matrix
    df = pd.DataFrame(fakeS)
    corr = df.corr()
    #corrMatrix = np.corrcoef(df)
    ax = sn.heatmap(corr, vmin=-1, vmax=1, center=0,
        cmap=sn.diverging_palette(20, 220, n=200), square=True)
    ax.set_rasterized(True)
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title(r"$Continuous \hspace{0.5} variables \hspace{0.5} S - Correlation \hspace{0.5} matrix$", fontsize=18,fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    plt.savefig('./results/correlation_matrix.png',bbox_inches = 'tight')
    #plt.savefig('./results/correlation_matrix.eps',rasterized=True,bbox_inches = 'tight',dpi=200)
    plt.close()

    # Print fakeS distribution
    #fakeS_std = np.std(fakeS)
    #fakeS_mean = np.mean(fakeS)

    # hfg = plt.figure(figsize=(12,6))
    # hax = hfg.add_subplot(111)
    # hax.set_rasterized(True)
    # x = np.linspace(0, 2.5, 1000) 
    # y1 = lognorm.pdf(x,1.0,loc=0.0) 
    # for i in range (fakeS.shape[0]):
    #     mu = np.mean(fakeS[i,:])
    #     sigma =np.std(fakeS[i,:])
    #     y2 = lognorm.pdf(x,sigma,loc=mu,scale=sigma)
    #     hfg = plt.figure(figsize=(12,6))
    #     hax = hfg.add_subplot(111)
    #     hax.set_rasterized(True)
    #     hax.plot(x,y1, linewidth=2, color='r', label=r'$PDF \mathcal{N} = (0,1)$')
    #     hax.plot(x,y2, linewidth=2, color='b', label=r'$PDF \hspace{0.5} Continuous \hspace{0.5} Variable \hspace{0.5} S$')
    #     plt.ylabel(r"$PDF  \hspace{0.5} [1]$", fontsize=20,fontweight='bold')
    #     plt.title(r"$Continuous \hspace{0.5} variables \hspace{0.5} S - Distribution$", fontsize=22,fontweight='bold')
    #     plt.tick_params(axis='both', labelsize=16)
    #     plt.legend(frameon=False)
    #     plt.savefig('./results/fakeS_{:>d}.png'.format(i),bbox_inches = 'tight')
    #     #plt.savefig('./results/fakeS_{:>d}.eps'.format(i),bbox_inches = 'tight',dpi=200)
    #     plt.close()

    # num_bins = 100
    # n, bins, patches = plt.hist(fakeS, num_bins, density=True, facecolor='blue', alpha=0.5)
    # # add a 'best fit' line
    # #y = ((1 / (np.sqrt(2 * np.pi) * fakeS_std)) *np.exp(-0.5 * (1 / fakeS_std * (bins - fakeS_mean))**2))
    # mean = 0
    # std = 0.5
    # variance = np.square(std)
    # #x = np.arange(-5,5,.01)
    # x = np.linspace(-5,5,101)
    # f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
    # #plt.set_rasterized(True)
    # plt.plot(bins,f,'r--',linewidth=2)

    # s_1 = np.random.lognormal(fakeS_mean, fakeS_std)
    # countS, binsS, ignoredS = plt.hist(s_1, 1000, density=True, align='mid')
    # xS = np.linspace(0.0, 2.5)
    # pdfS = (np.exp(-(np.log(xS) - fakeS_mean)**2 / (2 * fakeS_std**2))/ (xS * fakeS_std * np.sqrt(2 * np.pi)))

    # # Plot lognorm distribution
    # mu, sigma = 0., 1. # mean and standard deviation
    # s = np.random.lognormal(mu, sigma)
    # count, bins, ignored = plt.hist(s, 1000, density=True, align='mid')
    # x = np.linspace(0.0,2.5)
    # pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))/ (x * sigma * np.sqrt(2 * np.pi)))

    # x = np.linspace(0, 2.5, 1000) 
    # y2 = lognorm.pdf(x,fakeS_std,loc=fakeS_mean,scale=fakeS_std) 
    # y1 = lognorm.pdf(x,1.0,loc=0.0) 
    # #plt.plot(x, y1, "*", x, y2, "r--") 

    # hfg = plt.figure(figsize=(12,6))
    # hax = hfg.add_subplot(111)
    # hax.set_rasterized(True)
    # hax.plot(x,y1, linewidth=2, color='r', label=r'$PDF \mathcal{N} = (0,1)$')
    # hax.plot(x,y2, linewidth=2, color='b', label=r'$PDF \hspace{0.5} Continuous \hspace{0.5} Variable \hspace{0.5} S$')
    # plt.xlabel(r"$F(x)\vert_s \hspace{0.5} [1]$", fontsize=20,fontweight='bold')
    # plt.ylabel(r"$p(s\vert x) \hspace{0.5} [1]$", fontsize=20,fontweight='bold')
    # plt.title(r'$Continuous \hspace{0.5} variables \hspace{0.5} S - Distribution$', fontsize=22,fontweight='bold')
    # plt.tick_params(axis='both', labelsize=16)
    # plt.legend(frameon=False)
    # plt.savefig('./results/fakeS_distribution.png',bbox_inches = 'tight')
    # plt.savefig('./results/fakeS_distribution.eps',bbox_inches = 'tight',dpi=200)
    # plt.close()

def PlotDistributionN(model,realXC):
    # Plot n distribution
    #realX, realC = realXC
    realX = np.concatenate([x for x, c in realXC], axis=0)
    realC = np.concatenate([c for x, c in realXC], axis=0)
    fakeX,fakeC,fakeS,fakeN = model.predict(realX)
    # Print fakeN distribution
    # fakeN_std = np.std(fakeN)
    # fakeN_mean = np.mean(fakeN)
    # num_bins = 100
    # n, bins, patches = plt.hist(fakeN, num_bins, density=True, facecolor='blue', alpha=0.5)
    # add a 'best fit' line
    #y = ((1 / (np.sqrt(2 * np.pi) * fakeN_std)) *np.exp(-0.5 * (1 / fakeN_std * (bins - fakeN_mean))**2))
    #y = norm.pdf(bins, fakeN_mean, fakeN_std)
    # mean = 0
    # std = 0.3
    # variance = np.square(std)
    # #x = np.arange(-5,5,.01)
    # x = np.linspace(-5,5,101)
    # f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.set_rasterized(True)
    for i in range (fakeN.shape[0]):
        mu = np.mean(fakeN[i,:])
        sigma =np.std(fakeN[i,:])
        x = np.linspace(0, 2.5, 1000) 
        y = lognorm.pdf(x,sigma,loc=mu)
        hfg = plt.figure(figsize=(12,6))
        hax = hfg.add_subplot(111)
        hax.set_rasterized(True)
        hax.plot(x,y, linewidth=2, color='r',label=r'$PDF \hspace{0.5} Noise \hspace{0.5} \hspace{0.5} N$')
        plt.ylabel(r"$PDF  \hspace{0.5} [1]$", fontsize=20,fontweight='bold')
        plt.title(r"$Noise \hspace{0.5} N - Distribution$", fontsize=22,fontweight='bold')
        plt.tick_params(axis='both', labelsize=16)
        plt.legend(frameon=False)
        plt.savefig('./results/fakeN_{:>d}.png'.format(i),bbox_inches = 'tight')
        #plt.savefig('./results/fakeN_{:>d}.eps'.format(i),bbox_inches = 'tight',dpi=200)
        plt.close()

    # plt.plot(bins,f,'r--',linewidth=2)
    # plt.xlabel(r"$F(x)\vert_n \hspace{0.5} [1]$", fontsize=20,fontweight='bold')
    # plt.ylabel(r"$p(n\vert x) \hspace{0.5} [1]$", fontsize=20,fontweight='bold')
    # plt.title(r"$Noise \hspace{0.5} N - Distribution$", fontsize=22,fontweight='bold')
    # plt.tick_params(axis='both', labelsize=16)
    # plt.legend(frameon=False)
    # plt.savefig('./results/fakeN_distribution.png',bbox_inches = 'tight')
    # #plt.savefig('./results/fakeN_distribution.eps',bbox_inches = 'tight',dpi=200)
    # plt.close()


def PlotTHSGoFs(model,realXC):
    # Plot reconstructed time-histories
    #realX, realC = realXC
    realX = np.concatenate([x for x, c, d in realXC], axis=0)
    realC = np.concatenate([c for x, c, d in realXC], axis=0)
    realD = np.concatenate([d for x, c, d in realXC], axis=0)
    recX,fakeC,fakeS,fakeN = model.predict(realX)

    
    # a = np.zeros(realX.shape[1])
    # b = np.zeros(fakeX.shape[1])

    ## Print signal GoF
    for j in range(realX.shape[2]):
        for k in range(10):
            i = randint(0, realX.shape[0]-1)
            plot_tf_gofs(realX[i,:,j],recX[i,:,j],dt=0.04,fmin=0.1,fmax=30.0,t0=0.0,nf=100,w0=6,norm='global',st2_isref=True,
                a=10.,k=1.,left=0.1,bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2,w_1=0.2,w_2=0.6,w_cb=0.01, d_cb=0.0,show=False,
                plot_args=['k', 'r', 'b'],ylim=0., clim=0.)
            plt.savefig('./results/gof_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
            plt.savefig('./results/gof_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
            plt.close()


    
    # i = randint(0, realX.shape[0]-1)
    # j = randint(0,realX.shape[2]-1) 
    # plot_tf_gofs(realX[i,:,j],fakeX[i,:,j],dt=0.04,fmin=0.1,fmax=30.0,t0=0.0,nf=100,w0=6,norm='global',st2_isref=True,
    #     a=10.,k=1.,left=0.1,bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2,w_1=0.2,w_2=0.6,w_cb=0.01, d_cb=0.0,show=False,
    #     plot_args=['k', 'r', 'b'],ylim=0., clim=0.)
    # plt.savefig('./results/gof_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    # plt.savefig('./results/gof_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    # plt.close()

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
    plt.savefig('./results/Gz(Fx(X))_gofs_{:>d}.png'.format(i),bbox_inches = 'tight')
    #plt.savefig('./results/Gz(Fx(X))_gofs_{:>d}.eps'.format(i),bbox_inches = 'tight',dpi=200)
    plt.close()


def PlotBatchGoFs(model,Xtrn,Xvld,i):
    # Plot GoFs on a batch
    #realX, realC = realXC
    realX_trn = np.concatenate([x for x, c, d in Xtrn], axis=0)
    realC_trn = np.concatenate([c for x, c, d in Xtrn], axis=0)
    realD_trn = np.concatenate([d for x, c, d in Xtrn], axis=0)
    fakeX_trn,_,_,_ = model.predict(realX_trn)

    realX_vld = np.concatenate([x for x, c, d in Xvld], axis=0)
    realC_vld = np.concatenate([c for x, c, d in Xvld], axis=0)
    realD_vld = np.concatenate([d for x, c, d in Xvld], axis=0)
    fakeX_vld,_,_,_ = model.predict(realX_vld) 

    
    fakeC_new = np.zeros_like(realC_trn)
    fakeC_new[:,i] = 1.0
    fakeX_trn = model.generate(realX_trn,fakeC_new)

    fakeC_new = np.zeros_like(realC_vld)
    fakeC_new[:,i] = 1.0
    fakeX_vld = model.generate(realX_vld,fakeC_new)

    egpg_trn = {}
    for j in range(realX_trn.shape[2]):
        egpg_trn['egpg_trn_%d' % j] = np.zeros((realX_trn.shape[0],2),dtype=np.float32)
        st1 = np.zeros((realX_trn.shape[0],realX_trn.shape[1]))
        st2 = np.zeros((realX_trn.shape[0],realX_trn.shape[1]))
        for k in range(realX_trn.shape[0]):
            st1 = realX_trn[i,:,k]
            st2 = fakeX_trn[i,:,k]
            egpg_trn['egpg_trn_%d' % j][i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
            egpg_trn['egpg_trn_%d' % j][i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

    egpg_vld = {}
    for j in range(realX_vld.shape[2]):
        egpg_vld['egpg_vld_%d' % j] = np.zeros((realX_vld.shape[0],2),dtype=np.float32)
        st1 = np.zeros((realX_vld.shape[0],realX_vld.shape[1]))
        st2 = np.zeros((realX_vld.shape[0],realX_vld.shape[1]))
        for k in range(realX_vld.shape[0]):
            st1 = realX_vld[i,:,k]
            st2 = fakeX_vld[i,:,k]
            egpg_vld['egpg_vld_%d' % j][i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
            egpg_vld['egpg_vld_%d' % j][i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)


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

    egpg_df.to_csv('./results/EG_PG_{:>d}.csv'.format(i), index= True)
    PlotEGPGgrid('EG','PG','kind',i,df=egpg_df)

# def PlotBatchGoFs(model,Xtrn,Xvld,i):
#     # Plot GoFs on a batch
#     #realX, realC = realXC
#     realX_trn = np.concatenate([x for x, c in Xtrn], axis=0)
#     realC_trn = np.concatenate([c for x, c in Xtrn], axis=0)
#     fakeX_trn,_,_,_ = model.predict(realX_trn)

#     realX_vld = np.concatenate([x for x, c in Xvld], axis=0)
#     realC_vld = np.concatenate([c for x, c in Xvld], axis=0)
#     fakeX_vld,_,_,_ = model.predict(realX_vld) 

#     if n=='s':
#         fakeC_new = np.zeros_like(realC_trn)
#         fakeC_new[:,1] = 1.0
#         fakeX_trn = model.generate(realX_trn,fakeC_new)

#         fakeC_new = np.zeros_like(realC_vld)
#         fakeC_new[:,1] = 1.0
#         fakeX_vld = model.generate(realX_vld,fakeC_new)

#     if n=='t':
#         fakeC_new = np.zeros_like(realC_trn)
#         fakeC_new[:,2] = 1.0
#         fakeX_trn = model.generate(realX_trn,fakeC_new)

#         fakeC_new = np.zeros_like(realC_vld)
#         fakeC_new[:,2] = 1.0
#         fakeX_vld = model.generate(realX_vld,fakeC_new)


#     egpg_trn_0 = np.zeros((realX_trn.shape[0],2),dtype=np.float32)
#     egpg_trn_1 = np.zeros((realX_trn.shape[0],2),dtype=np.float32)
#     egpg_trn_2 = np.zeros((realX_trn.shape[0],2),dtype=np.float32)
#     egpg_trn_3 = np.zeros((realX_trn.shape[0],2),dtype=np.float32)
#     st1 = np.zeros((realX_trn.shape[0],realX_trn.shape[1]))
#     st2 = np.zeros((realX_trn.shape[0],realX_trn.shape[1]))

#     for i in range(realX_trn.shape[0]):
#         st1 = realX_trn[i,:,0]
#         st2 = fakeX_trn[i,:,0]
#         egpg_trn_0[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
#         egpg_trn_0[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

#     for i in range(realX_trn.shape[0]):
#         st1 = realX_trn[i,:,1]
#         st2 = fakeX_trn[i,:,1]
#         egpg_trn_1[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
#         egpg_trn_1[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

#     for i in range(realX_trn.shape[0]):
#         st1 = realX_trn[i,:,2]
#         st2 = fakeX_trn[i,:,2]
#         egpg_trn_2[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
#         egpg_trn_2[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

#     for i in range(realX_trn.shape[0]):
#         st1 = realX_trn[i,:,3]
#         st2 = fakeX_trn[i,:,3]
#         egpg_trn_3[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
#         egpg_trn_3[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

#     egpg_vld_0 = np.zeros((realX_vld.shape[0],2),dtype=np.float32)
#     egpg_vld_1 = np.zeros((realX_vld.shape[0],2),dtype=np.float32)
#     egpg_vld_2 = np.zeros((realX_vld.shape[0],2),dtype=np.float32)
#     egpg_vld_3 = np.zeros((realX_vld.shape[0],2),dtype=np.float32)
#     st1 = np.zeros((realX_vld.shape[0],realX_vld.shape[1]))
#     st2 = np.zeros((realX_vld.shape[0],realX_vld.shape[1]))

#     for i in range(realX_vld.shape[0]):
#         st1 = realX_vld[i,:,0]
#         st2 = fakeX_vld[i,:,0]
#         egpg_vld_0[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
#         egpg_vld_0[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

#     for i in range(realX_vld.shape[0]):
#         st1 = realX_vld[i,:,1]
#         st2 = fakeX_vld[i,:,1]
#         egpg_vld_1[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
#         egpg_vld_1[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

#     for i in range(realX_vld.shape[0]):
#         st1 = realX_vld[i,:,2]
#         st2 = fakeX_vld[i,:,2]
#         egpg_vld_2[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
#         egpg_vld_2[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

#     for i in range(realX_vld.shape[0]):
#         st1 = realX_vld[i,:,3]
#         st2 = fakeX_vld[i,:,3]
#         egpg_vld_3[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
#         egpg_vld_3[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

#     egpg_df_trn_0 = pd.DataFrame(egpg_trn_0,columns=['EG','PG'])
#     egpg_df_trn_0['kind']=r"$G_z(F_x(x)) \hspace{0.5} train$"

#     egpg_df_trn_1 = pd.DataFrame(egpg_trn_1,columns=['EG','PG'])
#     egpg_df_trn_1['kind']=r"$G_z(F_x(x)) \hspace{0.5} train$"

#     egpg_df_trn_2 = pd.DataFrame(egpg_trn_2,columns=['EG','PG'])
#     egpg_df_trn_2['kind']=r"$G_z(F_x(x)) \hspace{0.5} train$"

#     egpg_df_trn_3 = pd.DataFrame(egpg_trn_3,columns=['EG','PG'])
#     egpg_df_trn_3['kind']=r"$G_z(F_x(x)) \hspace{0.5} train$"

#     egpg_df_vld_0 = pd.DataFrame(egpg_vld_0,columns=['EG','PG'])
#     egpg_df_vld_0['kind']=r"$G_z(F_x(x)) \hspace{0.5} validation$"

#     egpg_df_vld_1 = pd.DataFrame(egpg_vld_1,columns=['EG','PG'])
#     egpg_df_vld_1['kind']=r"$G_z(F_x(x)) \hspace{0.5} validation$"

#     egpg_df_vld_2 = pd.DataFrame(egpg_vld_2,columns=['EG','PG'])
#     egpg_df_vld_2['kind']=r"$G_z(F_x(x)) \hspace{0.5} validation$"

#     egpg_df_vld_3 = pd.DataFrame(egpg_vld_3,columns=['EG','PG'])
#     egpg_df_vld_3['kind']=r"$G_z(F_x(x)) \hspace{0.5} validation$"

#     egpg_df = pd.concat([egpg_df_trn_0,egpg_df_trn_1,egpg_df_trn_2,egpg_df_trn_3,
#         egpg_df_vld_0,egpg_df_vld_1,egpg_df_vld_2,egpg_df_vld_3])

#     egpg_df.to_csv('./results/EG_PG_{:>s}.csv'.format(n), index= True)
#     PlotEGPGgrid('EG','PG','kind',n,df=egpg_df)




def PlotClassificationMetrics(model,realXC):
    # Plot classification metrics
    #realX, realC = realXC
    realX = np.concatenate([x for x, c, d in realXC], axis=0)
    realC = np.concatenate([c for x, c, d in realXC], axis=0)
    realD = np.concatenate([d for x, c, d in realXC], axis=0)
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
    damage = {}
    for i in range(options['latentCdim']):
        target_names.append(damage['damage class %d' % i])

    # target_names = ['undamaged', 'damage class 1', 'damage class 2'] 

    fig, ax = plt.subplots()
    report = classification_report(y_true = labels_real, y_pred = labels_fake,
            target_names=target_names,output_dict=True,zero_division=1)
    df = pd.DataFrame(report).transpose()
    df.to_csv('./results/Classification Report C.csv', index= True)
    cr = sn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, annot_kws={"size": 12})
    cr.tick_params(axis='both', labelsize=12)
    cr.set_yticklabels(cr.get_yticklabels(), rotation=0)
    plt.savefig('./results/classification_report_fakeC.png',bbox_inches = 'tight')
    #plt.savefig('./results/classification_report.eps',bbox_inches = 'tight',dpi=200)
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
    plt.savefig('./results/confusion_matrix_fakeC.png',bbox_inches = 'tight')
    #plt.savefig('./results/confusion_matrixC.eps',bbox_inches = 'tight',dpi=200)
    plt.close()


    fig, ax = plt.subplots()
    report = classification_report(y_true = labels_real, y_pred = labels_rec,
            target_names=target_names,output_dict=True,zero_division=1)
    df = pd.DataFrame(report).transpose()
    df.to_csv('./results/Classification Report recC.csv', index= True)
    cr = sn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, annot_kws={"size": 12})
    cr.tick_params(axis='both', labelsize=12)
    cr.set_yticklabels(cr.get_yticklabels(), rotation=0)
    plt.savefig('./results/classification_report_recC.png',bbox_inches = 'tight')
    #plt.savefig('./results/classification_reportrec.eps',bbox_inches = 'tight',dpi=200)
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
    plt.savefig('./results/confusion_matrix_recC.png',bbox_inches = 'tight')
    #plt.savefig('./results/confusion_matrixrecC.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    return

def PlotPredictor(model,realXC):
    realX = np.concatenate([x for x, c, d in realXC], axis=0)
    realC = np.concatenate([c for x, c, d in realXC], axis=0)
    realD = np.concatenate([d for x, c, d in realXC], axis=0)
    predD = model.domain_classifier(realX)

    labels_pred = np.zeros((predD.shape[0]))
    for i in range(predD.shape[0]):
        labels_pred[i] = np.argmax(predD[i,:])

    labels_real = np.zeros((realD.shape[0]))
    for i in range(realD.shape[0]):
        labels_real[i] = np.argmax(realD[i,:])

    labels_pred = labels_pred.astype(int)
    labels_real = labels_real.astype(int)

    target_names = ['source', 'target'] 

    fig, ax = plt.subplots()
    report = classification_report(y_true = labels_real, y_pred = labels_pred,
            target_names=target_names,output_dict=True,zero_division=1)
    df = pd.DataFrame(report).transpose()
    df.to_csv('./results/Classification Report D.csv', index= True)
    cr = sn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, annot_kws={"size": 12})
    cr.tick_params(axis='both', labelsize=12)
    cr.set_yticklabels(cr.get_yticklabels(), rotation=0)
    plt.savefig('./results/classification_report_domain.png',bbox_inches = 'tight')
    #plt.savefig('./results/classification_domain.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    conf_mat = confusion_matrix(labels_real, labels_pred)
    fig, ax = plt.subplots(figsize=(10,10),tight_layout=True)
    sn.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names,yticklabels=target_names, vmin=0, vmax=realD.shape[0],
        annot_kws={"size": 20})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(axis='both', labelsize=20)
    plt.ylabel("True class",fontsize=22,labelpad=10)
    plt.xlabel("Predicted class",fontsize=22,labelpad=10)
    plt.savefig('./results/confusion_matrix_domain.png',bbox_inches = 'tight')
    #plt.savefig('./results/confusion_matrix_domain.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    return


def ViolinPlot(model,realXC):
    realX = np.concatenate([x for x, c in realXC], axis=0)
    realC = np.concatenate([c for x, c in realXC], axis=0)
    fakeX,fakeC,_,_ = model.predict(realX)

    labels_fake = np.zeros((fakeC.shape[0]))
    for i in range(fakeC.shape[0]):
        labels_fake[i] = np.argmax(fakeC[i,:])

    labels_real = np.zeros((realC.shape[0]))
    for i in range(realC.shape[0]):
        labels_real[i] = np.argmax(realC[i,:])

    d = {'realC': labels_real, 'fakeC': labels_fake}
    df = pd.DataFrame(data=d)
    df.to_csv('./results/Violin plot.csv', index= True)

    fig = go.Figure()

    categorical = ['realC', 'fakeC']

    
    fig.add_trace(go.Violin(y=df['realC'],line_color='blue',legendgroup=r'$C$',
            name=r'$C$',box_visible=True,meanline_visible=True,points='all'))
    fig.add_trace(go.Violin(y=df['fakeC'],line_color='orange',legendgroup=r'$F_x(x)$',
            name=r'$F_x(x)$',box_visible=True,meanline_visible=True,points='all'))
    fig.update_yaxes(title_text="Classes")
    fig.update_layout(title_text="Categorical variables C - Violin plot")
    fig.write_image('./results/violinC.png')
    #fig.write_image("violinC",format='eps',width=700*200,height=500*200)
    
    return

def PlotPSD(realXC_u,realXC_d):
    realX_u = np.concatenate([x for x, c in realXC_u], axis=0)
    realC_u = np.concatenate([c for x, c in realXC_u], axis=0)
    realX_d = np.concatenate([x for x, c in realXC_d], axis=0)
    realC_d = np.concatenate([c for x, c in realXC_d], axis=0)
    freqs, psd = signal.welch(realX_u[0,:,0])
    plt.figure(figsize=(12, 6),tight_layout=True)
    plt.semilogx(freqs, psd)
    plt.title(r"$Power \hspace{0.5} Spectral \hspace{0.5} Density - Undamaged \hspace{0.5} signals$", fontsize=22,fontweight='bold')
    plt.xlabel(r"$Frequency \hspace{0.5} [Hz]$", fontsize=20,fontweight='bold')
    plt.ylabel(r"$Power \hspace{0.5} [dB/Hz]$", fontsize=20,fontweight='bold')
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(frameon=False)
    plt.savefig('./results/psd_undamaged.png',bbox_inches = 'tight')
    #plt.savefig('./results/psd_undamaged.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    freqs, psd = signal.welch(realX_d[0,:,0])
    plt.figure(figsize=(12, 6),tight_layout=True)
    plt.semilogx(freqs, psd)
    plt.title(r"$Power \hspace{0.5} Spectral \hspace{0.5} Density - Damaged \hspace{0.5} signals$", fontsize=22,fontweight='bold')
    plt.xlabel(r"$Frequency \hspace{0.5} [Hz]$", fontsize=20,fontweight='bold')
    plt.ylabel(r"$Power \hspace{0.5} [dB/Hz]$", fontsize=20,fontweight='bold')
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(frameon=False)
    plt.savefig('./results/psd_damaged.png',bbox_inches = 'tight')
    #plt.savefig('./results/psd_damaged.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

def SwarmPlot(model,realXC):
    realX = np.concatenate([x for x, c in realXC], axis=0)
    realC = np.concatenate([c for x, c in realXC], axis=0)
    fakeX,fakeC,_,_ = model.predict(realX)

    labels_fake = np.zeros((fakeC.shape[0]))
    for i in range(fakeC.shape[0]):
        labels_fake[i] = np.argmax(fakeC[i,:])

    labels_real = np.zeros((realC.shape[0]))
    for i in range(realC.shape[0]):
        labels_real[i] = np.argmax(realC[i,:])

    d = {r'$C$': labels_real, r'$F_x(x)$': labels_fake}
    df = pd.DataFrame(data=d)

    plt.figure(figsize=(12,6))
    ax = sn.swarmplot(data=df, dodge=True, palette='rocket')
    ax.legend([r'$C$', r'$F_x(x)$'], loc='best',frameon=False,fontsize=14)
    ax.set_ylabel(r'$Classes$', fontsize=20,fontweight='bold')
    ax.set_yticks([0,1])
    ax.set_yticklabels(['undamaged', 'damaged'],fontsize=20,rotation=45)
    ax.set_xticklabels([r'$C$', r'$F_x(x)$'],fontsize=20)
    #plt.title(r"$Categorical \hspace{0.5} variables \hspace{0.5} C - Swarm \hspace{0.5} Plot$",fontsize=16)
    plt.savefig('./results/swarm_plot_c.png',bbox_inches = 'tight')
    plt.savefig('./results/swarm_plot_c.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

class RepGAN_fwd_plot():
    def __init__(self, Xset, model):
        super().__init__()
        self.Xset_ref = Xset
        self.Xset = Xset.shuffle(model.nX, reshuffle_each_iteration=False).batch(model.batchSize)
        self.model = model

    # def on_epoch_end(self, epoch, logs={}): 
    # def on_batch_begin(self, epoch, logs={}): 

    def PlotGeneration(self, nBatch, nSample, Cvalue, Svalue, Sdim, Nvalue, Ndim, s_change, n_change, c_change,
                       cm_change):
        Xset_np = list(self.Xset_ref)  # sorted dataset on np format (all samples)

        iterator = iter(self.Xset)  # restart data iter
        data = iterator.get_next()

        for b in range(nBatch):
            data = iterator.get_next()
        realX, realC, idx = data

        # Filter
        # Fx output: Zmu, Zsigma, s, c, n
        _, _, recS, recC, recN = self.model.Q(realX, training=False)
        # import pdb
        # pdb.set_trace()
        # Z Tensor: s,n,c
        s_list = []
        for s_i in range(self.model.latentSdim):
            if s_i == Sdim and s_change:
                s_list.append(tf.fill(realX.shape[0], float(Svalue)))
                print("hi")
            else:
                s_list.append(tf.fill(realX.shape[0], 0.0))  # recS[nSample][s_i].numpy()))
        # import pdb
        # pdb.set_trace()
        s_tensor = tf.stack(s_list, axis=1)

        if n_change:
            n_list = []
            for n_i in range(self.model.latentNdim):
                if n_i == Ndim:
                    n_list.append(tf.fill(realX.shape[0], float(Nvalue)))
                else:
                    n_list.append(tf.fill(realX.shape[0], recN[nSample][n_i].numpy()))
            n_tensor = tf.stack(n_list, axis=1)
        else:
            n_tensor = recN

        if c_change:
            classes = tf.fill(realX.shape[0], Cvalue)
            c_tensor = tf.one_hot(classes, self.model.latentCdim)
        else:
            c_tensor = realC

        recX = self.model.Gz((realX, s_tensor, c_tensor, n_tensor), training=False)

        # Reference signal plot per batch
        # recover genetation id
        id_sample = idx[nSample].numpy()
        if c_change: #find reference siganl of requested class
            class_sample = Cvalue
        else: #used reference siganl of original class
            class_sample = tf.argmax(realC[nSample]).numpy()
        # get reference signal by position on Xset_np
        refX = Xset_np[id_sample + (len(Xset_np) // 2) * class_sample][0].numpy().squeeze()

        genX_sample = recX[nSample, :, 0].numpy()

        # fig, axs = plt.subplots()

        # import pdb
        # pdb.set_trace()

        fig = plot_tf_gofs(refX, genX_sample, dt=0.0146e-6, t0=0.0, fmin=0.1, fmax=1e8, show=False)
        plt.text(0.2, 12, 'original class:' + str(tf.argmax(realC[nSample]).numpy()))
        plt.text(0.2, 1, 'generated class:' + str(Cvalue))
        # plt.savefig('generated/c_{}-idx_{}-s_{}_{}-{}-ep_{}'.format(class_sample, id_sample,
        #                                                             str(s_i).replace('.', ''),
        #                                                             str(s_j).replace('.', ''),
        #                                                             cnt, epoch))
        # plt.close()

        # CONFUSION MATRIX PLOT Filter for batch
        labelC = tf.argmax(realC, axis=1)

        if not cm_change:
            predictC = tf.argmax(recC, axis=1)
            title = "Confusion matrix F(x)"
            # predictC_ = tf.argmax(c_, axis=1)
            # import pdb
            # pdb.set_trace()
            # cm_ = tf.math.confusion_matrix(labelC, predictC, num_classes=2)
            # fig_cm, axs = plt.subplots()
        else:
            _, _, _, c_, _ = self.model.Fx(recX, training=False)
            predictC = tf.argmax(c_, axis=1)
            title = "Confusion matrix F(G(s,c,n))"


        cm = tf.math.confusion_matrix(labelC, predictC, num_classes=self.model.latentCdim)
        fig_cm = plot_confusion_matrix(cm.numpy(), class_names=['TT', 'ALL'], title=title)

        z = Label(text='s{} = {:.4f}; c = {}, n{} = {:.2f}'.format(Sdim, 0.0, class_sample, Ndim, recN[Ndim][Ndim]))
        return fig, fig_cm, z.text 