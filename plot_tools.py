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
from sklearn.utils import shuffle
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
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.io import curdoc,output_file, show
import bokeh
from bokeh.models import Text, Label
import panel as pn
pn.extension()


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
    realX_s = np.concatenate([xs for xs, cs, ds, xt, dt in realXC], axis=0)
    realC_s = np.concatenate([cs for xs, cs, ds, xt, dt in realXC], axis=0)
    realD_s = np.concatenate([ds for xs, cs, ds, xt, dt in realXC], axis=0)

    realX_t = np.concatenate([xt for xs, cs, ds, xt, dt in realXC], axis=0)
    realD_t = np.concatenate([dt for xs, cs, ds, xt, dt in realXC], axis=0)

    realX = np.concatenate((realX_s, realX_t))

    recX,fakeC,fakeS,fakeN,fakeX = model.plot(realX_s,realC_s,realX)

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


def PlotSwitchedTHs(model,real_u,real_d,d):
    # Plot reconstructed time-histories
    
    realX_s_u = np.concatenate([xs for xs, cs, ds, xt, dt in real_u], axis=0)
    realC_s_u = np.concatenate([cs for xs, cs, ds, xt, dt in real_u], axis=0)
    realD_s_u = np.concatenate([ds for xs, cs, ds, xt, dt in real_u], axis=0)

    realX_t_u = np.concatenate([xt for xs, cs, ds, xt, dt in real_u], axis=0)
    realD_t_u = np.concatenate([dt for xs, cs, ds, xt, dt in real_u], axis=0)

    realX_u = np.concatenate((realX_s_u,realX_t_u))

    recX_u,_,_,_ = model.predict(realX_u)

    realX_s_d = np.concatenate([xs for xs, cs, ds, xt, dt in real_d], axis=0)
    realC_s_d = np.concatenate([cs for xs, cs, ds, xt, dt in real_d], axis=0)
    realD_s_d = np.concatenate([ds for xs, cs, ds, xt, dt in real_d], axis=0)

    realX_t_d = np.concatenate([xt for xs, cs, ds, xt, dt in real_d], axis=0)
    realD_t_d = np.concatenate([dt for xs, cs, ds, xt, dt in real_d], axis=0)

    realX_d = np.concatenate((realX_s_d,realX_t_d))

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
                plt.savefig('./results/reconstruction0_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
                #plt.savefig('./results/reconstruction0_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
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
                plt.savefig('./results/fft0_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
                #plt.savefig('./results/fft0_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
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
            plt.savefig('./results/reconstruction{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results/reconstruction{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
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
            plt.savefig('./results/fft{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results/fft{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

    
    fakeC_new = np.zeros_like((realC_s_d))
    fakeC_new[:,d] = 1.0
    fakeX_new = model.generate(realX_s_u,fakeC_new)
    fakeX_new_fft = tf.make_ndarray(tf.make_tensor_proto(fakeX_new))

    t = np.zeros(realX_u.shape[1])
    for m in range(realX_u.shape[1]-1):
        t[m+1] = (m+1)*0.04

    for j in range(realX_s_u.shape[2]):
        for i in range(realX_s_u.shape[0]):
            #i = randint(0, realX_u.shape[0]-1)
            fig, (ax0, ax1, ax2) = plt.subplots(3, 1,figsize=(12,18))
            ax0.plot(t,realX_s_u[i,:,j], color='green')
            ax1.plot(t,realX_s_d[i,:,j], color='black')
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
            plt.savefig('./results/reconstruction_switch{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results/reconstruction_switch{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            N = realX_s_u.shape[1]
            SAMPLE_RATE = 25
            yf_real_d = rfft(realX_s_d[i,:,j])
            xf_real_d = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_real_d, np.abs(yf_real_d), color='black')
            yf_switch = rfft(fakeX_new_fft[i,:,j])
            xf_switch = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_switch, np.abs(yf_switch), color='orange',linestyle="--")
            hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X_d$', r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig('./results/fft_switch{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results/fft_switch{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            hax.plot(t,realX_s_d[i,:,j], color='black')
            hax.plot(t,fakeX_new[i,:,j], color='orange',linestyle="--")
            hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X_d$',r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
            hax.tick_params(axis='both', labelsize=18)
            hax.set_ylim([-1.0, 1.0])           

            plt.savefig('./results/switch{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig('./results/switch{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()


def PlotTHSGoFs(model,realXC):
    # Plot reconstructed time-histories
    #realX, realC = realXC
    realX_s = np.concatenate([xs for xs, cs, ds, xt, dt in realXC], axis=0)
    realC_s = np.concatenate([cs for xs, cs, ds, xt, dt in realXC], axis=0)
    realD_s = np.concatenate([ds for xs, cs, ds, xt, dt in realXC], axis=0)

    realX_t = np.concatenate([xt for xs, cs, ds, xt, dt in realXC], axis=0)
    realD_t = np.concatenate([dt for xs, cs, ds, xt, dt in realXC], axis=0)

    realX = np.concatenate((realX_s, realX_t))
    recX,fakeC,fakeS,fakeN = model.predict(realX)

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

    realX_s_trn = np.concatenate([xs for xs, cs, ds, xt, dt in Xtrn], axis=0)
    realC_s_trn = np.concatenate([cs for xs, cs, ds, xt, dt in Xtrn], axis=0)
    realD_s_trn = np.concatenate([ds for xs, cs, ds, xt, dt in Xtrn], axis=0)

    realX_t_trn = np.concatenate([xt for xs, cs, ds, xt, dt in Xtrn], axis=0)
    realD_t_trn = np.concatenate([dt for xs, cs, ds, xt, dt in Xtrn], axis=0)

    realX_trn = np.concatenate((realX_s_trn,realX_t_trn))

    fakeX_trn,_,_,_ = model.predict(realX_trn)

    realX_s_vld = np.concatenate([xs for xs, cs, ds, xt, dt in Xvld], axis=0)
    realC_s_vld = np.concatenate([cs for xs, cs, ds, xt, dt in Xvld], axis=0)
    realD_s_vld = np.concatenate([ds for xs, cs, ds, xt, dt in Xvld], axis=0)

    realX_t_vld = np.concatenate([xt for xs, cs, ds, xt, dt in Xvld], axis=0)
    realD_t_vld = np.concatenate([dt for xs, cs, ds, xt, dt in Xvld], axis=0)

    realX_vld = np.concatenate((realX_s_vld,realX_t_vld))

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

    egpg_df.to_csv('./results/EG_PG_{:>d}.csv'.format(i), index= True)
    PlotEGPGgrid('EG','PG','kind',i,df=egpg_df)

def PlotClassificationMetrics(model,realXC):
    # Plot classification metrics
    realX_s = np.concatenate([xs for xs, cs, ds, xt, dt in realXC], axis=0)
    realC_s = np.concatenate([cs for xs, cs, ds, xt, dt in realXC], axis=0)
    realD_s = np.concatenate([ds for xs, cs, ds, xt, dt in realXC], axis=0)

    realX_t = np.concatenate([xt for xs, cs, ds, xt, dt in realXC], axis=0)
    realD_t = np.concatenate([dt for xs, cs, ds, xt, dt in realXC], axis=0)

    fakeC, recC = model.label_predictor(realX_s,realC_s)

    labels_fake = np.zeros((fakeC.shape[0]))
    for i in range(fakeC.shape[0]):
        labels_fake[i] = np.argmax(fakeC[i,:])

    labels_rec = np.zeros((recC.shape[0]))
    for i in range(recC.shape[0]):
        labels_rec[i] = np.argmax(recC[i,:])
    
    labels_real = np.zeros((realC_s.shape[0]))
    for i in range(realC_s.shape[0]):
        labels_real[i] = np.argmax(realC_s[i,:])

    labels_fake = labels_fake.astype(int)
    labels_rec = labels_rec.astype(int)
    labels_real = labels_real.astype(int)

    target_names = []
    for i in range(realC_s.shape[1]):
        target_names.append('damage class %d'.format(i)) 

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
    sn.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names,yticklabels=target_names, vmin=0, vmax=realC_s.shape[0],
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
    sn.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names,yticklabels=target_names, vmin=0, vmax=realC_s.shape[0],
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
    realX_s = np.concatenate([xs for xs, cs, ds, xt, dt in realXC], axis=0)
    realC_s = np.concatenate([cs for xs, cs, ds, xt, dt in realXC], axis=0)
    realD_s = np.concatenate([ds for xs, cs, ds, xt, dt in realXC], axis=0)

    realX_t = np.concatenate([xt for xs, cs, ds, xt, dt in realXC], axis=0)
    realD_t = np.concatenate([dt for xs, cs, ds, xt, dt in realXC], axis=0)

    realX = np.concatenate((realX_s,realX_t))
    realD = np.concatenate((realD_s,realD_t))
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

def PlotGeneration(model, realX, realC, nBatch, nSample, Cvalue, Svalue, Sdim, Nvalue, Ndim, s_change, n_change, c_change,
                       cm_change,batchSize):
        
        
        recX,fakeC,fakeS,fakeN = model.predict(realX)

        s_list = []
        for s_i in range(latentSdim):
            if s_i == Sdim and s_change:
                s_list.append(tf.fill(realX.shape[0], float(Svalue)))
                print("hi")
            else:
                s_list.append(tf.fill(realX.shape[0], 0.0))  # recS[nSample][s_i].numpy()))
        

        s_tensor = tf.stack(s_list, axis=1)

        if n_change:
            n_list = []
            for n_i in range(latentNdim):
                if n_i == Ndim:
                    n_list.append(tf.fill(realX.shape[0], float(Nvalue)))
                else:
                    n_list.append(tf.fill(realX.shape[0], fakeN[nSample][n_i].numpy()))
            n_tensor = tf.stack(n_list, axis=1)
        else:
            n_tensor = fakeN

        if c_change:
            classes = tf.fill(realX.shape[0], Cvalue)
            c_tensor = tf.one_hot(classes, latentCdim)
        else:
            c_tensor = realC

        
        # Reference signal plot per batch

        if c_change: #find reference siganl of requested class
            class_sample = Cvalue
        else: #used reference siganl of original class
            class_sample = tf.argmax(realC[nSample]).numpy()


        fig = plot_tf_gofs(realX[nSample, :, 0],recX[nSample, :, 0],dt=0.04,fmin=0.1,fmax=30.0,t0=0.0,nf=100,w0=6,norm='global',st2_isref=True,
            a=10.,k=1.,left=0.1,bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2,w_1=0.2,w_2=0.6,w_cb=0.01, d_cb=0.0,show=False,
            plot_args=['k', 'r', 'b'],ylim=0., clim=0.)
        plt.text(0.2, 12, 'original class:' + str(tf.argmax(realC[nSample]).numpy()))
        plt.text(0.2, 1, 'generated class:' + str(Cvalue))
        

        z = Label(text='s{} = {:.4f}; c = {}, n{} = {:.2f}'.format(Sdim, 0.0, class_sample, Ndim, fakeN[Ndim][Ndim]))
        return fig, z.text

# def PlotBokeh(model,realXC,**kwargs):
#     PlotBokeh.__globals__.update(kwargs)

#     realX_s = np.concatenate([xs for xs, cs, ds, xt, dt in realXC], axis=0)
#     realC_s = np.concatenate([cs for xs, cs, ds, xt, dt in realXC], axis=0)
#     realD_s = np.concatenate([ds for xs, cs, ds, xt, dt in realXC], axis=0)

#     realX_t = np.concatenate([xt for xs, cs, ds, xt, dt in realXC], axis=0)
#     realD_t = np.concatenate([dt for xs, cs, ds, xt, dt in realXC], axis=0)

#     realX = np.concatenate((realX_s,realX_t))
#     realD = np.concatenate((realD_s,realD_t))

#     batch_select = pn.widgets.IntSlider(value=0, start=0, end=(len(realX_s)//batchSize - 1),
#                                             name='Batch index')
#     ex_select = pn.widgets.IntSlider(value=0, start=0, end=(batchSize - 1), name='Example index on batch')
#     # select_plot = pn.widgets.Select(name='Select dataset', options=['Reconstruct', 'Generate'])
#     s_dim_select = pn.widgets.IntSlider(value=0, start=0, end=latentSdim - 1, step=1, name='Sdim to modify')
#     n_dim_select = pn.widgets.IntSlider(value=0, start=0, end=latentNdim - 1, step=1, name='Ndim to modify')
#     c_select = pn.widgets.IntSlider(value=0, start=0, end=latentCdim - 1, step=1, name='Class to generate')
#     s_val_select = pn.widgets.FloatSlider(value=0.0, start=-3.0, end=3.0, step=0.01, name='S value')
#     n_val_select = pn.widgets.FloatSlider(value=0.0, start=-3.0, end=3.0, step=0.01, name='N value')
#     s_change = pn.widgets.Checkbox(name='Change S')
#     n_change = pn.widgets.Checkbox(name='Change N')
#     c_change = pn.widgets.Checkbox(name='Change C')
#     cm_change = pn.widgets.Checkbox(name=r"$F_x(G_z(z))$")



#     @pn.depends(batch_select=batch_select, ex_select=ex_select,
#                 s_dim_select=s_dim_select, n_dim_select=n_dim_select, c_select=c_select,
#                 s_val_select=s_val_select, n_val_select=n_val_select,
#                 s_change=s_change, n_change=n_change, c_change=c_change, cm_change_val=cm_change)
#     def image(batch_select, ex_select,
#               s_dim_select, n_dim_select, c_select,
#               s_val_select, n_val_select, s_change, n_change, c_change, cm_change_val):
#         print(s_val_select,s_dim_select,n_change)
#         fig1, z = PlotGeneration(model,realX_s,realC_s,batch_select, ex_select, c_select,
#                                                s_val_select, s_dim_select,
#                                                n_val_select, n_dim_select, s_change,
#                                                n_change, c_change, cm_change_val,batchSize)
#         fig1.set_size_inches(8, 5)
#         figArray = pn.Column(pn.Row(fig1, pn.Column(cm_change)), z)

#         return figArray
    
#     pn.panel(pn.Column(pn.Row(pn.Column(batch_select,
#                                         ex_select,
#                                         s_dim_select,
#                                         n_dim_select),
#                               pn.Column(pn.Row(c_select, c_change),
#                                         pn.Row(s_val_select, s_change),
#                                         pn.Row(n_val_select, n_change))),
#                        image)).servable(title='Plot RepGAN')
