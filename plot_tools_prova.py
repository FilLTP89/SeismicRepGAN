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

    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_prova/loss.png',format='png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_prova/loss.eps',format='eps',rasterized=True,bbox_inches = 'tight',dpi=200)
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
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_prova/D_x.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_prova/D_x.eps',bbox_inches = 'tight',dpi=200)
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
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_prova/D_c.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_prova/D_c.eps',bbox_inches = 'tight',dpi=200)
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
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_prova/D_s.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_prova/D_s.eps',bbox_inches = 'tight',dpi=200)
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
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_prova/D_n.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_prova/D_n.eps',bbox_inches = 'tight',dpi=200)
    plt.close()



