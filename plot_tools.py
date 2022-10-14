import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import statistics
import scipy
import itertools
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
from matplotlib import cm
from collections import OrderedDict
cmaps = OrderedDict()
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
from statistics import mean
from PIL import Image
import io

import matplotlib.font_manager
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Helvetica']

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

def PlotLoss(history,checkpoint_dir):

    loss = {}
    loss[r"$AdvDLoss$"] = history.history['AdvDLoss']
    loss[r"$AdvGLoss$"] = history.history['AdvGLoss']
    loss[r"$AdvDlossX$"] = history.history['AdvDlossX']
    loss[r"$AdvDlossC$"] = history.history['AdvDlossC']
    loss[r"$AdvDlossS$"] = history.history['AdvDlossS']
    loss[r"$AdvDlossN$"] = history.history['AdvDlossN']
    loss[r"$AdvGlossX$"] = history.history['AdvGlossX']
    loss[r"$AdvGlossC$"] = history.history['AdvGlossC']
    loss[r"$AdvGlossS$"] = history.history['AdvGlossS']
    loss[r"$AdvGlossN$"] = history.history['AdvGlossN']
    loss[r"$RecGlossX$"] = history.history['RecGlossX']
    loss[r"$RecGlossC$"] = history.history['RecGlossC']
    loss[r"$RecGlossS$"] = history.history['RecGlossS']
    loss[r"$Qloss$"] = history.history['Qloss']
    loss[r"$FakeCloss$"] = history.history['FakeCloss']

    clr = sn.color_palette("bright",len(loss.keys()))

    #  Categorical Data
    a = 3  # number of rows
    b = 5  # number of columns
    c = 1  # initialize plot counter

    fig = plt.figure(figsize=(50,18))

    i = 0
    for k,v in loss.items():
        plt.subplot(a, b, c)
        plt.title('{}'.format(k))
        plt.xlabel(r"$n_{epochs}$")
        plt.ylabel(r'$Loss \hspace{0.5} [1]$')
        plt.plot(range(len(v)),v,linewidth=2,label=r"{}".format(k),color=clr[i])
        plt.tight_layout()
        c = c + 1
        i+=1

    plt.tight_layout()
    plt.savefig(checkpoint_dir + '/loss.png',bbox_inches = 'tight')
    plt.close()

    
    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.set_rasterized(True)
    hax.plot(history.history['fakeX'],linewidth=2,color='r',marker='^', linestyle='', label=r'$D_x(G_z(s,c,n))$')
    hax.plot(history.history['X'],linewidth=2,color='b',marker='s', linestyle='', label=r'$D_x(x)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_x$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_x \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=20)
    hax.tick_params(axis='both', labelsize=18)
    plt.savefig(checkpoint_dir + '/D_x.png',bbox_inches = 'tight')
    #plt.savefig(self.checkpoint_dir + '/D_x.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.plot(history.history['c_fake'],linewidth=2, color='r', marker='^', linestyle='', label=r'$D_c(F_x(x))$')
    hax.plot(history.history['c'],linewidth=2, color='b', marker='s', linestyle='', label=r'$D_c(c)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_c$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_c \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=20)
    hax.tick_params(axis='both', labelsize=18)
    plt.savefig(checkpoint_dir + '/D_c.png',bbox_inches = 'tight')
    #plt.savefig(self.checkpoint_dir + '/D_c.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.plot(history.history['s_fake'],linewidth=2, color='r', marker='^', linestyle='', label=r'$D_s(F_x(x))$')
    hax.plot(history.history['s_prior'],linewidth=2, color='b', marker='s', linestyle='', label=r'$D_s(s)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_s$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_s \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=20)
    hax.tick_params(axis='both', labelsize=18)
    plt.savefig(checkpoint_dir + '/D_s.png',bbox_inches = 'tight')
    #plt.savefig(self.checkpoint_dir + '/D_s.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.plot(history.history['n_fake'],linewidth=2, color='r', marker='^', linestyle='', label=r'$D_n(F_x(x))$')
    hax.plot(history.history['n_prior'],linewidth=2, color='b', marker='s', linestyle='', label=r'$D_n(n)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_n$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_n \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=20)
    hax.tick_params(axis='both', labelsize=18)
    plt.savefig(checkpoint_dir + '/D_n.png',bbox_inches = 'tight')
    #plt.savefig(self.checkpoint_dir + '/D_n.eps',bbox_inches = 'tight',dpi=200)
    plt.close()
