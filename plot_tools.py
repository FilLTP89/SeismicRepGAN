import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import statistics
import scipy
from scipy import integrate
from scipy import signal
from scipy.stats import norm,lognorm
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
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1,figsize=(18,27))
    #hfg = plt.figure(figsize=(12,6))
    #hax = hfg.add_subplot(111)
    ax0.set_rasterized(True)
    ax1.set_rasterized(True)
    ax2.set_rasterized(True)
    loss0 = {}
    loss1 = {}
    loss2 = {}
    loss0[r"$AdvDlossX$"] = history.history['AdvDlossX']
    loss0[r"$AdvDlossC$"] = history.history['AdvDlossC']
    loss0[r"$AdvDlossS$"] = history.history['AdvDlossS']
    loss0[r"$AdvDlossN$"] = history.history['AdvDlossN']
    loss0[r"$AdvGlossX$"] = history.history['AdvGlossX']
    loss1[r"$AdvGlossC$"] = history.history['AdvGlossC']
    loss1[r"$AdvGlossS$"] = history.history['AdvGlossS']
    loss1[r"$AdvGlossN$"] = history.history['AdvGlossN']
    loss1[r"$RecGlossX$"] = history.history['RecGlossX']
    loss1[r"$RecGlossC$"] = history.history['RecGlossC']
    loss2[r"$RecGlossS$"] = history.history['RecGlossS']
    loss1[r"$AdvDlossClass$"] = history.history['AdvDlossClass']
    #loss1[r"$AdvGlossClass$"] = history.history['AdvGlossClass']
    
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

    clr = sn.color_palette("bright",len(loss2.keys()))
    i=0
    for k,v in loss2.items():
        ax2.plot(range(len(v)),v,linewidth=2,label=r"{}".format(k),color=clr[i])
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
        r"$\mathcal{L}_{GenX}$"]#,r"$\mathcal{L}_{AdvClass}$",r"$\mathcal{L}_{GenClass}$"
    ax0.legend(labels_legend0,fontsize=18,frameon=False,loc='upper right')
    ax0.tick_params(axis='both', labelsize=14)
    ax0.set_xlabel(r"$n_{epochs}$",fontsize=20,fontweight='bold')
    ax0.set_ylabel(r'$Loss \hspace{0.5} [1]$',fontsize=20,fontweight='bold')

    labels_legend1= [r"$\mathcal{L}_{GenC}$",r"$\mathcal{L}_{GenS}$",r"$\mathcal{L}_{GenN}$",r"$\mathcal{L}_{RecX}$",
        r"$\mathcal{L}_{RecC}$",r"$\mathcal{L}_{AdvClass}$"]
    ax1.legend(labels_legend1,fontsize=18,frameon=False,loc='upper right')
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel(r"$n_{epochs}$",fontsize=20,fontweight='bold')
    ax1.set_ylabel(r'$Loss \hspace{0.5} [1]$',fontsize=20,fontweight='bold')

    labels_legend2= [r"$\mathcal{L}_{RecS}$"]
    ax2.legend(labels_legend2,fontsize=18,frameon=False,loc='upper right')
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlabel(r"$n_{epochs}$",fontsize=20,fontweight='bold')
    ax2.set_ylabel(r'$Loss \hspace{0.5} [1]$',fontsize=20,fontweight='bold')


    #hax.set_title(r'$Model \hspace{0.5} Loss$',fontsize=22,fontweight='bold')
    #plt.legend(['AdvDlossX','AdvDlossC','AdvDlossS','AdvDlossN','AdvGlossX',
    #    'AdvGlossC','AdvGlossS','AdvGlossN','RecGlossX','RecGlossC','RecGlossS'], loc='lower right')
    #plt.tight_layout()
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/loss.png',format='png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/loss.eps',format='eps',rasterized=True,bbox_inches = 'tight',dpi=200)
    plt.close()

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.set_rasterized(True)
    hax.plot(history.history['fakeX'],linewidth=2,color='r',marker='^', linestyle='', label=r'$D_x(G_z(s,c,n))$')
    hax.plot(history.history['realX'],linewidth=2,color='b',marker='s', linestyle='', label=r'$D_x(x)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_x$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_x \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=12)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/D_x.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/D_x.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.plot(history.history['fakeC'],linewidth=2, color='r', marker='^', linestyle='', label=r'$D_c(F_x(x))$')
    hax.plot(history.history['realC'],linewidth=2, color='b', marker='s', linestyle='', label=r'$D_c(c)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_c$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_c \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=12)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/D_c.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/D_c.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.plot(history.history['fakeS'],linewidth=2, color='r', marker='^', linestyle='', label=r'$D_s(F_x(x))$')
    hax.plot(history.history['realS'],linewidth=2, color='b', marker='s', linestyle='', label=r'$D_s(s)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_s$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_s \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=12)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/D_s.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/D_s.eps',bbox_inches = 'tight',dpi=200)
    #plt.close()

    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.plot(history.history['fakeN'],linewidth=2, color='r', marker='^', linestyle='', label=r'$D_n(F_x(x))$')
    hax.plot(history.history['realN'],linewidth=2, color='b', marker='s', linestyle='', label=r'$D_n(n)$')
    #hax.set_title(r'$Discriminator \hspace{0.5} D_n$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$D_n \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    hax.set_xlabel(r'$n_{epochs}$', fontsize=20,fontweight='bold')
    hax.legend(loc='best',frameon=False,fontsize=12)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/D_n.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/D_n.eps',bbox_inches = 'tight',dpi=200)
    plt.close()



def PlotReconstructedTHs(model,realXC,realXC_u,realXC_d):
    # Plot reconstructed time-histories
    #realX, realC = realXC
    realX = np.concatenate([x for x, c in realXC], axis=0)
    realC = np.concatenate([c for x, c in realXC], axis=0)
    fakeX,fakeC,fakeS,fakeN = model.predict(realX)

    t = np.zeros(realX.shape[1])
    for k in range(realX.shape[1]-1):
        t[k+1] = (k+1)*0.04

    # # Print real vs reconstructed signal
    # for j in range(2):
    #     for i in range(realX.shape[0]):
    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(t,realX[i,:,j], color='black')
    #         hax.plot(t,fakeX[i,:,j], color='orange')
    #         #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=20,fontweight='bold')
    #         hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=20,fontweight='bold')
    #         hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=12)
    #         hax.set_ylim([-1.0, 1.0])
    #         hax.tick_params(axis='both', labelsize=14)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/reconstruction_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/reconstruction_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()

    i = randint(0, realX.shape[0]-1)
    j = randint(0,1) 
    hfg = plt.figure(figsize=(12,6),tight_layout=True)
    hax = hfg.add_subplot(111)
    hax.plot(t,realX[i,:,j], color='black')
    hax.plot(t,fakeX[i,:,j], color='orange')
    #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
    hax.set_ylim([-1.0, 1.0])
    hax.tick_params(axis='both', labelsize=18)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/reconstruction_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')

    

    realX_u = np.concatenate([x for x, c in realXC_u], axis=0)
    realC_u = np.concatenate([c for x, c in realXC_u], axis=0)
    #fakeC_new = np.ones_like(realC_u)
    #fakeC_new[:,1] = 1.0
    #fakeX_new = model.generate(realX_u,fakeC_new)

    fakeX_u,_,_,_ = model.predict(realX_u)

    hfg = plt.figure(figsize=(12,6),tight_layout=True)
    hax = hfg.add_subplot(111)
    hax.plot(t,realX_u[0,:,0], color='black')
    hax.plot(t,fakeX_u[0,:,0], color='orange')
    hax.set_ylim([-1.0, 1.0])
    #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    hax.legend([r'$X_u$', r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
    hax.tick_params(axis='both', labelsize=18)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/reconstruction_undamaged.png',bbox_inches = 'tight')
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/reconstruction_undamaged.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    realX_d = np.concatenate([x for x, c in realXC_d], axis=0)
    realC_d = np.concatenate([c for x, c in realXC_d], axis=0)

    fakeX_d,_,_,_ = model.predict(realX_d)

    # Print real vs reconstructed signal
    hfg = plt.figure(figsize=(12,6),tight_layout=True)
    hax = hfg.add_subplot(111)
    hax.plot(t,realX_d[0,:,0], color='black')
    hax.plot(t,fakeX_d[0,:,0], color='orange')
    hax.set_ylim([-1.0, 1.0])
    #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
    hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    hax.legend([r'$X_d$', r"$G_z(F_x(x_d))$"], loc='best',frameon=False,fontsize=20)
    hax.tick_params(axis='both', labelsize=18)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/reconstruction_damaged.png',bbox_inches = 'tight')
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/reconstruction_damaged.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    fakeC_new = np.zeros_like(realC_u)
    fakeC_new[:,1] = 1.0
    fakeX_new = model.generate(realX_u,fakeC_new)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1,figsize=(12,18))
    ax0.plot(t,realX_u[0,:,0], color='green')
    ax1.plot(t,realX_d[0,:,0], color='black')
    ax2.plot(t,fakeX_new[0,:,0], color='orange')
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
    ax2.legend([r"$G_z(F_x(x_d))$"], loc='best',frameon=False,fontsize=20)
    ax2.tick_params(axis='both', labelsize=18)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/reconstruction_switch.png',bbox_inches = 'tight')
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/reconstruction_switch.eps',bbox_inches = 'tight',dpi=200)
    plt.close()





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
    # plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/autocorrelation_fakeS.png',bbox_inches = 'tight')
    # #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/autocorrelation_fakeS.eps',bbox_inches = 'tight',dpi=200)
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
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/correlation_matrix.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/correlation_matrix.eps',rasterized=True,bbox_inches = 'tight',dpi=200)
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
    #     plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/fakeS_{:>d}.png'.format(i),bbox_inches = 'tight')
    #     #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/fakeS_{:>d}.eps'.format(i),bbox_inches = 'tight',dpi=200)
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
    # plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/fakeS_distribution.png',bbox_inches = 'tight')
    # plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/fakeS_distribution.eps',bbox_inches = 'tight',dpi=200)
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
        plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/fakeN_{:>d}.png'.format(i),bbox_inches = 'tight')
        #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/fakeN_{:>d}.eps'.format(i),bbox_inches = 'tight',dpi=200)
        plt.close()

    # plt.plot(bins,f,'r--',linewidth=2)
    # plt.xlabel(r"$F(x)\vert_n \hspace{0.5} [1]$", fontsize=20,fontweight='bold')
    # plt.ylabel(r"$p(n\vert x) \hspace{0.5} [1]$", fontsize=20,fontweight='bold')
    # plt.title(r"$Noise \hspace{0.5} N - Distribution$", fontsize=22,fontweight='bold')
    # plt.tick_params(axis='both', labelsize=16)
    # plt.legend(frameon=False)
    # plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/fakeN_distribution.png',bbox_inches = 'tight')
    # #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/fakeN_distribution.eps',bbox_inches = 'tight',dpi=200)
    # plt.close()


def PlotTHSGoFs(model,realXC):
    # Plot reconstructed time-histories
    #realX, realC = realXC
    realX = np.concatenate([x for x, c in realXC], axis=0)
    realC = np.concatenate([c for x, c in realXC], axis=0)
    fakeX,fakeC,fakeS,fakeN = model.predict(realX)

    
    # a = np.zeros(realX.shape[1])
    # b = np.zeros(fakeX.shape[1])

    # ## Print signal GoF
    # for j in range(2):
    #     for i in range(realX.shape[0]):
    #         plot_tf_gofs(realX[i,:,j],fakeX[i,:,j],dt=0.04,fmin=0.1,fmax=30.0,t0=0.0,nf=100,w0=6,norm='global',st2_isref=True,
    #             a=10.,k=1.,left=0.1,bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2,w_1=0.2,w_2=0.6,w_cb=0.01, d_cb=0.0,show=False,
    #             plot_args=['k', 'r', 'b'],ylim=0., clim=0.)
    #         plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/gof_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    #         #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/gof_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()


    
    i = randint(0, realX.shape[0]-1)
    j = randint(0,1) 
    plot_tf_gofs(realX[i,:,j],fakeX[i,:,j],dt=0.04,fmin=0.1,fmax=30.0,t0=0.0,nf=100,w0=6,norm='global',st2_isref=True,
        a=10.,k=1.,left=0.1,bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2,w_1=0.2,w_2=0.6,w_cb=0.01, d_cb=0.0,show=False,
        plot_args=['k', 'r', 'b'],ylim=0., clim=0.)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/gof_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/gof_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
    plt.close()

    # plot_tf_gofs(realX[0,:,0],fakeX[0,:,0],dt=0.04,t0=0.0,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,
    #     a=10.,k=1.,left=0.1,bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2,w_1=0.2,w_2=0.6,w_cb=0.01, d_cb=0.0,show=False,
    #     plot_args=['k', 'r', 'b'],ylim=0., clim=0.)
    # plt.tick_params(axis='both', labelsize=14)
    # plt.title(r"$Signal X - Goodness of Fit$", fontsize=22,fontweight='bold')
    # plt.savefig("gofs.png',bbox_inches = 'tight')
    # plt.savefig("gofs.eps',bbox_inches = 'tight',dpi=200)
    # plt.close()

# def colored_scatter(*args, **kwargs):
#     plt.scatter(*args, **kwargs)
#     return

def PlotEGPGgrid(col_x,col_y,col_k,df,k_is_color=False, scatter_alpha=.7):
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
    g = sn.JointGrid(x=col_x,y=col_y,data=df)
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
    plt.legend(legends,frameon=False)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/Gz(Fx(X))_gofs.png',bbox_inches = 'tight')
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/Gz(Fx(X))_gofs.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

def PlotEGPGgrid_new(col_x,col_y,col_k,df, k_is_color=False, scatter_alpha=.7):
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
    g = sn.JointGrid(x=col_x,y=col_y,data=df)
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
    plt.legend(legends)
    plt.title(r"$EG / PG \hspace{0.5} plot $", fontsize=22,fontweight='bold')
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/Gz(Fx(X))_gofs_new.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/Gz(Fx(X))_gofs_new.eps',bbox_inches = 'tight',dpi=200)
    plt.close()


    return 

def PlotBatchGoFs(model,Xtrn,Xvld):
    # Plot GoFs on a batch
    #realX, realC = realXC
    realX_trn = np.concatenate([x for x, c in Xtrn], axis=0)
    realC_trn = np.concatenate([c for x, c in Xtrn], axis=0)
    fakeX_trn,_,_,_ = model.predict(realX_trn)

    realX_vld = np.concatenate([x for x, c in Xvld], axis=0)
    realC_vld = np.concatenate([c for x, c in Xvld], axis=0)
    fakeX_vld,_,_,_ = model.predict(realX_vld)    
    
    egpg_trn_0 = np.zeros((realX_trn.shape[0],2),dtype=np.float32)
    egpg_trn_1 = np.zeros((realX_trn.shape[0],2),dtype=np.float32)
    st1 = np.zeros((realX_trn.shape[0],realX_trn.shape[1]))
    st2 = np.zeros((realX_trn.shape[0],realX_trn.shape[1]))

    for i in range(realX_trn.shape[0]):
        st1 = realX_trn[i,:,0]
        st2 = fakeX_trn[i,:,0]
        egpg_trn_0[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
        egpg_trn_0[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

    for i in range(realX_trn.shape[0]):
        st1 = realX_trn[i,:,1]
        st2 = fakeX_trn[i,:,1]
        egpg_trn_1[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
        egpg_trn_1[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

    # egpg_vld_0 = np.zeros((realX_vld.shape[0],2),dtype=np.float32)
    # egpg_vld_1 = np.zeros((realX_vld.shape[0],2),dtype=np.float32)
    # st1 = np.zeros((realX_vld.shape[0],realX_vld.shape[1]))
    # st2 = np.zeros((realX_vld.shape[0],realX_vld.shape[1]))

    # for i in range(realX_vld.shape[0]):
    #     st1 = realX_vld[i,:,0]
    #     st2 = fakeX_vld[i,:,0]
    #     egpg_vld_0[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
    #     egpg_vld_0[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

    # for i in range(realX_vld.shape[0]):
    #     st1 = realX_vld[i,:,1]
    #     st2 = fakeX_vld[i,:,1]
    #     egpg_vld_1[i,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
    #     egpg_vld_1[i,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

    egpg_df_trn_0 = pd.DataFrame(egpg_trn_0,columns=['EG','PG'])
    egpg_df_trn_0['kind']=r"$G_z(F_x(x))$"

    egpg_df_trn_1 = pd.DataFrame(egpg_trn_1,columns=['EG','PG'])
    egpg_df_trn_1['kind']=r"$G_z(F_x(x))$"

    # egpg_df_vld_0 = pd.DataFrame(egpg_vld_0,columns=['EG','PG'])
    # egpg_df_vld_0['kind']=r"$G_z(F_x(x)) \hspace{0.5} validation$"

    # egpg_df_vld_1 = pd.DataFrame(egpg_vld_1,columns=['EG','PG'])
    # egpg_df_vld_1['kind']=r"$G_z(F_x(x)) \hspace{0.5} validation$"

    egpg_df = pd.concat([egpg_df_trn_0,egpg_df_trn_1])

    egpg_df.to_csv('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/EG_PG.csv', index= True)
    PlotEGPGgrid('EG','PG','kind',df=egpg_df)

def PlotBatchGoFs_new(model,realXC_u,realXC_d):
    # Plot GoFs on a batch

    realX_u = np.concatenate([x for x, c in realXC_u], axis=0)
    realC_u = np.concatenate([c for x, c in realXC_u], axis=0)
    fakeC_new = np.zeros_like(realC_u)
    fakeC_new[:,1] = 1.0
    fakeX_new = model.generate(realX_u,fakeC_new)
    egpg = np.zeros((realX_u.shape[0],2),dtype=np.float32)

    realX_d = np.concatenate([x for x, c in realXC_d], axis=0)
    realC_d = np.concatenate([c for x, c in realXC_d], axis=0)

    for i in range(realX_u.shape[0]):
        st1 = np.squeeze((realX_d[i,:,:]))
        st2 = np.squeeze((fakeX_new[i,:,:]))
        egpg[i,0] = np.mean(eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.))
        egpg[i,1] = np.mean(pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.))
        # eg_1 = eg(realX[i,:,0],fakeX[i,:,0],dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
        # pg_1 = pg(realX[i,:,0],fakeX[i,:,0],dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
        # eg_2 = eg(realX[i,:,1],fakeX[i,:,1],dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
        # pg_2 = pg(realX[i,:,1],fakeX[i,:,1],dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
        # egpg[i,0] = (eg_1+eg_2)//2
        # egpg[i,1] = (pg_1+pg_2)//2
    egpg_df = pd.DataFrame(egpg,columns=['EG','PG'])
    egpg_df['kind']=r"$G_z(F_x(x))$"
    egpg_df.to_csv('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/EG_PG_new.csv', index= True)
    PlotEGPGgrid_new('EG','PG','kind',df=egpg_df)

def PlotClassificationMetrics(model,realXC):
    # Plot classification metrics
    #realX, realC = realXC
    realX = np.concatenate([x for x, c in realXC], axis=0)
    realC = np.concatenate([c for x, c in realXC], axis=0)
    fakeX,fakeC,fakeS,fakeN = model.predict(realX)

    labels_fake = np.zeros((fakeC.shape[0]))
    for i in range(fakeC.shape[0]):
        labels_fake[i] = np.argmax(fakeC[i,:])

    
    labels_real = np.zeros((realC.shape[0]))
    for i in range(realC.shape[0]):
        labels_real[i] = np.argmax(realC[i,:])

    target_names = ['class 0', 'class 1'] #, 'class 2', 'class 3', 'class 4'
    #report = classification_report(y_true = np.argmax(realC, axis=1), y_pred = np.argmax(fakeC, axis=1),
    #        target_names=target_names,output_dict=True)
    report = classification_report(y_true = labels_real, y_pred = labels_fake,
            target_names=target_names,output_dict=True,zero_division=1)
    df = pd.DataFrame(report).transpose()
    df.to_csv('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/Classification Report C.csv', index= True)
    cr = sn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1)
    cr.set_title(r"$Categorical \hspace{0.5} variables \hspace{0.5} C - Classification \hspace{0.5} report$", fontsize=14,fontweight='bold')
    #cr.set_rasterized(True)
    cr.tick_params(axis='both', labelsize=10)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/classification_report.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/classification_report.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    #conf_mat = confusion_matrix(realC.argmax(axis=1), fakeC.argmax(axis=1))
    conf_mat = confusion_matrix(labels_real, labels_fake)
    fig, ax = plt.subplots(figsize=(10,10),tight_layout=True)
    sn.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names,yticklabels=target_names, vmin=0, vmax=realC.shape[0])
    plt.tick_params(axis='both', labelsize=16)
    plt.ylabel(r"$C \hspace{0.5} [1]$",fontsize=20,fontweight='bold')
    plt.xlabel(r"$F_x(x) \hspace{0.5} [1]$",fontsize=20,fontweight='bold')
    plt.title("Categorical variables C - Confusion matrix", fontsize=22,fontweight='bold')
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/confusion_matrixC.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/confusion_matrixC.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    # multi = multilabel_confusion_matrix(labels_real, labels_fake)
    # fig, ax = plt.subplots(figsize=(10,10))
    # sn.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names,yticklabels=target_names)
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    #plt.title('Categorical variables C - Confusion matrix')
    # plt.savefig('multilabel_confusion_matrixC.png',bbox_inches = 'tight')
    # plt.close()

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
    df.to_csv('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/Violin plot.csv', index= True)

    fig = go.Figure()

    categorical = ['realC', 'fakeC']

    
    fig.add_trace(go.Violin(y=df['realC'],line_color='blue',legendgroup=r'$C$',
            name=r'$C$',box_visible=True,meanline_visible=True,points='all'))
    fig.add_trace(go.Violin(y=df['fakeC'],line_color='orange',legendgroup=r'$F_x(x)$',
            name=r'$F_x(x)$',box_visible=True,meanline_visible=True,points='all'))
    fig.update_yaxes(title_text="Classes")
    fig.update_layout(title_text="Categorical variables C - Violin plot")
    fig.write_image('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/violinC.png')
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
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/psd_undamaged.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/psd_undamaged.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    freqs, psd = signal.welch(realX_d[0,:,0])
    plt.figure(figsize=(12, 6),tight_layout=True)
    plt.semilogx(freqs, psd)
    plt.title(r"$Power \hspace{0.5} Spectral \hspace{0.5} Density - Damaged \hspace{0.5} signals$", fontsize=22,fontweight='bold')
    plt.xlabel(r"$Frequency \hspace{0.5} [Hz]$", fontsize=20,fontweight='bold')
    plt.ylabel(r"$Power \hspace{0.5} [dB/Hz]$", fontsize=20,fontweight='bold')
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(frameon=False)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/psd_damaged.png',bbox_inches = 'tight')
    #plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/psd_damaged.eps',bbox_inches = 'tight',dpi=200)
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
    ax.legend([r'$C$', r'$F_x(x)$'], loc='best',frameon=False,fontsize=12)
    ax.set_ylabel(r'$Classes$', fontsize=20,fontweight='bold')
    ax.set_yticks([0,1])
    ax.set_yticklabels(['undamaged', 'damaged'],fontsize=20,rotation=45)
    ax.set_xticklabels([r'$C$', r'$F_x(x)$'],fontsize=20)
    #plt.title(r"$Categorical \hspace{0.5} variables \hspace{0.5} C - Swarm \hspace{0.5} Plot$",fontsize=16)
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/swarm_plot_c.png',bbox_inches = 'tight')
    plt.savefig('/gpfs/workdir/invsem07/GiorgiaGAN/results_8/swarm_plot_c.eps',bbox_inches = 'tight',dpi=200)
    plt.close()
