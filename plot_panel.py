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
import subprocess

from RepGAN_ultimo_1 import RepGAN, ParseOptions, WassersteinDiscriminatorLoss, WassersteinGeneratorLoss, GaussianNLL
from tensorflow.keras.optimizers import Adam

checkpoint_dir = "/ckpt"


import MDOFload as mdof


from random import seed
from random import randint

import matplotlib.font_manager
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Helvetica']
#families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
#rcParams['text.usetex'] = True

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def PlotBokeh(model,realXC,**kwargs):
    PlotBokeh.__globals__.update(kwargs)

    subprocess.run("panel serve plot_panel.py --show", shell=True)


    realX_s = np.concatenate([xs for xs, cs, ds, xt, dt in realXC], axis=0)
    realC_s = np.concatenate([cs for xs, cs, ds, xt, dt in realXC], axis=0)
    realD_s = np.concatenate([ds for xs, cs, ds, xt, dt in realXC], axis=0)

    realX_t = np.concatenate([xt for xs, cs, ds, xt, dt in realXC], axis=0)
    realD_t = np.concatenate([dt for xs, cs, ds, xt, dt in realXC], axis=0)

    realX = np.concatenate((realX_s,realX_t))
    realD = np.concatenate((realD_s,realD_t))

    batch_select = pn.widgets.IntSlider(value=0, start=0, end=(len(realX_s)//batchSize - 1),
                                            name='Batch index')
    ex_select = pn.widgets.IntSlider(value=0, start=0, end=(batchSize - 1), name='Example index on batch')
    # select_plot = pn.widgets.Select(name='Select dataset', options=['Reconstruct', 'Generate'])
    s_dim_select = pn.widgets.IntSlider(value=0, start=0, end=latentSdim - 1, step=1, name='Sdim to modify')
    n_dim_select = pn.widgets.IntSlider(value=0, start=0, end=latentNdim - 1, step=1, name='Ndim to modify')
    c_select = pn.widgets.IntSlider(value=0, start=0, end=latentCdim - 1, step=1, name='Class to generate')
    s_val_select = pn.widgets.FloatSlider(value=0.0, start=-3.0, end=3.0, step=0.01, name='S value')
    n_val_select = pn.widgets.FloatSlider(value=0.0, start=-3.0, end=3.0, step=0.01, name='N value')
    s_change = pn.widgets.Checkbox(name='Change S')
    n_change = pn.widgets.Checkbox(name='Change N')
    c_change = pn.widgets.Checkbox(name='Change C')
    cm_change = pn.widgets.Checkbox(name=r"$F_x(G_z(z))$")



    @pn.depends(batch_select=batch_select, ex_select=ex_select,
                s_dim_select=s_dim_select, n_dim_select=n_dim_select, c_select=c_select,
                s_val_select=s_val_select, n_val_select=n_val_select,
                s_change=s_change, n_change=n_change, c_change=c_change, cm_change_val=cm_change)
    def image(batch_select, ex_select,
              s_dim_select, n_dim_select, c_select,
              s_val_select, n_val_select, s_change, n_change, c_change, cm_change_val):
        print(s_val_select,s_dim_select,n_change)
        fig1, z = PlotGeneration(model,realX_s,realC_s,batch_select, ex_select, c_select,
                                               s_val_select, s_dim_select,
                                               n_val_select, n_dim_select, s_change,
                                               n_change, c_change, cm_change_val,batchSize)
        fig1.set_size_inches(8, 5)
        figArray = pn.Column(pn.Row(fig1, pn.Column(cm_change)), z)

        return figArray
    
    pn.panel(pn.Column(pn.Row(pn.Column(batch_select,
                                        ex_select,
                                        s_dim_select,
                                        n_dim_select),
                              pn.Column(pn.Row(c_select, c_change),
                                        pn.Row(s_val_select, s_change),
                                        pn.Row(n_val_select, n_change))),
                       image)).servable(title='Plot RepGAN')
    
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

GiorgiaGAN.Fx = keras.models.load_model("Fx",compile=False)
GiorgiaGAN.Gz = keras.models.load_model("Gz",compile=False)
GiorgiaGAN.Dx = keras.models.load_model("Dx",compile=False)
GiorgiaGAN.Ds = keras.models.load_model("Ds",compile=False)
GiorgiaGAN.Dn = keras.models.load_model("Dn",compile=False)
GiorgiaGAN.Dc = keras.models.load_model("Dc",compile=False)
GiorgiaGAN.Q  = keras.models.load_model("Q",compile=False)
GiorgiaGAN.Gq = keras.models.load_model("Gq",compile=False)
GiorgiaGAN.h0 = keras.models.load_model("h0",compile=False)
GiorgiaGAN.h1 = keras.models.load_model("h1",compile=False)
GiorgiaGAN.h2 = keras.models.load_model("h2",compile=False)
GiorgiaGAN.h3 = keras.models.load_model("h3",compile=False)


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

PlotBokeh(GiorgiaGAN,Xvld) # Plot reconstructed time-histories