# TO EJECTUTE
# bokeh serve --show .\src_rmsp\plot_results.py
#from load_data import ut_data_loader
import MDOFload
from RepGAN_ultimo import GiorgiaGAN, ParseOptions, WassersteinDiscriminatorLoss, WassersteinGeneratorLoss, GaussianNLL
import itertools
from numpy.lib.type_check import imag
import sklearn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import io
from obspy.signal.tf_misfit import plot_tf_gofs, eg, pg
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import os

from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import CustomJS, Slider, ColumnDataSource
from bokeh.io import curdoc
import bokeh
from bokeh.models import Text, Label
import panel as pn


def class_generation(model, realX, class_to_gen):
    # import pdb 
    # pdb.set_trace()
    classes = tf.fill(realX.shape[0], class_to_gen)
    recX_new_class = model.generate_with_C(realX, classes)
    return recX_new_class


def plot_confusion_matrix(cm, class_names, title):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


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
        _, _, recS, recC, recN = self.model.Fx(realX, training=False)
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

        recX = self.model.Gz((s_tensor, c_tensor, n_tensor), training=False)

        # Reference signal plot per batch
        # recover genetation id
        id_sample = idx[nSample].numpy()
        if c_change: #find reference siganl of requested class
            class_sample = Cvalue
        else: #used reference siganl of original class
            class_sample = tf.argmax(realC[nSample]).numpy()
        # get reference signal by position on Xset_np
        refX = Xset_np[id_sample + (len(Xset_np) // 2) * class_sample][0].numpy().squeeze()

        genX_sample = genX[nSample, :, 0].numpy()

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

        # # CONFUSION MATRIX PLOT Filter for batch
        # labelC = tf.argmax(realC, axis=1)

        # if not cm_change:
        #     predictC = tf.argmax(recC, axis=1)
        #     title = "Confusion matrix F(x)"
        #     # predictC_ = tf.argmax(c_, axis=1)
        #     # import pdb
        #     # pdb.set_trace()
        #     # cm_ = tf.math.confusion_matrix(labelC, predictC, num_classes=2)
        #     # fig_cm, axs = plt.subplots()
        # else:
        #     _, _, _, c_, _ = self.model.Fx(genX, training=False)
        #     predictC = tf.argmax(c_, axis=1)
        #     title = "Confusion matrix F(G(s,c,n))"


        # cm = tf.math.confusion_matrix(labelC, predictC, num_classes=self.model.latentCdim)
        # fig_cm = plot_confusion_matrix(cm.numpy(), class_names=['TT', 'ALL'], title=title)

        z = Label(text='s{} = {:.4f}; c = {}, n{} = {:.2f}'.format(Sdim, 0.0, class_sample, Ndim, recN[Ndim][Ndim]))
        return fig, fig_cm, z.text


# BOKEH runs
checkpoint_dir = "ckpt_rmsp"
options = ParseOptions()

# MODEL LOADING
optimizers = {}

# WGAN
# optimizers['DxOpt'] = RMSprop(learning_rate=0.0001)
# optimizers['DcOpt'] = RMSprop(learning_rate=0.0001)
# optimizers['DsOpt'] = RMSprop(learning_rate=0.0001)
# optimizers['DnOpt'] = RMSprop(learning_rate=0.0001)
# optimizers['FxOpt'] = RMSprop(learning_rate=0.0001)
# optimizers['QOpt'] = RMSprop(learning_rate=0.0001)
# optimizers['GzOpt'] = RMSprop(learning_rate=0.0001)
# GAN
optimizers['DxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['DcOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['DsOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['DnOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['FxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
# optimizers['QOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)
optimizers['GzOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)

# optimizers['GqOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9999)

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
losses['PenRecXloss'] = 10.
losses['PenRecCloss'] = 1.
losses['PenRecSloss'] = 1.

# Instantiate the RepGAN model.
RepGAN_fdw = RepGAN(options)

# Compile the RepGAN model.
RepGAN_fdw.compile(optimizers, losses, losses)  # run_eagerly=True

# data structure
'''def train_step(self, realXC):

    realX, realC = realXC
'''
Xset = ut_data_loader(options)[0]

RepGAN_fdw.build(input_shape=(options['batchSize'], options['Xsize'], options['nXchannels']))

latest = tf.train.latest_checkpoint(checkpoint_dir)
print('restoring model from ' + latest)
RepGAN_fdw.load_weights(latest)
initial_epoch = int(latest[len(checkpoint_dir) + 7:])
RepGAN_fdw.summary()
RepGAN_fdw.Fx.trainable = False
RepGAN_fdw.Gz.trainable = False
RepGAN_fdw.Dx.trainable = False
RepGAN_fdw.Dc.trainable = False
RepGAN_fdw.Ds.trainable = False
RepGAN_fdw.Dn.trainable = False

plotter = RepGAN_fwd_plot(Xset, RepGAN_fdw)
#
# import pdb
# pdb.set_trace()

# BOKEH PANEL
# interaction sliders
batch_select = pn.widgets.IntSlider(value=0, start=0, end=(len(Xset) // RepGAN_fdw.batchSize) - 1,
                                    name='Batch index')
ex_select = pn.widgets.IntSlider(value=0, start=0, end=RepGAN_fdw.batchSize - 1, name='Example index on batch')
# select_plot = pn.widgets.Select(name='Select dataset', options=['Reconstruct', 'Generate'])
s_dim_select = pn.widgets.IntSlider(value=0, start=0, end=RepGAN_fdw.latentSdim - 1, step=1, name='Sdim to modify')
n_dim_select = pn.widgets.IntSlider(value=0, start=0, end=RepGAN_fdw.latentNdim - 1, step=1, name='Ndim to modify')
c_select = pn.widgets.IntSlider(value=0, start=0, end=RepGAN_fdw.latentCdim - 1, step=1, name='Class to generate')
s_val_select = pn.widgets.FloatSlider(value=0.0, start=-3.0, end=3.0, step=0.01, name='S value')
n_val_select = pn.widgets.FloatSlider(value=0.0, start=-3.0, end=3.0, step=0.01, name='N value')
s_change = pn.widgets.Checkbox(name='Change S')
n_change = pn.widgets.Checkbox(name='Change N')
c_change = pn.widgets.Checkbox(name='Change C')
cm_change = pn.widgets.Checkbox(name='F(G(s,c,n))')



@pn.depends(batch_select=batch_select, ex_select=ex_select,
            s_dim_select=s_dim_select, n_dim_select=n_dim_select, c_select=c_select,
            s_val_select=s_val_select, n_val_select=n_val_select,
            s_change=s_change, n_change=n_change, c_change=c_change, cm_change_val=cm_change)
def image(batch_select, ex_select,
          s_dim_select, n_dim_select, c_select,
          s_val_select, n_val_select, s_change, n_change, c_change, cm_change_val):
    # print(s_val_select,s_dim_select,n_change)
    fig1, fig2, z = plotter.PlotGeneration(batch_select, ex_select, c_select,
                                           s_val_select, s_dim_select,
                                           n_val_select, n_dim_select, s_change,
                                           n_change, c_change, cm_change_val)
    fig1.set_size_inches(8, 5)
    fig2.set_size_inches(4, 4)
    figArray = pn.Column(pn.Row(fig1, pn.Column(cm_change, fig2)), z)

    return figArray


pn.panel(pn.Column(pn.Row(pn.Column(batch_select,
                                    ex_select,
                                    s_dim_select,
                                    n_dim_select),
                          pn.Column(pn.Row(c_select, c_change),
                                    pn.Row(s_val_select, s_change),
                                    pn.Row(n_val_select, n_change))),
                   image)).servable(title='Plot RepGAN')
