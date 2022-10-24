## -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
__author__ = "Giorgia Colombera, Filippo Gatti"
__copyright__ = "Copyright 2021, CentraleSupélec (LMPS UMR CNRS 9026)"
__credits__ = ["Giorgia Colombera,Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__Maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow.keras.metrics as km
import tensorflow.keras.constraints as kc

tfd = tfp.distributions

loss_names = [
    "AdvDLossX",
    "AdvDlossC",
    "AdvDlossS",
    "AdvDlossN",
    "AdvGlossX",
    "AdvGlossC",
    "AdvGlossS",
    "AdvGlossN",
    "RecXloss",
    "RecCloss",
    "RecSloss",
    "FakeCloss"]

class ClipConstraint(kc.Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
    # clip model weights to hypercube
    def __call__(self, weights):
        if self.clip_value is not None:
            return tf.clip_by_value(weights, -self.clip_value, self.clip_value)
        else:             
            return tf.clip_by_value(weights, None, None)
    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

class sampleSlayer(kl.Layer):
    def __init__(self,latentSdim=1):
        super(sampleS, self).__init__()
        self.latentSdim = latentSdim
    def call(self,hs):
        μs, logΣs = tf.split(hs,num_or_size_splits=2, axis=1)
        ε = tf.random.normal(shape=self.latentSdim, mean=0.0, stddev=1.0)
        return μs + tf.exp(0.5*logΣs)*ε

def sampleS(hs,latentSdim):
    μs, logΣs = tf.split(hs,num_or_size_splits=2, axis=1)
    ε = tf.random.normal(shape=tf.shape(μs), mean=0.0, stddev=1.0)
    return μs + tf.math.exp(0.5*logΣs)*ε

class RepGAN(tf.keras.Model):

    def __init__(self,options):
        super(RepGAN, self).__init__()
        """
            Setup
        """
        self.__dict__.update(options)
        
        # Create Metric instances to track the losses
        self.loss_trackers = {"{:>s}_tracker".format(l): km.Mean(name=l) for l in loss_names}
        self.loss_val = {"{:>s}".format(l): 0.0 for l in loss_names}
        # define the constraint
        self.ClipD = ClipConstraint(0.01)
        
        self.ps = tfd.MultivariateNormalDiag(loc=tf.zeros(shape=(self.latentSdim,),
                                                          dtype=tf.float32),
                                             scale_diag=tf.ones(shape=(self.latentSdim,),
                                             dtype=tf.float32))
        self.pn = tfd.MultivariateNormalDiag(loc=tf.zeros(shape=(self.latentNdim,),
                                                          dtype=tf.float32),
                                             scale_diag=tf.ones(shape=(self.latentNdim,),
                                             dtype=tf.float32))
        self.BuildModels()
    
    def reset_metrics(self):
        for k,v in self.loss_trackers.items():
            v.reset_states()

    def get_config(self):
        config = super().get_config().copy()
        config.update({'size': self.Xshape})
        return config

    @property
    def metrics(self):
        return list(self.loss_trackers.values())
    
    def BuildModels(self,GP=False):
        """
            Build the discriminators
        """
        self.Dx = self.BuildDx()
        self.Dc = self.BuildDc()
        self.Ds = self.BuildDs()
        self.Dn = self.BuildDn()
        """
            Build Fx/Gz (generators)
        """
        self.Fx = self.BuildFx()
        self.Gz = self.BuildGz()
        
        self.models = [self.Dx, self.Dc, self.Ds, self.Dn,
                       self.Fx, self.Gz]
    
    def compile(self,optimizers,losses,**kwargs):
        
        super(RepGAN, self).compile(**kwargs)
        """
            Optimizers
        """
        self.__dict__.update(optimizers)
        """
            Losses
        """
        self.__dict__.update(losses)
            
    #@tf.function
    def train_XZX(self, X, c_prior):

        # Sample factorial prior S
        s_prior = self.ps.sample(self.batchSize)

        # Sample factorial prior N
        n_prior = self.pn.sample(self.batchSize)

        # Train discriminative part
        for _ in range(self.nCritic):
            
            # Tape gradients
            with tf.GradientTape(persistent=True) as tape:

                # Encode real signals X
                [_,s_fake,c_fake,n_fake] = self.Fx(X, training=True)

                # Discriminates real and fake S
                Ds_real = self.Ds(s_prior, training=True)
                Ds_fake = self.Ds(s_fake, training=True)

                # Discriminates real and fake N
                Dn_real = self.Dn(n_prior, training=True)
                Dn_fake = self.Dn(n_fake, training=True)
                
                # Discriminates real and fake C
                Dc_real = self.Dc(c_prior, training=True)
                Dc_fake = self.Dc(c_fake, training=True)

                # Compute XZX adversarial loss (JS(s),JS(n),JS(c))
                AdvDlossC = self.AdvDlossC(Dc_real, Dc_fake)
                AdvDlossS = self.AdvDlossS(Ds_real, Ds_fake)
                AdvDlossN = self.AdvDlossN(Dn_real, Dn_fake)
                AdvDlossXZX = AdvDlossC + AdvDlossS + AdvDlossN
                
            # Compute the discriminator gradient
            gradDc_w, gradDs_w, gradDn_w = tape.gradient(AdvDlossXZX,
                                                         (self.Dc.trainable_variables, 
                                                          self.Ds.trainable_variables,
                                                          self.Dn.trainable_variables),
                                                         unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Update discriminators' weights
            self.DcOpt.apply_gradients(zip(gradDc_w, self.Dc.trainable_variables))
            self.DsOpt.apply_gradients(zip(gradDs_w, self.Ds.trainable_variables))
            self.DnOpt.apply_gradients(zip(gradDn_w, self.Dn.trainable_variables))

        self.loss_val["AdvDlossC"] = AdvDlossC
        self.loss_val["AdvDlossS"] = AdvDlossS
        self.loss_val["AdvDlossN"] = AdvDlossN
            
        # Train the generative part
        for _ in range(self.nGenerator):
            
            # Tape gradients
            with tf.GradientTape(persistent=True) as tape:
                
                # Encode real signals
                _, s_fake, c_fake, n_fake = self.Fx(X, training=True)
                
                # Discriminate fake latent space
                Ds_fake = self.Ds(s_fake, training=True)
                Dc_fake = self.Dc(c_fake, training=True)
                Dn_fake = self.Dn(n_fake, training=True)

                # Compute adversarial loss for generator
                AdvGlossC = self.AdvGlossC(None, Dc_fake)
                AdvGlossS = self.AdvGlossS(None, Ds_fake)
                AdvGlossN = self.AdvGlossN(None, Dn_fake)
                
                # Compute total generator loss
                AdvGlossXZX = AdvGlossC + AdvGlossS + AdvGlossN
                
                # Reconstruct real signals
                X_rec = self.Gz((s_fake, c_fake, n_fake), training=True)

                # Compute reconstruction loss
                FakeCloss = self.FakeCloss(c_prior, c_fake)
                
                # Compute reconstruction loss
                RecXloss = self.RecXloss(X, X_rec)
                
                # Total generator loss
                GeneratorLossXZX = AdvGlossXZX + RecXloss + FakeCloss

            # Get the gradients w.r.t the generator loss
            gradFx_w, gradGz_w = tape.gradient(GeneratorLossXZX,
                                               (self.Fx.trainable_variables,
                                                self.Gz.trainable_variables),
                                               unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Update the weights of the generator using the generator optimizer
            self.FxOpt.apply_gradients(zip(gradFx_w,self.Fx.trainable_variables))
            self.GzOpt.apply_gradients(zip(gradGz_w,self.Gz.trainable_variables))
            
        self.loss_val["AdvGlossC"] = AdvGlossC
        self.loss_val["AdvGlossS"] = AdvGlossS
        self.loss_val["AdvGlossN"] = AdvGlossN
        self.loss_val["FakeCloss"] = FakeCloss
        self.loss_val["RecXloss"] = RecXloss
            
        return Dc_fake,Ds_fake,Dn_fake,Dc_real,Ds_real,Dn_real

    # #@tf.function
    def train_ZXZ(self, X, c_prior):

        # Sample factorial prior S
        s_prior = self.ps.sample(self.batchSize)

        # Sample factorial prior N
        n_prior = self.pn.sample(self.batchSize)

        # Train discriminative part
        for _ in range(self.nCritic):

            # Tape gradients
            with tf.GradientTape(persistent=True) as tape:

                # Decode factorial prior
                X_fake = self.Gz((s_prior, c_prior, n_prior), training=True)

                # Discriminate real and fake X
                Dx_real = self.Dx(X, training=True)
                Dx_fake = self.Dx(X_fake, training=True)

                # Compute the discriminator loss GAN loss (penalized)
                AdvDlossX = self.AdvDlossX(Dx_real, Dx_fake)
                
                AdvDlossZXZ = AdvDlossX
                if self.DxTrainType.upper() == "WGANGP":
                    
                    # Regularize with Gradient Penalty (WGANGP)
                    PenDxLoss = self.PenDxLoss(X,X_fake)
                    
                    AdvDlossZXZ += PenDxLoss
                
            # Compute the discriminator gradient
            gradDx_w = tape.gradient(AdvDlossZXZ, self.Dx.trainable_variables,
                                     unconnected_gradients=tf.UnconnectedGradients.ZERO)
            # Update the weights of the discriminator using the discriminator optimizer
            self.DxOpt.apply_gradients(zip(gradDx_w, self.Dx.trainable_variables))
            
            if self.DxTrainType.upper() == "WGANGP":
                self.loss_val["PenDxLoss"] = PenDxLoss
            else:
                self.loss_val["PenDxLoss"] = None
            
            self.loss_val["AdvDlossX"] = AdvDlossX
            
        # Train generative part
        for _ in range(self.nGenerator):
            
            # Tape gradients
            with tf.GradientTape(persistent=True) as tape:

                # Decode factorial prior
                X_fake = self.Gz((s_prior, c_prior, n_prior), training=True)

                # Discriminate real and fake X
                Dx_fake = self.Dx(X_fake, training=True)

                # Compute adversarial loos (penalized)
                AdvGlossX = self.AdvGlossX(None, Dx_fake)
                
                # Encode fake signals
                [hs, s_rec, c_rec, _] = self.Fx(X_fake, training=True)
                #Q_cont_distribution = tfp.distributions.MultivariateNormalDiag(loc=μs_rec, scale_diag=logΣs_rec)
                #RecSloss = -tf.reduce_mean(Q_cont_distribution.log_prob(s_rec))
                RecSloss = self.RecSloss(s_prior, hs)
                RecCloss = self.RecCloss(c_prior, c_rec)
                
                # Compute InfoGAN Q loos
                Qloss = RecSloss + RecCloss
                                
                # Total ZXZ generator loss
                GeneratorLossZXZ = AdvGlossX + Qloss

            # Get the gradients w.r.t the generator loss
            gradFx_w, gradGz_w = tape.gradient(GeneratorLossZXZ, (self.Fx.trainable_variables,
                                                                  self.Gz.trainable_variables),
                unconnected_gradients=tf.UnconnectedGradients.ZERO)
            
            # Update the weights of the generator using the generator optimizer
            self.GzOpt.apply_gradients(zip(gradGz_w,self.Gz.trainable_variables))
            self.FxOpt.apply_gradients(zip(gradFx_w,self.Fx.trainable_variables))

            self.loss_val["AdvGlossX"] = AdvGlossX
            self.loss_val["RecSloss"] = RecSloss
            self.loss_val["RecCloss"] = RecCloss

        return Dx_fake, Dx_real

    # #@tf.function
    def train_step(self, XC):
        if isinstance(XC, tuple):
            X, metadata = XC
            damage_class, magnitude, damage_index = metadata

        self.batchSize = tf.shape(X)[0]

        for _ in range(self.nRepXRep):  
            ZXZout = self.train_ZXZ(X, damage_class)
        for _ in range(self.nXRepX):
            XZXout = self.train_XZX(X, damage_class)

        (Dx_fake,Dx_real) = ZXZout

        (Dc_fake,Ds_fake,Dn_fake,Dc_real,Ds_real,Dn_real) = XZXout
        
        # Compute our own metrics
        for k,v in self.loss_trackers.items():
            v.update_state(self.loss_val[k.strip("_tracker")])

        return {k: v.result() for v in self.loss_trackers.values()} 
    
    @tf.function
    def test_step(self, XC):
        if isinstance(XC, tuple):
            X, c = XC
            c, mag, di = c
        # Compute predictions
        X_rec, c_fake, s_fake, n_fake = self(X, training=False)

        # Updates the metrics tracking the loss
        self.RecXloss(X, X_rec)
        # Update the metrics.
        self.RecGlossX_tracker.update_state(X, X_rec)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"RecXloss": RecGlossX_tracker.result()}


    def call(self, X):
        [_,s_fake,c_fake,n_fake] = self.Fx(X)
        X_rec = self.Gz((s_fake,c_fake,n_fake))
        return X_rec, c_fake, s_fake, n_fake

    def plot(self,X,c):
        [_,s_fake,c_fake,n_fake] = self.Fx(X,training=False)
        s_prior = self.ps.sample(X.shape[0])
        n_prior = self.pn.sample(X.shape[0])
        fakeX = self.Gz((s_prior,c,n_prior),training=False)
        X_rec = self.Gz((s_fake,c_fake,n_fake),training=False)
        return X_rec, c_fake, s_fake, n_fake, fakeX

    def label_predictor(self, X, c):
        [_,s_fake,c_fake,n_fake] = self.Fx(X)
        s_prior = self.ps.sample(s_fake.shape[0])
        n_prior = self.pn.sample(n_fake.shape[0])
        fakeX = self.Gz((s_prior,c,n_prior),training=False)
        [_,_,c_rec,_] = self.Fx(fakeX,training=False)
        return c_fake, c_rec
    
    def distribution(self,X,c):
        [hs_fake,s_fake,c_fake,n_fake] = self.Fx(X,training=False)
        μs_fake,logΣs_fake = tf.split(hs_fake,num_or_size_splits=2, axis=1)
        s_prior = self.ps.sample(X.shape[0])
        n_prior = self.pn.sample(X.shape[0])
        fakeX = self.Gz((s_prior,c,n_prior),training=False)
        [hs_rec,s_rec,c_rec,n_rec] = self.Fx(fakeX,training=False)
        μs_rec, logΣs_rec = tf.split(hs_fake,num_or_size_splits=2, axis=1)
        return s_prior, n_prior, s_fake, n_fake, s_rec, n_rec, μs_fake, logΣs_fake, μs_rec, logΣs_rec

    def reconstruct(self,X):
        [_,s_fake,c_fake,n_fake] = self.Fx(X)
        fakeX = self.Gz((s_fake,c_fake,n_fake),training=False)
        return fakeX
    
    def generate(self, X, c_fake_new):
        [_,s_fake,c_fake,n_fake] = self.Fx(X)
        X_rec_new = self.Gz((s_fake,c_fake_new,n_fake),training=False)
        return X_rec_new

    # BN : do not apply batchnorm to the generator output layer and the discriminator input layer
    def BuildFx(self):
        """
            Fx encoder structure
        """
        # To build this model using the functional API

        # Input layer
        X = kl.Input(shape=self.Xshape,name="X")

        # Initial CNN layer
        layer = -1
        h = kl.Conv1D(self.nZfirst,self.kernel,1,padding="same",
                data_format="channels_last",name="FxCNN0")(X)
        h = kl.BatchNormalization(momentum=0.95)(h)
        h = kl.LeakyReLU(alpha=0.1,name="FxA0")(h)
        h = kl.Dropout(self.dpout,name="FxDO0")(h)

        # Common encoder CNN layers
        for layer in range(self.nAElayers):
            h = kl.Conv1D(self.nZfirst*self.stride**(layer+1),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
            h = kl.BatchNormalization(momentum=0.95)(h)
            h = kl.LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
            h = kl.Dropout(self.dpout,name="FxDO{:>d}".format(layer+1))(h)

        # Last common CNN layer (no stride, same channels) before branching
        layer = self.nAElayers
        h = kl.Conv1D(self.nZchannels,
            self.kernel,1,padding="same",
            data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
        h = kl.BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
        h = kl.LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
        z = kl.Dropout(self.dpout,name="FxDO{:>d}".format(layer+1))(h)
        # z ---> Zshape = (Zsize,nZchannels)

        # variable s
        layer = 0
        # s-average
        h_μs = kl.Conv1D(self.nZchannels*self.Sstride**(layer+1),
            self.Skernel,self.Sstride,padding="same",
            data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(z)
        h_μs = kl.BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(h_μs)
        h_μs = kl.LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(h_μs)
        h_μs = kl.Dropout(self.dpout,name="FxDOmuS{:>d}".format(layer+1))(h_μs)
        if self.sdouble_branch:
            # s-logvar
            h_logDiagΣs = kl.Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(z)
            h_logDiagΣs = kl.BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(h_logDiagΣs)
            h_logDiagΣs = kl.LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(h_logDiagΣs)
            h_logDiagΣs = kl.Dropout(self.dpout,name="FxDOlvS{:>d}".format(layer+1))(h_logDiagΣs)
            
        # variable c
        h_c = kl.Conv1D(self.nZchannels*self.Cstride**(layer+1),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(z)
        h_c = kl.BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(h_c)
        h_c = kl.LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(h_c)
        #h_c = tfa.layers.InstanceNormalization()(h_c)
        h_c = kl.Dropout(self.dpout,name="FxDOC{:>d}".format(layer+1))(h_c)

        # variable n
        h_n = kl.Conv1D(self.nZchannels*self.Nstride**(layer+1),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(z)
        h_n = kl.BatchNormalization(momentum=0.95)(h_n)
        h_n = kl.LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(h_n)
        #h_n = tfa.layers.InstanceNormalization()(h_n)
        h_n = kl.Dropout(self.dpout,name="FxDON{:>d}".format(layer+1))(h_n)

        # variable s
        for layer in range(1,self.nSlayers):
            # s-average
            h_μs = kl.Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(h_μs)
            h_μs = kl.BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(h_μs)
            h_μs = kl.LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(h_μs)
            h_μs = kl.Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(h_μs)

        if self.sdouble_branch:
            for layer in range(1, self.nSlayers):
                # s-logvar
                h_logDiagΣs = kl.Conv1D(self.nZchannels*self.Sstride**(layer+1),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(h_logDiagΣs)
                h_logDiagΣs = kl.BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(h_logDiagΣs)
                h_logDiagΣs = kl.LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(h_logDiagΣs)
                h_logDiagΣs = kl.Dropout(self.dpout,name="FxDOlvS{:>d}".format(layer+1))(h_logDiagΣs)

        # variable c
        for layer in range(1,self.nClayers):
            h_c = kl.Conv1D(self.nZchannels*self.Cstride**(layer+1),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(h_c)
            h_c = kl.BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(h_c)
            h_c = kl.LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(h_c)
            #h_c = tfa.layers.InstanceNormalization()(h_c)
            h_c = kl.Dropout(self.dpout,name="FxDOC{:>d}".format(layer+1))(h_c)

        # variable n
        for layer in range(1,self.nNlayers):
            h_n = kl.Conv1D(self.nZchannels*self.Nstride**(layer+1),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(h_n)
            h_n = kl.BatchNormalization(momentum=0.95)(h_n)
            h_n = kl.LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(h_n)
            #h_n = tfa.layers.InstanceNormalization()(h_n)
            h_n = kl.Dropout(self.dpout,name="FxDON{:>d}".format(layer+1))(h_n)

        # variable s
        # s-average
        h_μs = kl.Flatten(name="FxFLmuS{:>d}".format(layer+1))(h_μs)
        h_μs = kl.Dense(1024)(h_μs)
        h_μs = kl.BatchNormalization(momentum=0.95)(h_μs)
        h_μs = kl.LeakyReLU(alpha=0.1)(h_μs)
        
        if self.sdouble_branch:
            # s-average
            h_μs = kl.Dense(self.latentSdim,name="FxFWmuS")(h_μs)
            
            # s-logvar
            h_logDiagΣs = kl.Flatten(name="FxFLlvS{:>d}".format(layer+1))(h_logDiagΣs)
            h_logDiagΣs = kl.Dense(1024)(h_logDiagΣs)
            h_logDiagΣs = kl.BatchNormalization(momentum=0.95)(h_logDiagΣs)
            h_logDiagΣs = kl.LeakyReLU(alpha=0.1)(h_logDiagΣs)
            h_logDiagΣs = kl.Dense(self.latentSdim,name="FxFWlvS")(h_logDiagΣs)
            
            if 'sigmoid' in self.sigmas2:
                logΣs = tf.keras.activations.sigmoid(h_logDiagΣs)
            elif 'softplus' in self.sigmas2:
                logΣs = tf.keras.activations.softplus(h_logDiagΣs)
            else:
                logΣs = h_logDiagΣs
            
            hs = kl.Concatenate([h_μs,h_logDiagΣs],axis=-1,name="FxFWS")
            
        else:
            # s-average & s-logvar
            hs = kl.Dense(self.latentSdim+self.latentSdim,name="FxFWS")(h_μs)
            if 'sigmoid' in self.sigmas2:
                hs = tf.keras.activations.sigmoid(hs)
            elif 'softplus' in self.sigmas2:
                hs = tf.keras.activations.softplus(hs)
        
        # variable c
        layer = self.nClayers
        h_c = kl.Flatten(name="FxFLC{:>d}".format(layer+1))(h_c)
        h_c = kl.Dense(1024)(h_c)
        h_c = kl.BatchNormalization(momentum=0.95,axis=-1)(h_c)
        h_c = kl.LeakyReLU(alpha=0.1)(h_c)
        h_c = kl.Dense(self.latentCdim,name="FxFWC")(h_c)
        h_c = kl.BatchNormalization(momentum=0.95,axis=-1)(h_c)
        #h_c = tfa.layers.InstanceNormalization()(h_c)

        # variable n
        layer = self.nNlayers
        # h_n = kl.Conv1D(self.nZchannels,self.Nkernel,self.Nstride,padding="same",
        #         data_format="channels_last")(h_n)
        # h_n = kl.BatchNormalization(momentum=0.95)(h_n)
        # h_n = kl.LeakyReLU(alpha=0.1)(h_n)
        # h_n = kl.Dropout(0.2)(h_n)

        h_n = kl.Flatten(name="FxFLN{:>d}".format(layer+1))(h_n)
        h_n = kl.Dense(1024)(h_n)
        h_n = kl.BatchNormalization(momentum=0.95)(h_n)
        h_n = kl.LeakyReLU(alpha=0.1)(h_n)
        h_n = kl.Dense(self.latentNdim,name="FxFWN")(h_n)

        # variable s
        s = sampleS(hs,self.latentSdim)

        # variable c
        #c = kl.Dense(self.latentCdim,activation=tf.keras.activations.softmax)(h_c)
        c = kl.Softmax(name="damage_class")(h_c)

        # variable n
        n = kl.BatchNormalization(name="bn_noise",momentum=0.95)(h_n)
        #n = tfa.layers.InstanceNormalization()(h_n)

        Fx = tf.keras.Model(X,[hs,s,c,n],name="Fx")

        return Fx

    def BuildGz(self):
        """
            Conv1D Gz structure
            https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

        """
    
        s = kl.Input(shape=(self.latentSdim,),name="s")
        c = kl.Input(shape=(self.latentCdim,),name="c")
        n = kl.Input(shape=(self.latentNdim,),name="n")

        layer = 0
        # variable s
        h_s = tfa.layers.SpectralNormalization(kl.Dense(self.Ssize*self.nSchannels,
                                            name="GzFWS0"))(s)
        h_s = kl.BatchNormalization(momentum=0.95)(h_s)
        h_s = kl.LeakyReLU(alpha=0.1)(h_s)
        h_s = kl.Reshape((self.Ssize,self.nSchannels))(h_s)

        if self.skip:
            for layer in range(1,self.nSlayers):
                s1 = tfa.layers.SpectralNormalization(kl.Dense(h_s.shape[1]*h_s.shape[2]))(s)
                s1 = kl.Reshape((h_s.shape[1],h_s.shape[2]))(s1)
                h_s = kl.concatenate([h_s,s1])
                h_s = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nSchannels*self.Sstride**(-layer)),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last"))(h_s)
                h_s = kl.LeakyReLU(alpha=0.1)(h_s)
                h_s = kl.BatchNormalization(momentum=0.95)(h_s)
            s1 = tfa.layers.SpectralNormalization(kl.Dense(h_s.shape[1]*h_s.shape[2]))(s)
            s1 = kl.Reshape((h_s.shape[1],h_s.shape[2]))(s1)
            h_s = kl.concatenate([h_s,s1])
            h_s = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nSchannels*self.Sstride**(-self.nSlayers)),
            self.Skernel,self.Sstride,padding="same",data_format="channels_last"))(h_s)
            h_s = kl.LeakyReLU(alpha=0.1)(h_s)
            h_s = kl.BatchNormalization(momentum=0.95,name="GzBNS{:>d}".format(self.nSlayers))(h_s)
            GzS = keras.Model(s,h_s)
        
        else:
            for layer in range(1,self.nSlayers):
                h_s = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nSchannels*self.Sstride**(-layer)),
                    self.Skernel,self.Sstride,padding="same",
                    data_format="channels_last"))(h_s)
                h_s = kl.LeakyReLU(alpha=0.1)(h_s)
                h_s = kl.BatchNormalization(momentum=0.95)(h_s)
                #h_s = kl.Dropout(self.dpout,name="GzDOS{:>d}".format(layer))(h_s)
            h_s = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nSchannels*self.Sstride**(-self.nSlayers)),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last"))(h_s)
            h_s = kl.BatchNormalization(momentum=0.95,name="GzBNS{:>d}".format(self.nSlayers))(h_s)
            h_s = kl.LeakyReLU(alpha=0.1)(h_s)
            #h_s = kl.Dropout(self.dpout)(h_s)
            GzS = tf.keras.Model(s,h_s)

        # variable c
        h_c = tfa.layers.SpectralNormalization(kl.Dense(self.Csize*self.nCchannels))(c)
        h_c = kl.BatchNormalization(momentum=0.95)(h_c)
        h_c = kl.LeakyReLU(alpha=0.1,)(h_c)
        h_c = kl.Reshape((self.Csize,self.nCchannels))(h_c)

        if self.skip:
            for layer in range(1,self.nClayers):
                c1 = tfa.layers.SpectralNormalization(kl.Dense(h_c.shape[1]*h_c.shape[2]))(c)
                c1 = kl.Reshape((h_c.shape[1],h_c.shape[2]))(c1)
                h_c = kl.concatenate([h_c,c1])
                h_c = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nCchannels*self.Cstride**(-layer)),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last"))(h_c)
                h_c = kl.LeakyReLU(alpha=0.1)(h_c)
                h_c = kl.BatchNormalization(momentum=0.95)(h_c)
            c1 = tfa.layers.SpectralNormalization(kl.Dense(h_c.shape[1]*h_c.shape[2]))(c)
            c1 = kl.Reshape((h_c.shape[1],h_c.shape[2]))(c1)
            h_c = kl.concatenate([h_c,c1])
            h_c = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nCchannels*self.Cstride**(-self.nClayers)),
            self.Ckernel,self.Cstride,padding="same",data_format="channels_last"))(h_c)
            h_c = kl.LeakyReLU(alpha=0.1)(h_c)
            h_c = kl.BatchNormalization(momentum=0.95,name="GzBNC{:>d}".format(self.nClayers))(h_c)
            GzC = keras.Model(c,h_c)

        else:
            for layer in range(1,self.nClayers):
                h_c = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nCchannels*self.Cstride**(-layer)),
                    self.Ckernel,self.Cstride,padding="same",
                    data_format="channels_last"))(h_c)
                h_c = kl.BatchNormalization(momentum=0.95)(h_c)
                h_c = kl.LeakyReLU(alpha=0.1)(h_c)
                # h_c = kl.Dropout(self.dpout)(h_c)
            h_c = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nCchannels*self.Cstride**(-self.nClayers)),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last"))(h_c)
            h_c = kl.BatchNormalization(momentum=0.95)(h_c)
            h_c = kl.LeakyReLU(alpha=0.1)(h_c)
            #h_c = kl.Dropout(self.dpout)(h_c)
            GzC = tf.keras.Model(c,h_c)

        # variable n
        h_n = tfa.layers.SpectralNormalization(kl.Dense(self.Nsize*self.nNchannels))(n)
        h_n = kl.BatchNormalization(momentum=0.95)(h_n)
        h_n = kl.LeakyReLU(alpha=0.1)(h_n)
        h_n = kl.Reshape((self.Nsize,self.nNchannels))(h_n)

        if self.skip:
            for layer in range(1,self.nNlayers):
                n1 = tfa.layers.SpectralNormalization(kl.Dense(h_n.shape[1]*h_n.shape[2]))(n)
                n1 = kl.Reshape((h_n.shape[1],h_n.shape[2]))(n1)
                h_n = kl.concatenate([h_n,n1])
                h_n = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nNchannels*self.Nstride**(-layer)),
                    self.Nkernel,self.Nstride,padding="same",
                    data_format="channels_last"))(h_n)
                h_n = kl.LeakyReLU(alpha=0.1)(h_n)
                h_n = kl.BatchNormalization(momentum=0.95)(h_n)
            n1 = tfa.layers.SpectralNormalization(kl.Dense(h_n.shape[1]*h_n.shape[2]))(n)
            n1 = kl.Reshape((h_n.shape[1],h_n.shape[2]))(n1)
            h_n = kl.concatenate([h_n,n1])
            h_n = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nNchannels*self.Nstride**(-self.nNlayers)),
            self.Nkernel,self.Nstride,padding="same",data_format="channels_last"))(h_n)
            h_n = kl.LeakyReLU(alpha=0.1)(h_n)
            h_n = kl.BatchNormalization(momentum=0.95,name="GzBNN{:>d}".format(self.nNlayers))(h_n)
            GzN = keras.Model(n,h_n)

        else:
            for layer in range(1,self.nNlayers):
                h_n = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nNchannels*self.Nstride**(-layer)),
                    self.Nkernel,self.Nstride,padding="same",
                    data_format="channels_last"))(h_n)
                h_n = kl.LeakyReLU(alpha=0.1)(h_n)
                h_n = kl.BatchNormalization(momentum=0.95)(h_n)
                #h_n = kl.Dropout(self.dpout)(h_n)
            h_n = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nNchannels*self.Nstride**(-self.nNlayers)),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last"))(h_n)
            h_n = kl.BatchNormalization(momentum=0.95)(h_n)
            h_n = kl.LeakyReLU(alpha=0.1)(h_n)
            # h_n = kl.Dropout(self.dpout)(h_n)
            GzN = tf.keras.Model(n,h_n)

        if self.skip:
            s1 = kl.Dense(h_s.shape[1]*h_s.shape[2])(s)
            s1 = kl.Reshape((h_s.shape[1],h_s.shape[2]))(s1)
            c1 = kl.Dense(h_c.shape[1]*h_c.shape[2])(c)
            c1 = kl.Reshape((h_c.shape[1],h_c.shape[2]))(c1)
            n1 = kl.Dense(h_n.shape[1]*h_n.shape[2])(n)
            n1 = kl.Reshape((h_n.shape[1],h_n.shape[2]))(n1)
            Gz = kl.concatenate([GzS.output,GzC.output,GzN.output,s1,c1,n1])
            Gz = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels,self.kernel,1,padding="same",
            data_format="channels_last"))(Gz)
            Gz = kl.LeakyReLU(alpha=0.1)(Gz)
            Gz = kl.BatchNormalization(axis=-1,momentum=0.95)(Gz)
        
        else:
            Gz = kl.concatenate([GzS.output,GzC.output,GzN.output])
            Gz = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels,
                    self.kernel,1,padding="same",
                    data_format="channels_last"))(Gz)
            Gz = kl.BatchNormalization(axis=-1,momentum=0.95)(Gz)
            Gz = kl.LeakyReLU(alpha=0.1)(Gz)
        
        if self.skip:
            for layer in range(self.nAElayers-1):
                s1 = kl.Dense(Gz.shape[1]*Gz.shape[2])(s)
                s1 = kl.Reshape((Gz.shape[1],Gz.shape[2]))(s1)
                c1 = kl.Dense(Gz.shape[1]*Gz.shape[2])(c)
                c1 = kl.Reshape((Gz.shape[1],Gz.shape[2]))(c1)
                n1 = kl.Dense(Gz.shape[1]*Gz.shape[2])(n)
                n1 = kl.Reshape((Gz.shape[1],Gz.shape[2]))(n1)
                Gz = kl.concatenate([Gz,s1,c1,n1])
                Gz = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                    self.kernel,self.stride,padding="same",use_bias=False))(Gz)
                Gz = kl.LeakyReLU(alpha=0.1)(Gz)
                Gz = kl.BatchNormalization(axis=-1,momentum=0.95)(Gz)

        else:
            for layer in range(self.nAElayers-1):
                Gz = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                    self.kernel,self.stride,padding="same",use_bias=False))(Gz)
                Gz = kl.BatchNormalization(axis=-1,momentum=0.95)(Gz)
                Gz = kl.LeakyReLU(alpha=0.1)(Gz)

        layer = self.nAElayers-1

        if self.skip:
            s1 = kl.Dense(Gz.shape[1]*Gz.shape[2])(s)
            s1 = kl.Reshape((Gz.shape[1],Gz.shape[2]))(s1)
            c1 = kl.Dense(Gz.shape[1]*Gz.shape[2])(c)
            c1 = kl.Reshape((Gz.shape[1],Gz.shape[2]))(c1)
            n1 = kl.Dense(Gz.shape[1]*Gz.shape[2])(n)
            n1 = kl.Reshape((Gz.shape[1],Gz.shape[2]))(n1)
            Gz = kl.concatenate([Gz,s1,c1,n1])
            Gz = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                    self.kernel,self.stride,padding="same",use_bias=False))(Gz)
            Gz = kl.LeakyReLU(alpha=0.1,name="GzA{:>d}".format(layer+1))(Gz)
            Gz = kl.BatchNormalization(axis=-1,momentum=0.95)(Gz)

        else:
            Gz = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                    self.kernel,self.stride,padding="same",use_bias=False))(Gz)
            Gz = kl.BatchNormalization(axis=-1,momentum=0.95)(Gz)
            Gz = kl.LeakyReLU(alpha=0.1,name="GzA{:>d}".format(layer+1))(Gz)

        layer = self.nAElayers

        if self.skip:
            s1 = kl.Dense(Gz.shape[1]*Gz.shape[2])(s)
            s1 = kl.Reshape((Gz.shape[1],Gz.shape[2]))(s1)
            c1 = kl.Dense(Gz.shape[1]*Gz.shape[2])(c)
            c1 = kl.Reshape((Gz.shape[1],Gz.shape[2]))(c1)
            n1 = kl.Dense(Gz.shape[1]*Gz.shape[2])(n)
            n1 = kl.Reshape((Gz.shape[1],Gz.shape[2]))(n1)
            Gz = kl.concatenate([Gz,s1,c1,n1])
            X = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nXchannels,self.kernel,1,padding="same",
            activation='tanh',use_bias=False))(Gz)
        
        else:
            X = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nXchannels,self.kernel,1,
            padding="same",activation='tanh',use_bias=False))(Gz)

        Gz = tf.keras.Model(inputs=[GzS.input,GzC.input,GzN.input],outputs=X,name="Gz")
        return Gz

    def BuildDx(self):
        """
            Conv1D discriminator structure
        """
        layer = 0
        X = kl.Input(shape=self.Xshape,name="X")
        
        if self.DxSN:
            h = tfa.layers.SpectralNormalization(kl.Conv1D(self.Xsize*self.stride**(-(layer+1)),
                    self.kernel,self.stride,padding="same",
                    data_format="channels_last",name="DxCNN0"))(X)
            h = kl.LeakyReLU(alpha=0.1,name="DxA0")(h)
            h = kl.Dropout(0.25)(h)

            for layer in range(1,self.nDlayers):
                h = tfa.layers.SpectralNormalization(kl.Conv1D(self.Xsize*self.stride**(-(layer+1)),
                    self.kernel,self.stride,padding="same",
                    data_format="channels_last",name="DxCNN{:>d}".format(layer)))(h)
                h = kl.LeakyReLU(alpha=0.1,name="DxA{:>d}".format(layer))(h)
                h = kl.Dropout(0.25)(h)

            layer = self.nDlayers    
            h = kl.Flatten(name="DxFL{:>d}".format(layer))(h)
            h = tfa.layers.SpectralNormalization(kl.Dense(1024))(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Px = tfa.layers.SpectralNormalization(kl.Dense(1))(h)
        else:
            h = kl.Conv1D(self.Xsize*self.stride**(-(layer+1)),
                    self.kernel,self.stride,padding="same",
                    data_format="channels_last",name="DxCNN0")(X)
            h = kl.LeakyReLU(alpha=0.1,name="DxA0")(h)
            h = kl.Dropout(0.25)(h)

            for layer in range(1,self.nDlayers):
                h = kl.Conv1D(self.Xsize*self.stride**(-(layer+1)),
                    self.kernel,self.stride,padding="same",
                    data_format="channels_last",name="DxCNN{:>d}".format(layer))(h)
                h = kl.LeakyReLU(alpha=0.1,name="DxA{:>d}".format(layer))(h)
                h = kl.Dropout(0.25)(h)

            layer = self.nDlayers    
            h = kl.Flatten(name="DxFL{:>d}".format(layer))(h)
            h = kl.Dense(1024)(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Px = kl.Dense(1)(h)
            
        
        Dx = tf.keras.Model(X,Px,name="Dx")
        return Dx


    def BuildDc(self):
        """
            Dense discriminator structure
        """
        c = kl.Input(shape=(self.latentCdim,))
        if self.DzSN:
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(c)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Pc = tfa.layers.SpectralNormalization(kl.Dense(1))(h)
        else:    
            h = kl.Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(c)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = kl.Dense(1000,kernel_constraint=ClipConstraint(self.clipValue))(h)
            h = kl.BatchNormalization(momentum=0.95)(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Pc = kl.Dense(1,kernel_constraint=ClipConstraint(self.clipValue))(h)
        Dc = tf.keras.Model(c,Pc,name="Dc")
        return Dc


    def BuildDn(self):
        """
            Dense discriminator structure
        """
        n = kl.Input(shape=(self.latentNdim,))
        if self.DzSN:
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(n)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Pn = tfa.layers.SpectralNormalization(kl.Dense(1))(h)
        else:
            h = kl.Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(n) 
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = kl.Dense(1000,kernel_constraint=ClipConstraint(self.clipValue))(h)
            h = kl.BatchNormalization(momentum=0.95)(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Pn = kl.Dense(1,kernel_constraint=ClipConstraint(self.clipValue))(h)
        Dn = tf.keras.Model(n,Pn,name="Dn")
        return Dn

    def BuildDs(self):
        """
            Dense discriminator structure
        """
        s = kl.Input(shape=(self.latentSdim,))
        if self.DzSN:
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(s)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Ps = tfa.layers.SpectralNormalization(kl.Dense(1))(h)
        else:
            h = kl.Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(s)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = kl.Dense(1000,kernel_constraint=ClipConstraint(self.clipValue))(h)
            h = kl.BatchNormalization(momentum=0.95)(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Ps = kl.Dense(1,kernel_constraint=ClipConstraint(self.clipValue))(h)
        Ds = tf.keras.Model(s,Ps,name="Ds")
        return Ds