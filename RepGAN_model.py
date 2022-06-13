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
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow_addons as tfa
from tensorflow.keras.constraints import Constraint
import tensorflow_probability as tfp

from RepGAN_losses import GradientPenalty as GP

AdvDLoss_tracker = keras.metrics.Mean(name="loss")
AdvDlossX_tracker = keras.metrics.Mean(name="loss")
AdvDlossC_tracker = keras.metrics.Mean(name="loss")
AdvDlossS_tracker = keras.metrics.Mean(name="loss")
AdvDlossN_tracker = keras.metrics.Mean(name="loss")
AdvGLoss_tracker = keras.metrics.Mean(name="loss")
AdvGlossX_tracker = keras.metrics.Mean(name="loss")
AdvGlossC_tracker = keras.metrics.Mean(name="loss")
AdvGlossS_tracker = keras.metrics.Mean(name="loss")
AdvGlossN_tracker = keras.metrics.Mean(name="loss")
RecGlossX_tracker = keras.metrics.Mean(name="loss")
RecGlossC_tracker = keras.metrics.Mean(name="loss")
RecGlossS_tracker = keras.metrics.Mean(name="loss")
Qloss_tracker = keras.metrics.Mean(name="loss")

class ClipConstraint(tf.keras.constraints.Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
    # clip model weights to hypercube
    def __call__(self, weights):
        if self.clip_value is not None:
            return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)
        else:             
            return tf.keras.backend.clip(weights, None, None)
    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

class sampleS(kl.Layer):
    def call(self, inputs):
        μ,σ2 = inputs
        ε = tf.random.normal(shape=tf.shape(μ),mean=0.0,stddev=1.0)
        #return μ + tf.multiply(tf.sqrt(σ2),ε)
        return μ + tf.multiply(σ2,ε)

class RepGAN(tf.keras.Model):

    def __init__(self,options):
        super(RepGAN, self).__init__()
        """
            Setup
        """
        self.__dict__.update(options)

        # define the constraint
        self.ClipD = ClipConstraint(0.01)
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

    def get_config(self):
        config = super().get_config().copy()
        config.update({'size': self.Xshape})
        return config

    @property
    def metrics(self):
        return AdvDLoss_tracker,AdvGLoss_tracker,AdvDlossX_tracker,AdvDlossC_tracker,AdvDlossS_tracker,\
            AdvDlossN_tracker,AdvGlossX_tracker,AdvGlossC_tracker,AdvGlossS_tracker,AdvGlossN_tracker,\
            RecGlossX_tracker,RecGlossC_tracker,RecGlossS_tracker,Qloss_tracker

    def compile(self,optimizers,losses): #run_eagerly
        super(RepGAN, self).compile()
        """
            Optimizers
        """
        self.__dict__.update(optimizers)
        """
            Losses
        """
        self.__dict__.update(losses)

    def train_XZX(self,X,c):

        # Create labels for BCE in GAN loss
        realBCE_C = tf.ones((self.batchSize,1), dtype=tf.float32)
        fakeBCE_C = -tf.ones((self.batchSize,1), dtype=tf.float32)
        realBCE_S = tf.ones((self.batchSize,1), dtype=tf.float32)
        fakeBCE_S = -tf.ones((self.batchSize,1), dtype=tf.float32)
        realBCE_N = tf.ones((self.batchSize,1), dtype=tf.float32)
        fakeBCE_N = -tf.ones((self.batchSize,1), dtype=tf.float32)

        # Sample factorial prior S
        s_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[self.batchSize,self.latentSdim],dtype=tf.float32)

        # Sample factorial prior N
        n_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[self.batchSize,self.latentNdim],dtype=tf.float32)

        # Train generators
        self.Fx.trainable = True
        self.Gz.trainable = True
        self.Dx.trainable = False
        self.Dc.trainable = False
        self.Ds.trainable = False
        self.Dn.trainable = False

        # Train nGenerator times the generators
        #for _ in range(self.nGenerator):

        with tf.GradientTape(persistent=True) as tape:

            # Encode real signals
            _,_,s_fake,c_fake,n_fake = self.Fx(X,training=True)

            # Reconstruct real signals
            X_rec = self.Gz((s_fake,c_fake,n_fake),training=True)

            # Compute reconstruction loss
            RecGlossX = self.RecXloss(X,X_rec)*self.PenRecXloss #-tf.reduce_mean(logpz - logqz_x)
    
        # Get the gradients w.r.t the generator loss
        gradFx, gradGz = tape.gradient(RecGlossX,
        (self.Fx.trainable_variables,self.Gz.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Update the weights of the generator using the generator optimizer
        self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))
        self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))

        # Train discriminators
        self.Fx.trainable = False
        self.Gz.trainable = False
        self.Dx.trainable = False
        self.Dc.trainable = True
        self.Ds.trainable = True
        self.Dn.trainable = True

        # Train nCritic times the discriminators
        for _ in range(self.nCritic):
            
            with tf.GradientTape(persistent=True) as tape:

                # Encode real signals X
                [_,_,s_fake,c_fake,n_fake] = self.Fx(X,training=True)

                # Discriminates real and fake S
                s_fakecritic = self.Ds(s_fake,training=True)
                s_priorcritic = self.Ds(s_prior,training=True)

                # Discriminates real and fake N
                n_fakecritic = self.Dn(n_fake,training=True)
                n_priorcritic = self.Dn(n_prior,training=True)

                # Discriminates real and fake C
                c_fakecritic = self.Dc(c_fake,training=True)
                c_critic = self.Dc(c,training=True)

                #Compute XZX adversarial loss (JS(s),JS(n),JS(c))
                AdvDlossC  = self.AdvDlossDz(realBCE_C,c_critic)*self.PenAdvCloss
                AdvDlossC += self.AdvDlossDz(fakeBCE_C,c_fakecritic)*self.PenAdvCloss
                AdvDlossS  = self.AdvDlossDz(realBCE_S,s_priorcritic)*self.PenAdvSloss
                AdvDlossS += self.AdvDlossDz(fakeBCE_S,s_fakecritic)*self.PenAdvSloss
                AdvDlossN  = self.AdvDlossDz(realBCE_N,n_priorcritic)*self.PenAdvNloss
                AdvDlossN += self.AdvDlossDz(fakeBCE_N,n_fakecritic)*self.PenAdvNloss
                
                AdvDloss = 0.5*(AdvDlossC + AdvDlossS + AdvDlossN)

            # Compute the discriminator gradient
            gradDc, gradDs, gradDn = tape.gradient(AdvDloss,
                    (self.Dc.trainable_variables,self.Ds.trainable_variables, 
                    self.Dn.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Update discriminators' weights
            self.DcOpt.apply_gradients(zip(gradDc,self.Dc.trainable_variables))
            self.DsOpt.apply_gradients(zip(gradDs,self.Ds.trainable_variables))
            self.DnOpt.apply_gradients(zip(gradDn,self.Dn.trainable_variables))
        

        self.Fx.trainable = True
        self.Gz.trainable = False
        self.Dx.trainable = False
        self.Dc.trainable = False
        self.Ds.trainable = False
        self.Dn.trainable = False

        with tf.GradientTape(persistent=True) as tape:

            # Encode real signals
            _,_,s_fake,c_fake,n_fake = self.Fx(X,training=True)

            # Discriminate fake latent space
            s_fakecritic = self.Ds(s_fake,training=True)
            c_fakecritic = self.Dc(c_fake,training=True)
            n_fakecritic = self.Dn(n_fake,training=True)

            # Compute adversarial loss for generator
            AdvGlossC = self.AdvGlossDz(realBCE_C,c_fakecritic)*self.PenAdvCloss
            AdvGlossS = self.AdvGlossDz(realBCE_S,s_fakecritic)*self.PenAdvSloss
            AdvGlossN = self.AdvGlossDz(realBCE_N,n_fakecritic)*self.PenAdvNloss

            # Compute total generator loss
            AdvGloss = AdvGlossC + AdvGlossS + AdvGlossN
            
        # Get the gradients w.r.t the generator loss
        gradFx = tape.gradient(AdvGloss,(self.Fx.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Update the weights of the generator using the generator optimizer
        self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))

        return RecGlossX,AdvDloss,AdvDlossC,AdvDlossS,AdvDlossN,AdvGloss,AdvGlossC,AdvGlossS,AdvGlossN,\
                c_fakecritic,s_fakecritic,n_fakecritic,c_critic,s_priorcritic,n_priorcritic

    def train_ZXZ(self,X,c):

        # Create labels for BCE in GAN loss
        realBCE_X = tf.ones((self.batchSize,1), dtype=tf.float32)
        fakeBCE_X = tf.zeros((self.batchSize,1), dtype=tf.float32)

        # Sample factorial prior S
        s_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[self.batchSize,self.latentSdim],dtype=tf.float32)

        # Sample factorial prior N
        n_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[self.batchSize,self.latentNdim],dtype=tf.float32)

        for _ in range(self.nCritic):

            # Train discriminators
            self.Fx.trainable = False
            self.Gz.trainable = False
            self.Dx.trainable = True

            with tf.GradientTape(persistent=True) as tape:

                # Decode factorial prior
                X_fake = self.Gz((s_prior,c,n_prior),training=True)

                # Discriminate real and fake X
                X_fakecritic = self.Dx(X_fake,training=True)
                X_critic = self.Dx(X,training=True)

                # Compute the discriminator loss GAN loss (penalized)
                #AdvDlossX = self.AdvDlossDx(fakeBCE_X,X_fakecritic)*self.PenAdvXloss
                #AdvDlossX += self.AdvDlossDx(realBCE_X,X_critic)*self.PenAdvXloss
                AdvDlossX = -tf.reduce_mean(tf.math.log(X_critic+1e-8) + tf.math.log(1 - X_fakecritic+1e-8))*self.PenAdvXloss

            # Compute the discriminator gradient
            gradDx = tape.gradient(AdvDlossX,self.Dx.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            # Update the weights of the discriminator using the discriminator optimizer
            self.DxOpt.apply_gradients(zip(gradDx,self.Dx.trainable_variables))


            # Train generators
            self.Fx.trainable = False
            self.Gz.trainable = True
            self.Dx.trainable = False

            with tf.GradientTape(persistent=True) as tape:

                # Decode factorial prior
                X_fake = self.Gz((s_prior,c,n_prior),training=True)

                # Discriminate real and fake X
                X_fakecritic = self.Dx(X_fake,training=True)

                # Compute adversarial loos (penalized)
                #AdvGlossX = self.AdvGlossDx(realBCE_X,X_fakecritic)*self.PenAdvXloss
                AdvGlossX = -tf.reduce_mean(tf.math.log(X_fakecritic+1e-8))*self.PenAdvXloss

            # Get the gradients w.r.t the generator loss
            gradGz = tape.gradient(AdvGlossX,(self.Gz.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Update the weights of the generator using the generator optimizer
            self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))

            # Train generators
            self.Fx.trainable = True
            self.Gz.trainable = True
            self.Dx.trainable = False

            with tf.GradientTape(persistent=True) as tape:
                
                # Encode fake signals
                [μs_rec,σs2_rec,s_rec,c_rec,_] = self.Fx(X_fake,training=True)

                #Q_cont_distribution = tfp.distributions.MultivariateNormalDiag(loc=μs_rec, scale_diag=σs2_rec)
                #RecGlossS = -tf.reduce_mean(Q_cont_distribution.log_prob(s_rec))
                RecGlossS = self.RecSloss(s_prior,μs_rec,σs2_rec)*self.PenRecSloss
                RecGlossC = self.RecCloss(c,c_rec)*self.PenRecCloss #+ self.RecCloss(c,c_fake)*self.PenRecCloss

                # Compute InfoGAN Q loos
                Qloss = RecGlossS + RecGlossC

            # Get the gradients w.r.t the generator loss
            gradFx, gradGz = tape.gradient(Qloss,
                (self.Fx.trainable_variables,self.Gz.trainable_variables),unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Update the weights of the generator using the generator optimizer
            self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))
            self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))

        return AdvDlossX,AdvGlossX,RecGlossS,RecGlossC,Qloss,X_fakecritic,X_critic

    def train_step(self, XC):

        X, c, mag, di = XC

        self.batchSize = tf.shape(X)[0]

        #------------------------------------------------
        #           Construct Computational Graph
        #               for the Discriminator
        #------------------------------------------------
        #       
        for _ in range(self.nRepXRep):  
            ZXZout = self.train_ZXZ(X,c)
        for _ in range(self.nXRepX):
            XZXout = self.train_XZX(X,c)

        (AdvDlossX,AdvGlossX,RecGlossS,RecGlossC,Qloss,X_fakecritic,X_critic) = ZXZout

        (RecGlossX,AdvDloss,AdvDlossC,AdvDlossS,AdvDlossN,AdvGloss,AdvGlossC,AdvGlossS,AdvGlossN,\
            c_fakecritic,s_fakecritic,n_fakecritic,c_critic,s_priorcritic,n_priorcritic) = XZXout     
      
        # Compute our own metrics
        AdvDLoss_tracker.update_state(AdvDloss)
        AdvGLoss_tracker.update_state(AdvGloss)
        AdvDlossX_tracker.update_state(AdvDlossX)
        AdvDlossC_tracker.update_state(AdvDlossC)
        AdvDlossS_tracker.update_state(AdvDlossS)
        AdvDlossN_tracker.update_state(AdvDlossN)

        AdvGlossX_tracker.update_state(AdvGlossX)
        AdvGlossC_tracker.update_state(AdvGlossC)
        AdvGlossS_tracker.update_state(AdvGlossS)
        AdvGlossN_tracker.update_state(AdvGlossN)

        RecGlossX_tracker.update_state(RecGlossX)
        RecGlossC_tracker.update_state(RecGlossC)
        RecGlossS_tracker.update_state(RecGlossS)

        Qloss_tracker.update_state(Qloss)

        return {"AdvDLoss": AdvDLoss_tracker.result(),"AdvGLoss": AdvGLoss_tracker.result(),"AdvDlossX": AdvDlossX_tracker.result(),
            "AdvDlossC": AdvDlossC_tracker.result(), "AdvDlossS": AdvDlossS_tracker.result(),"AdvDlossN": AdvDlossN_tracker.result(),
            "AdvGlossX": AdvGlossX_tracker.result(),"AdvGlossC": AdvGlossC_tracker.result(),"AdvGlossS": AdvGlossS_tracker.result(),
            "AdvGlossN": AdvGlossN_tracker.result(),"RecGlossX": RecGlossX_tracker.result(),"RecGlossC": RecGlossC_tracker.result(),
            "RecGlossS": RecGlossS_tracker.result(), "Qloss": Qloss_tracker.result(),
            "fakeX":tf.math.reduce_mean(X_fakecritic),"X":tf.math.reduce_mean(X_critic),
            "c_fake":tf.math.reduce_mean(c_fakecritic),"c":tf.math.reduce_mean(c_critic),"n_fake":tf.math.reduce_mean(n_fakecritic),
            "n_prior":tf.math.reduce_mean(n_priorcritic),"s_fake":tf.math.reduce_mean(s_fakecritic),"s_prior":tf.math.reduce_mean(s_priorcritic)}

    def call(self, X):
        [_,_,s_fake,c_fake,n_fake] = self.Fx(X)
        X_rec = self.Gz((s_fake,c_fake,n_fake))
        return X_rec, c_fake, s_fake, n_fake

    def plot(self,X,c):
        [_,_,s_fake,c_fake,n_fake] = self.Fx(X,training=False)
        s_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],self.latentSdim])
        n_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],self.latentNdim])
        fakeX = self.Gz((s_prior,c,n_prior),training=False)
        X_rec = self.Gz((s_fake,c_fake,n_fake),training=False)
        return X_rec, c_fake, s_fake, n_fake, fakeX

    def label_predictor(self, X, c):
        [_,_,s_fake,c_fake,n_fake] = self.Fx(X)
        s_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[s_fake.shape[0],self.latentSdim])
        n_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[n_fake.shape[0],self.latentNdim])
        fakeX = self.Gz((s_prior,c,n_prior),training=False)
        [_,_,_,c_rec,_] = self.Fx(fakeX,training=False)
        return c_fake, c_rec
    
    def distribution(self,X,c):
        [μs_fake,σs2_fake,s_fake,c_fake,n_fake] = self.Fx(X,training=False)
        s_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],self.latentSdim])
        n_prior = tf.random.normal(mean=0.0,stddev=1.0,shape=[X.shape[0],self.latentNdim])
        fakeX = self.Gz((s_prior,c,n_prior),training=False)
        [μs_rec,σs2_rec,s_rec,c_rec,n_rec] = self.Fx(fakeX,training=False)
        return s_prior, n_prior, s_fake, n_fake, s_rec, n_rec, μs_fake, σs2_fake, μs_rec, σs2_rec

    def generate(self, X, c_fake_new):
        [_,_,s_fake,c_fake,n_fake] = self.Fx(X)
        X_rec_new = self.Gz((s_fake,c_fake_new,n_fake),training=False)
        return X_rec_new

    # BN : do not apply batchnorm to the generator output layer and the discriminator input layer

    def BuildFx(self):
        """
            kl.Conv1D Fx structure
        """
        # To build this model using the functional API

        # kl.Input layer
        X = kl.Input(shape=self.Xshape,name="X")

        # Initial CNN layer
        layer = -1
        h = kl.Conv1D(self.nZfirst, 
                self.kernel,1,padding="same",
                data_format="channels_last",name="FxCNN0")(X)
        h = kl.BatchNormalization(momentum=0.95)(h)
        h = kl.LeakyReLU(alpha=0.1,name="FxA0")(h)
        h = kl.Dropout(0.2,name="FxDO0")(h)

        # Common encoder CNN layers
        for layer in range(self.nAElayers):
            h = kl.Conv1D(self.nZfirst*self.stride**(layer+1),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
            h = kl.BatchNormalization(momentum=0.95)(h)
            h = kl.LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
            h = kl.Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)

        # Last common CNN layer (no stride, same channels) before branching
        layer = self.nAElayers
        h = kl.Conv1D(self.nZchannels,
            self.kernel,1,padding="same",
            data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
        h = kl.BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
        h = kl.LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
        z = kl.Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)
        # z ---> Zshape = (Zsize,nZchannels)

        layer = 0
        # variable s
        # s-average
        h_μs = kl.Conv1D(self.nZchannels*self.Sstride**(layer+1),
            self.Skernel,self.Sstride,padding="same",
            data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(z)
        h_μs = kl.BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(h_μs)
        h_μs = kl.LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(h_μs)
        h_μs = kl.Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(h_μs)

        # s-log std
        h_σs2 = kl.Conv1D(self.nZchannels*self.Sstride**(layer+1),
            self.Skernel,self.Sstride,padding="same",
            data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(z)
        h_σs2 = kl.BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(h_σs2)
        h_σs2 = kl.LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(h_σs2)
        h_σs2 = kl.Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(h_σs2)

        # variable c
        h_c = kl.Conv1D(self.nZchannels*self.Cstride**(layer+1),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(z)
        h_c = kl.BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(h_c)
        h_c = kl.LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(h_c)
        #h_c = tfa.layers.InstanceNormalization()(h_c)
        h_c = kl.Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(h_c)

        # variable n
        h_n = kl.Conv1D(self.nZchannels*self.Nstride**(layer+1),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(z)
        h_n = kl.BatchNormalization(momentum=0.95)(h_n)
        h_n = kl.LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(h_n)
        #h_n = tfa.layers.InstanceNormalization()(h_n)
        h_n = kl.Dropout(0.2,name="FxDON{:>d}".format(layer+1))(h_n)

        # variable s
        for layer in range(1,self.nSlayers):
            # s-average
            h_μs = kl.Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(h_μs)
            h_μs = kl.BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(h_μs)
            h_μs = kl.LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(h_μs)
            h_μs = kl.Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(h_μs)

            # s-log std
            h_σs2 = kl.Conv1D(self.nZchannels*self.Sstride**(layer+1),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(h_σs2)
            h_σs2 = kl.BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(h_σs2)
            h_σs2 = kl.LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(h_σs2)
            h_σs2 = kl.Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(h_σs2)

        # variable c
        for layer in range(1,self.nClayers):
            h_c = kl.Conv1D(self.nZchannels*self.Cstride**(layer+1),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(h_c)
            h_c = kl.BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(h_c)
            h_c = kl.LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(h_c)
            #h_c = tfa.layers.InstanceNormalization()(h_c)
            h_c = kl.Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(h_c)

        # variable n
        for layer in range(1,self.nNlayers):
            h_n = kl.Conv1D(self.nZchannels*self.Nstride**(layer+1),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(h_n)
            h_n = kl.BatchNormalization(momentum=0.95)(h_n)
            h_n = kl.LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(h_n)
            #h_n = tfa.layers.InstanceNormalization()(h_n)
            h_n = kl.Dropout(0.2,name="FxDON{:>d}".format(layer+1))(h_n)

        # variable s
        h_μs = kl.Flatten(name="FxFLmuS{:>d}".format(layer+1))(h_μs)
        h_μs = kl.Dense(1024)(h_μs)
        h_μs = kl.BatchNormalization(momentum=0.95)(h_μs)
        h_μs = kl.LeakyReLU(alpha=0.1)(h_μs)
        h_μs = kl.Dense(self.latentSdim,name="FxFWmuS")(h_μs)
        μs = kl.BatchNormalization(momentum=0.95,name="FxBNmuS")(h_μs)
        #h_μs = kl.BatchNormalization(momentum=0.95,name="FxBNmuS")(h_μs)
        #μs = kl.LeakyReLU(alpha=0.1)(h_μs)

        # s-sigma
        h_σs2 = kl.Flatten(name="FxFLlvS{:>d}".format(layer+1))(h_σs2)
        h_σs2 = kl.Dense(1024)(h_σs2)
        h_σs2 = kl.BatchNormalization(momentum=0.95)(h_σs2)
        h_σs2 = kl.LeakyReLU(alpha=0.1)(h_σs2)
        h_σs2 = kl.Dense(self.latentSdim,name="FxFWlvS")(h_σs2)
        h_σs2 = kl.BatchNormalization(momentum=0.95,axis=-1,name="FxBNlvS")(h_σs2)
        #h_σs2 = kl.LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+2))(h_σs2)
        if 'σs2' in self.σs2:
            σs2 = tf.math.sigmoid(h_σs2)
        else:
            σs2 = tf.keras.activations.softplus(h_σs2)

        

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
        h_n = kl.Flatten(name="FxFLN{:>d}".format(layer+1))(h_n)
        h_n = kl.Dense(1024)(h_n)
        h_n = kl.BatchNormalization(momentum=0.95)(h_n)
        h_n = kl.LeakyReLU(alpha=0.1)(h_n)
        h_n = kl.Dense(self.latentNdim,name="FxFWN")(h_n)

        # variable s
        s = sampleS()([μs,σs2])

        # variable c
        #c = kl.Dense(self.latentCdim,activation=tf.keras.activations.softmax)(h_c)
        c = kl.Softmax()(h_c)

        # variable n
        n = kl.BatchNormalization(momentum=0.95)(h_n)
        #n = tfa.layers.InstanceNormalization()(h_n)

        Fx = tf.keras.Model(X,[μs,σs2,s,c,n],name="Fx")

        return Fx

    def BuildGz(self):
        """
            kl.Conv1D Gz structure
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

        for layer in range(1,self.nSlayers):
            h_s = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nSchannels*self.Sstride**(-layer)),
                self.Skernel,self.Sstride,padding="same",
                data_format="channels_last"))(h_s)
            h_s = kl.LeakyReLU(alpha=0.1)(h_s)
            h_s = kl.BatchNormalization(momentum=0.95)(h_s)
            #h_s = kl.Dropout(0.2,name="GzDOS{:>d}".format(layer))(h_s)
        h_s = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nSchannels*self.Sstride**(-self.nSlayers)),
            self.Skernel,self.Sstride,padding="same",
            data_format="channels_last"))(h_s)
        h_s = kl.BatchNormalization(momentum=0.95,name="GzBNS{:>d}".format(self.nSlayers))(h_s)
        h_s = kl.LeakyReLU(alpha=0.1)(h_s)
        #h_s = kl.Dropout(0.2)(h_s)
        GzS = tf.keras.Model(s,h_s)


        # variable c
        h_c = tfa.layers.SpectralNormalization(kl.Dense(self.Csize*self.nCchannels))(c)
        h_c = kl.BatchNormalization(momentum=0.95)(h_c)
        h_c = kl.LeakyReLU(alpha=0.1,)(h_c)
        h_c = kl.Reshape((self.Csize,self.nCchannels))(h_c)
        for layer in range(1,self.nClayers):
            h_c = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nCchannels*self.Cstride**(-layer)),
                self.Ckernel,self.Cstride,padding="same",
                data_format="channels_last"))(h_c)
            h_c = kl.BatchNormalization(momentum=0.95)(h_c)
            h_c = kl.LeakyReLU(alpha=0.1)(h_c)
            #h_c = kl.Dropout(0.2)(h_c)
        h_c = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nCchannels*self.Cstride**(-self.nClayers)),
            self.Ckernel,self.Cstride,padding="same",
            data_format="channels_last"))(h_c)
        h_c = kl.BatchNormalization(momentum=0.95)(h_c)
        h_c = kl.LeakyReLU(alpha=0.1)(h_c)
        #h_c = kl.Dropout(0.2)(h_c)
        GzC = tf.keras.Model(c,h_c)

        # variable n
        h_n = tfa.layers.SpectralNormalization(kl.Dense(self.Nsize*self.nNchannels))(n)
        h_n = kl.BatchNormalization(momentum=0.95)(h_n)
        h_n = kl.LeakyReLU(alpha=0.1)(h_n)
        h_n = kl.Reshape((self.Nsize,self.nNchannels))(h_n)
        for layer in range(1,self.nNlayers):
            h_n = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nNchannels*self.Nstride**(-layer)),
                self.Nkernel,self.Nstride,padding="same",
                data_format="channels_last"))(h_n)
            h_n = kl.LeakyReLU(alpha=0.1)(h_n)
            h_n = kl.BatchNormalization(momentum=0.95)(h_n)
            #h_n = kl.Dropout(0.2)(h_n)
        h_n = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(int(self.nNchannels*self.Nstride**(-self.nNlayers)),
            self.Nkernel,self.Nstride,padding="same",
            data_format="channels_last"))(h_n)
        h_n = kl.BatchNormalization(momentum=0.95)(h_n)
        h_n = kl.LeakyReLU(alpha=0.1)(h_n)
        #h_n = kl.Dropout(0.2)(h_n)
        GzN = tf.keras.Model(n,h_n)

        Gz = kl.concatenate([GzS.output,GzC.output,GzN.output])
        Gz = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels,
                self.kernel,1,padding="same",
                data_format="channels_last"))(Gz)
        Gz = kl.BatchNormalization(axis=-1,momentum=0.95)(Gz)
        Gz = kl.LeakyReLU(alpha=0.1)(Gz)

        for layer in range(self.nAElayers-1):
            Gz = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                self.kernel,self.stride,padding="same",use_bias=False))(Gz)
            Gz = kl.BatchNormalization(axis=-1,momentum=0.95)(Gz)
            Gz = kl.LeakyReLU(alpha=0.1)(Gz)

        layer = self.nAElayers-1
        Gz = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                self.kernel,self.stride,padding="same",use_bias=False))(Gz)
        Gz = kl.BatchNormalization(axis=-1,momentum=0.95)(Gz)
        Gz = kl.LeakyReLU(alpha=0.1,name="GzA{:>d}".format(layer+1))(Gz)

        layer = self.nAElayers
        X = tfa.layers.SpectralNormalization(kl.Conv1DTranspose(self.nXchannels,self.kernel,1,
            padding="same",activation='tanh',use_bias=False))(Gz)

        Gz = tf.keras.Model(inputs=[GzS.input,GzC.input,GzN.input],outputs=X,name="Gz")
        return Gz

    def BuildDx(self):
        """
            kl.Conv1D discriminator structure
        """
        layer = 0
        X = kl.Input(shape=self.Xshape,name="X")
        h = tfa.layers.SpectralNormalization(kl.Conv1D(self.Xsize*self.stride**(-(layer+1)),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="DxCNN0"))(X)
        #h = kl.LayerNormalization(axis=[1,2])(h) #temp
        h = kl.LeakyReLU(alpha=0.1,name="DxA0")(h)
        h = kl.Dropout(0.25)(h)
        
        for layer in range(1,self.nDlayers):
            h = tfa.layers.SpectralNormalization(kl.Conv1D(self.Xsize*self.stride**(-(layer+1)),
                self.kernel,self.stride,padding="same",
                data_format="channels_last",name="DxCNN{:>d}".format(layer)))(h)
            #h = kl.LayerNormalization(axis=[1,2],name="DxLN{:>d}".format(layer))(h) #temp
            h = kl.LeakyReLU(alpha=0.1,name="DxA{:>d}".format(layer))(h)
            h = kl.Dropout(0.25)(h)
            
        layer = self.nDlayers    
        h = kl.Flatten(name="DxFL{:>d}".format(layer))(h)
        h = tfa.layers.SpectralNormalization(kl.Dense(1024))(h)
        #h = kl.LayerNormalization()(h)
        h = kl.LeakyReLU(alpha=0.1)(h)
        h = kl.Dropout(0.25)(h)
        #Px = tfa.layers.SpectralNormalization(kl.Dense(1,activation='linear'))(h)
        Px = tfa.layers.SpectralNormalization(kl.Dense(1,activation='sigmoid'))(h)
        Dx = tf.keras.Model(X,Px,name="Dx")
        return Dx


    def BuildDc(self):
        """
            kl.Dense discriminator structure
        """
        c = kl.Input(shape=(self.latentCdim,))
        if 'WGAN' in self.discriminator:
            h = kl.Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(c)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = kl.Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(h)
            h = kl.BatchNormalization(momentum=0.95)(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Pc = kl.Dense(1,activation=tf.keras.activations.linear,
                                    kernel_constraint=ClipConstraint(self.clipValue))(h)
        else:
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(c)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Pc = tfa.layers.SpectralNormalization(kl.Dense(1,activation=tf.keras.activations.linear))(h)
        Dc = tf.keras.Model(c,Pc,name="Dc")
        return Dc


    def BuildDn(self):
        """
            kl.Dense discriminator structure
        """
        n = kl.Input(shape=(self.latentNdim,))
        if 'WGAN' in self.discriminator:
            h = kl.Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(n) 
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = kl.Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(h)
            h = kl.BatchNormalization(momentum=0.95)(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Pn = kl.Dense(1,activation=tf.keras.activations.linear,
                                    kernel_constraint=ClipConstraint(self.clipValue))(h)
        else:
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(n)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Pn = tfa.layers.SpectralNormalization(kl.Dense(1,activation=tf.keras.activations.linear))(h)
        Dn = tf.keras.Model(n,Pn,name="Dn")
        return Dn

    def BuildDs(self):
        """
            kl.Dense discriminator structure
        """
        s = kl.Input(shape=(self.latentSdim,))
        if 'WGAN' in self.discriminator:
            h = kl.Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(s)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = kl.Dense(3000,kernel_constraint=ClipConstraint(self.clipValue))(h)
            h = kl.BatchNormalization(momentum=0.95)(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Ps = kl.Dense(1,activation=tf.keras.activations.linear,
                                    kernel_constraint=ClipConstraint(self.clipValue))(h)
        else:
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(s)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            h = tfa.layers.SpectralNormalization(kl.Dense(3000))(h)
            h = kl.LeakyReLU(alpha=0.1)(h)
            h = kl.Dropout(0.25)(h)
            Ps = tfa.layers.SpectralNormalization(kl.Dense(1,activation=tf.keras.activations.linear))(h)
        Ds = tf.keras.Model(s,Ps,name="Ds")
        return Ds
    
    def DumpModels(self):
        self.Fx.save(self.checkpoint_dir + "/Fx",save_format="tf")
        self.Gz.save(self.checkpoint_dir + "/Gz",save_format="tf")
        self.Dx.save(self.checkpoint_dir + "/Dx",save_format="tf")
        self.Ds.save(self.checkpoint_dir + "/Ds",save_format="tf")
        self.Dn.save(self.checkpoint_dir + "/Dn",save_format="tf")
        self.Dc.save(self.checkpoint_dir + "/Dc",save_format="tf")
        return
