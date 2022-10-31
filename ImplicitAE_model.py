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
import tensorflow_addons.layers as tfal

tfd = tfp.distributions

loss_names = [
    "AdvDLossXz",
    "AdvDLossZ",
    # "AdvDlossC",
    # "AdvDlossS",
    # "AdvDlossN",
    "AdvGlossXz",
    # "AdvGlossC",
    # "AdvGlossS",
    # "AdvGlossN",
    "AdvGlossZ",
    "RecXloss",]
    # "RecCloss",
    # "RecSloss",
    # "FakeCloss"]

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


class LatentCodeConditioner(kl.Layer):

    def __init__(self, 
                 LCC: str, 
                 batch_size: int,
                 latent_length : int =128, 
                 n_channels : int =1,
                 SN: bool = False,
                 name : str = "LCC"):
        """_summary_

        Args:
            LCC (str): _description_
            batch_size (int): _description_
            latent_length (int, optional): _description_. Defaults to 128.
            n_channels (int, optional): _description_. Defaults to 1.
            name (str, optional): _description_. Defaults to "LCC".
        """        
        super(LatentCodeConditioner, self).__init__(name=name)
        
        self.LCC = LCC
        self.batch_size = batch_size
        self.latent_length = latent_length
        self.n_channels = n_channels
        
        self.model = []        
        if LCC == "LDC":
            if SN:
                self.model = tf.keras.Sequential(
                    [tfal.SpectralNormalization(
                        kl.Dense(1000, activation="relu", name="{:s}FW0".format(name))
                        ),
                     tfal.SpectralNormalization(
                         kl.Dense(latent_length, name="{:s}FW1".format(name))
                         ),
                     tfal.SpectralNormalization(kl.Reshape(
                         (self.latent_length, 1), name="{:s}Rs0".format(name))
                                                )
                     ]
                )
            else:
                self.model=tf.keras.Sequential(
                    [kl.Dense(1000, activation="relu", name="{:s}FW0".format(name)),
                     kl.Dense(latent_length, name="{:s}FW1".format(name)),
                     kl.Reshape((self.latent_length, 1),name="{:s}Rs0".format(name))
                    ]
                )
        elif LCC == "LIC":
            if SN:
                self.model = tfal.SpectralNormalization(
                    kl.Dense(n_channels, name="{:s}FW1".format(name))
                )
            else:
                self.model = kl.Dense(n_channels, name="{:s}FW1".format(name))
                

    def call(self, z):
        
        rz = self.model(z) 
        
        return tf.broadcast_to(rz, shape=(-1,self.latent_length, self.n_channels))

class ImplicitAE(tf.keras.Model):

    def __init__(self,options):
        super(ImplicitAE, self).__init__()
        """
            Setup
        """
        self.__dict__.update(options)
        
        # Create Metric instances to track the losses
        self.loss_trackers = {"{:>s}_tracker".format(l): km.Mean(name=l) for l in loss_names}
        self.loss_val = {"{:>s}".format(l): 0.0 for l in loss_names}
        # define the constraint
        self.ClipD = ClipConstraint(0.01)
        
        self.px = tfd.MultivariateNormalDiag(
            loc = tf.zeros(shape = self.Xshape, dtype = tf.float32),
            scale_diag = tf.ones(shape = self.Xshape, dtype = tf.float32)
            )
        self.px_flat = tfd.MultivariateNormalDiag(
            loc = tf.zeros(shape = self.Xsize*self.nXchannels, dtype = tf.float32),
            scale_diag = tf.ones(shape = self.Xsize*self.nXchannels, dtype = tf.float32)
            )
        self.pz = tfd.MultivariateNormalDiag(
            loc = tf.zeros(shape = self.Zshape, dtype = tf.float32),
            scale_diag = tf.ones(shape = self.Zshape, dtype = tf.float32)
            )
        self.pz_flat = tfd.MultivariateNormalDiag(
            loc = tf.zeros(shape = self.latentZdim, dtype = tf.float32),
            scale_diag = tf.ones(shape = self.latentZdim, dtype=tf.float32)
            )
        
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
        self.Dx = self.BuildDz()
        self.Dxz = self.BuildDxz()
        """
            Build Fx/Gz (generators)
        """
        self.Fx = self.BuildFx()
        self.Gz = self.BuildGz()
        
        # self.models = [self.Dx, self.Dz, self.Dxz, self.Fx, self.Gz]
        self.models = [self.Dx, self.Dz, self.Dxz, self.Fx, self.Gz]
    
    def compile(self,optimizers,losses,**kwargs):
        
        super(ImplicitAE, self).compile(**kwargs)
        """
            Optimizers
        """
        self.__dict__.update(optimizers)
        """
            Losses
        """
        self.__dict__.update(losses)
            
    # @tf.function
    def train_XZX(self, X, c_prior):
        """
        
        """
        # Sample factorial prior Z
        z_prior = self.pz.sample(self.batchSize)
        ε = self.px_flat.sample(self.batchSize)
        n = self.pz_flat.sample(self.batchSize)
        
        # Train discriminative part
        for _ in range(self.nCritic):
            
            # Tape gradients
            with tf.GradientTape(persistent=True) as tape:

                # Encode real signals X
                z_hat = self.Fx((X,ε), training=True)
                
                # Decode reconstructed signals X_hat
                X_hat = self.Gz((z_hat,n), training=True)
                
                # Discriminates real (X,z_hat) from false (X_hat,z_hat)
                Dxz_real = self.Dxz((X, z_hat), training=True)
                Dxz_fake = self.Dxz((X_hat, z_hat), training=True)

                AdvDlossXz = self.AdvDlossXz(Dxz_real,Dxz_fake)
                # Discriminates real z from false z_hat
                Dxz_real = self.Dz(z, training=True)
                Dxz_fake = self.Dz(z_hat, training=True)
                
                # Compute XZX adversarial loss (JS(x,z))
                AdvDlossZ = self.AdvDlossZ(Dz_real, Dz_fake)

                AdvLossD = AdvLossXz+AdvLossZ
                
            # Compute the discriminator gradient
            gradDxz_w, gradDz_w = tape.gradient(AdvLossD,
                                                (self.Dxz.trainable_variables, 
                                                 self.Dz.trainable_variables),
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

    # @tf.function
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

    @tf.function
    def train_step(self, XC):
        if isinstance(XC, tuple):
            X, metadata = XC
            damage_class, magnitude, damage_index = metadata

        self.batchSize = tf.shape(X)[0]

        for _ in range(self.nXRepX):
            XZXout = self.train_XZX(X, damage_class)
        
        for _ in range(self.nRepXRep):  
            ZXZout = self.train_ZXZ(X, damage_class)
        

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
            Fx encoder structure CNN
        """
        # To build this model using the functional API

        # Input layer
        X = kl.Input(shape = self.Xshape, 
                     name="X")
        ε = kl.Input(shape = self.Xshape[0]*self.Xshape[1], 
                     name="ε")

        h_ε = LatentCodeConditioner(LCC=self.LCC,
                                    batch_size=self.batchSize,
                                    latent_length=self.Xshape[0],
                                    n_channels=self.nZfirst,
                                    SN=self.FxSN,
                                    name="FxLCC")(ε)
        if self.FxSN:
            # Initial CNN layer
            h_X = tfal.SpectralNormalization(kl.Conv1D(self.nZfirst, 
                                                       self.kernel, 
                                                       1, 
                                                       padding="same",
                                                       data_format="channels_last", 
                                                       name="FxCNN0"))(X)
            h_X = kl.Add()([h_X, h_ε])
            h_X = kl.BatchNormalization(momentum=0.95)(h_X)
            h_X = kl.LeakyReLU(alpha=0.1, 
                               name="FxA0")(h_X)
            h_X = kl.Dropout(self.dpout, 
                             name="FxDO0")(h_X)

            # Common encoder CNN layers
            for l in range(self.nAElayers):
                h_X = tfal.SpectralNormalization(kl.Conv1D(self.nZfirst*self.stride**(l+1),
                            self.kernel, 
                            self.stride, 
                            padding="same",
                            data_format="channels_last", 
                            name="FxCNN{:>d}".format(l+1)))(h_X)
                h_X = kl.BatchNormalization(momentum=0.95)(h_X)
                h_X = kl.LeakyReLU(alpha=0.1, 
                                   name="FxA{:>d}".format(l+1))(h_X)
                h_X = kl.Dropout(self.dpout,
                                 name="FxDO{:>d}".format(l+1))(h_X)

            # Last common CNN layer (no stride, same channels) before branching
            l = self.nAElayers
            h_X = tfal.SpectralNormalization(kl.Conv1D(self.nZchannels,
                        self.kernel, 
                        1, 
                        padding="same",
                        data_format="channels_last", 
                        name="FxCNN{:>d}".format(l+1)))(h_X)
            h_X = kl.BatchNormalization(momentum=0.95, 
                                        name="FxBN{:>d}".format(l+1))(h_X)
            h_X = kl.LeakyReLU(alpha=0.1, 
                               name="FxA{:>d}".format(l+1))(h_X)
            z = kl.Dropout(self.dpout, 
                           name="FxDO{:>d}".format(l+1))(h_X)
        else:        
            # Initial CNN layer        
            h_X = kl.Conv1D(self.nZfirst,
                            self.kernel,
                            1,
                            padding="same",
                            data_format="channels_last",
                            name="FxCNN0")(X)
            h_X = kl.Add()([h_X, h_ε])
            h_X = kl.BatchNormalization(momentum=0.95)(h_X)
            h_X = kl.LeakyReLU(alpha=0.1,
                               name="FxA0")(h_X)
            h_X = kl.Dropout(self.dpout,
                             name="FxDO0")(h_X)

            # Common encoder CNN layers
            for l in range(self.nAElayers):
                h_X = kl.Conv1D(self.nZfirst*self.stride**(l+1),
                    self.kernel,
                    self.stride,
                    padding="same",
                    data_format="channels_last",
                    name="FxCNN{:>d}".format(l+1))(h_X)
                h_X = kl.BatchNormalization(momentum=0.95)(h_X)
                h_X = kl.LeakyReLU(alpha=0.1,
                                   name="FxA{:>d}".format(l+1))(h_X)
                h_X = kl.Dropout(self.dpout,
                                 name="FxDO{:>d}".format(l+1))(h_X)

            # Last common CNN layer (no stride, same channels) before branching
            l = self.nAElayers
            h_X = kl.Conv1D(self.nZchannels,
                            self.kernel,
                            1,
                            padding="same",
                            data_format="channels_last",
                            name="FxCNN{:>d}".format(l+1))(h_X)
            h_X = kl.BatchNormalization(momentum=0.95,
                                        name="FxBN{:>d}".format(l+1))(h_X)
            h_X = kl.LeakyReLU(alpha=0.1,
                               name="FxA{:>d}".format(l+1))(h_X)
            z = kl.Dropout(self.dpout,
                           name="FxDO{:>d}".format(l+1))(h_X)
        # z ---> Zshape = (Zsize,nZchannels)

        # Option linear block
        # z = kl.Flatten(name="FxFLN{:>d}".format(l+1))(z)
        # z = kl.Dense(1024)(z)
        # z = kl.BatchNormalization(momentum=0.95)(z)
        # z = kl.LeakyReLU(alpha=0.1)(z)
        # z = kl.Dense(self.latentNdim,name="FxFWN")(z)

        F_x = tf.keras.Model(X, z, name="Fx")
        return F_x

    def BuildGz(self):
        """
            Conv1D Gz structure CNN
        """
    
        z = kl.Input(shape=self.Zshape, 
                     name="z")
        n = kl.Input(shape=self.Zshape[0]*self.Zshape[1], 
                     name="n")
        h_n = LatentCodeConditioner(LCC=self.LCC,
                                    batch_size=self.batchSize,
                                    latent_length=Zshape[1],
                                    n_channels=1,
                                    name="GzLCC")(n)
    
        if self.GzSN:
            h_z = tfal.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels,
                    self.kernel, 
                    1, 
                    padding="same", 
                    data_format="channels_last"))(z)
            h_z = kl.Add()([h_z, h_n])
            h_z = kl.BatchNormalization(axis=-1, 
                                        momentum=0.95)(h_z)
            h_z = kl.LeakyReLU(alpha=0.1)(h_z)
        
            for l in range(1, self.nAElayers-1):
                h_z = tfal.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                    self.kernel, 
                    self.stride, 
                    padding="same", 
                    use_bias=False))(h_z)
                h_z = kl.BatchNormalization(axis=-1, 
                                            momentum=0.95)(h_z)
                h_z = kl.LeakyReLU(alpha=0.1)(h_z)

            layer = self.nAElayers-1

            h_z = tfal.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                                                                self.kernel, 
                                                                self.stride, 
                                                                padding="same", 
                                                                use_bias=False))(h_z)
            h_z = kl.BatchNormalization(axis=-1, 
                                        momentum=0.95)(h_z)
            h_z = kl.LeakyReLU(alpha=0.1, 
                               name="GzA{:>d}".format(layer+1))(h_z)

            layer = self.nAElayers

            X = tfal.SpectralNormalization(kl.Conv1DTranspose(self.nXchannels, 
                                                              self.kernel, 
                                                              1, 
                                                              padding="same", 
                                                              activation='tanh', 
                                                              use_bias=False))(h_z)
        else:
            h_z = kl.Conv1DTranspose(self.nZchannels, 
                                     self.kernel, 
                                     1, 
                                     padding="same", 
                                     data_format="channels_last")(z)
            h_z = kl.Add()([h_z, hn])
            h_z = kl.BatchNormalization(axis=-1, 
                                        momentum=0.95)(h_z)
            h_z = kl.LeakyReLU(alpha=0.1)(h_z)

            for l in range(1, self.nAElayers-1):
                h_z = kl.Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                                         self.kernel, 
                                         self.stride, 
                                         padding="same", 
                                         use_bias=False)(h_z)
                h_z = kl.BatchNormalization(axis=-1, 
                                            momentum=0.95)(h_z)
                h_z = kl.LeakyReLU(alpha=0.1)(h_z)

            layer = self.nAElayers-1

            h_z = kl.Conv1DTranspose(self.nZchannels//self.stride**(layer+1),
                                     self.kernel, 
                                     self.stride, 
                                     padding="same", 
                                     use_bias=False)(h_z)
            h_z = kl.BatchNormalization(axis=-1, 
                                        momentum=0.95)(h_z)
            h_z = kl.LeakyReLU(alpha=0.1,
                               name="GzA{:>d}".format(layer+1))(h_z)

            layer = self.nAElayers

            X = tfal.SpectralNormalization(kl.Conv1DTranspose(self.nXchannels, 
                                                              self.kernel, 
                                                              1,
                                                              padding="same", 
                                                              activation='tanh', 
                                                              use_bias=False))(h_z)
        G_z = tf.keras.Model(inputs=z,
                             outputs=X,
                             name="Gz")
        return G_z

    def BuildDz(self):
        """
            Conv1D discriminator structure
        """
        layer = 0
        z = kl.Input(shape=self.Zshape,name="z")
        
        if self.DzSN:
            h_z = tfal.SpectralNormalization(kl.Conv1D(self.Zsize*self.stride**(-(layer+1)),
                                                       self.kernel,
                                                       self.stride,
                                                       padding="same",
                                                       data_format="channels_last",
                                                       name="DzCNN0"))(z)
            h_z = kl.LeakyReLU(alpha=0.1,
                              name="DzA0")(h_z)
            h_z = kl.Dropout(self.dpout)(h_z)

            for l in range(1,self.nDlayers):
                h_z = tfal.SpectralNormalization(kl.Conv1D(self.Zsize*self.stride**(-(layer+1)),
                                                           self.kernel,
                                                           self.stride,
                                                           padding="same",
                                                           data_format="channels_last",
                                                           name="DzCNN{:>d}".format(l+1)))(h_z)
                h_z = kl.LeakyReLU(alpha=0.1,
                                   name="DzA{:>d}".format(l+1))(h_z)
                h_z = kl.Dropout(self.dpout)(h_z)

            layer = self.nDlayers    
            h_z = kl.Flatten(name="DzFL{:>d}".format(l+1))(h_z)
            h_z = tfal.SpectralNormalization(kl.Dense(1024))(h_z)
            h_z = kl.LeakyReLU(alpha=0.1)(h_z)
            h_z = kl.Dropout(self.dpout)(h_z)
            P_z = tfal.SpectralNormalization(kl.Dense(1))(h_z)
        else:
            h_z = kl.Conv1D(self.Zsize*self.stride**(-(layer+1)),
                            self.kernel,
                            self.stride,
                            padding="same",
                            data_format="channels_last",
                            name="DzCNN0")(z)
            h_z = kl.LeakyReLU(alpha=0.1,
                             name="DzA0")(h_z)
            h_z = kl.Dropout(self.dpout)(h_z)

            for l in range(1,self.nDlayers):
                h_z = kl.Conv1D(self.Zsize*self.stride**(-(layer+1)),
                                self.kernel,
                                self.stride,
                                padding="same",
                                data_format="channels_last",
                                name="DzCNN{:>d}".format(l+1))(h_z)
                h_z = kl.LeakyReLU(alpha=0.1,
                                 name="DzA{:>d}".format(l+1))(h_z)
                h_z = kl.Dropout(self.dpout)(h_z)

            layer = self.nDlayers    
            h_z = kl.Flatten(name="DxFL{:>d}".format(l+1))(h_z)
            h_z = kl.Dense(1024)(h_z)
            h_z = kl.LeakyReLU(alpha=0.1)(h_z)
            h_z = kl.Dropout(self.dpout)(h_z)
            P_z = kl.Dense(1)(h_z)            
        
        D_z = tf.keras.Model(z, P_z, name="Dz")
        return D_z

    def BuildDxz(self):
        """
            Conv1D discriminator structure
        """
        X = kl.Input(shape=self.Xshape, name="X")
        z = kl.Input(shape=self.Zshape, name="z")

        k = self.kernel
        s = self.stride

        l = 0
        if self.DxzSN:
            # Branch Dx
            f = int(self.Xsize*1**(-(l+1)))
            h_X = tfal.SpectralNormalization(kl.Conv1D(f, k, strides=1,
                                                       padding="same",
                                                       data_format="channels_last",
                                                       name="DxCNN0"))(X)
            h_X = kl.LeakyReLU(alpha=0.1, name="DxA0")(h_X)
            h_X = kl.Dropout(self.dpout)(h_X)

            # Branch Dz
            f = int(self.Zsize*1**(-(l+1)))
            h_z = tfal.SpectralNormalization(kl.Conv1D(f, k, strides=1,
                                                       padding="same",
                                                       data_format="channels_last",
                                                       name="DzCNN0"))(z)
            h_z = kl.LeakyReLU(alpha=0.1, name="DzA0")(h_z)
            h_z = kl.Dropout(self.dpout)(h_z)
        else:
            # Branch Dx
            f = int(self.Xsize*s**(-(l+1)))
            h_X = kl.Conv1D(f,
                            k,
                            strides=1,
                            padding="same",
                            data_format="channels_last",
                            name="DxCNN0")(X)
            h_X = kl.LeakyReLU(alpha=0.1, name="DxA0")(h_X)
            h_X = kl.Dropout(self.dpout)(h_X)

            # Branch Dz
            f = int(self.Zsize*s**(-(l+1)))
            h_z = kl.Conv1D(f,
                            k,
                            strides=1,
                            padding="same",
                            data_format="channels_last",
                            name="DzCNN0")(z)
            h_z = kl.LeakyReLU(alpha=0.1, name="DzA0")(h_z)
            h_z = kl.Dropout(self.dpout)(h_z)

        nlayers_z = int(tf.math.log(self.Xsize/self.Zsize)/tf.math.log(float(s)))
                
        if self.DxzSN:
            for l in range(nlayers_z):
                # Branch Dx
                f = int(self.Xsize*s**(-(l+1)))
                h_X = tfal.SpectralNormalization(kl.Conv1D(f,
                                                           k,
                                                           strides=s,
                                                           padding="same",
                                                           data_format="channels_last",
                                                           name="DxCNN{:>d}".format(l+1)))(h_X)
                h_X = kl.LeakyReLU(alpha=0.1, 
                                   name="DxA{:>d}".format(l+1))(h_X)
                h_X = kl.Dropout(self.dpout)(h_X)
                
            # # Branch z
            # h_z = tfal.SpectralNormalization(kl.Conv1D(self.Xsize*self.stride**(-(l+1)),
            #                                            self.kernel,
            #                                            1,
            #                                            padding="same",
            #                                            data_format="channels_last",
            #                                            name="DzCNN{:>d}".format(l+1)))(h_z)
            # h_z = kl.LeakyReLU(alpha=0.1, name="DzA{:>d}".format(l+1))(h_z)
            # h_z = kl.Dropout(self.dpout)(h_z)
        else:
            for l in range(nlayers_z):
                # Branch Dx
                f = int(self.Xsize*s**(-(l+1)))
                h_X = kl.Conv1D(f,
                                k,
                                strides=s,
                                padding="same",
                                data_format="channels_last",
                                name="DxCNN{:>d}".format(l+1))(h_X)
                h_X = kl.LeakyReLU(alpha=0.1,
                                   name="DxA{:>d}".format(l+1))(h_X)
                h_X = kl.Dropout(self.dpout)(h_X)
            # # Branch z
            # h_z = kl.Conv1D(self.Xsize*self.stride**(-(l+1)),
            #                 self.kernel,
            #                 1,
            #                 padding="same",
            #                 data_format="channels_last",
            #                 name="DzCNN{:>d}".format(l+1))(h_z)
            # h_z = kl.LeakyReLU(alpha=0.1, name="DzA{:>d}".format(l+1))(h_z)
            # h_z = kl.Dropout(self.dpout)(h_z)

        D_x  = keras.Model(X, h_X)
        D_z =  keras.Model(z, h_z)

        h_Xz = kl.concatenate([D_x.output, D_z.output], axis=-1)

        # Branch (X,z)
        if self.DxzSN:
            for l in range(nlayers_z, self.nDlayers):
                h_Xz = tfal.SpectralNormalization(kl.Conv1D(self.Zsize*self.stride**(-(l+1)),
                                                            self.kernel,
                                                            self.stride,
                                                            padding="same",
                                                            data_format="channels_last",
                                                            name="DxzCNN{:>d}".format(l+1)))(h_Xz)
                h_Xz = kl.LeakyReLU(alpha=0.1, name="DzA{:>d}".format(l+1))(h_Xz)
                h_Xz = kl.Dropout(self.dpout)(h_Xz)
            h_Xz = kl.Flatten(name="DxFL{:>d}".format(l+1))(h_Xz)
            h_Xz = tfal.SpectralNormalization(kl.Dense(1024))(h_Xz)
            h_Xz = kl.LeakyReLU(alpha=0.1)(h_Xz)
            h_Xz = kl.Dropout(self.dpout)(h_Xz)
            P_Xz = tfal.SpectralNormalization(kl.Dense(1))(h_Xz)
        else:
            for l in range(nlayers_z, self.nDlayers):
                h_Xz = kl.Conv1D(self.Zsize*self.stride**(-(l+1)),
                                 self.kernel,
                                 self.stride,
                                 padding="same",
                                 data_format="channels_last",
                                 name="DxzCNN{:>d}".format(l+1))(h_Xz)

                h_Xz = kl.LeakyReLU(alpha=0.1, name="DzA{:>d}".format(l+1))(h_Xz)
                h_Xz = kl.Dropout(self.dpout)(h_Xz)
            h_Xz = kl.Flatten(name="DxFL{:>d}".format(l+1))(h_Xz)
            h_Xz = kl.Dense(1024)(h_Xz)
            h_Xz = kl.LeakyReLU(alpha=0.1)(h_Xz)
            h_Xz = kl.Dropout(self.dpout)(h_Xz)
            P_Xz = kl.Dense(1)(h_Xz)
        
        D_Xz = tf.keras.Model(inputs=[D_x.input, D_z.input], 
                              outputs=P_Xz, 
                              name="Dxz")
        return D_Xz
