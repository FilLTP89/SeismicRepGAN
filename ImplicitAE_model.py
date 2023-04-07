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
import ImplicitAE_layers as il
tfd = tfp.distributions


loss_names = [
    "AdvDlossXz",
    "AdvDlossZ",
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
        self.Dz = self.BuildDz()
        self.Dxz = self.BuildDxz()
        """
            Build Fx/Gz (generators)
        """
        self.Fx = self.BuildFx()
        self.Gz = self.BuildGz()
        
        # self.models = [self.Dx, self.Dz, self.Dxz, self.Fx, self.Gz]
        self.models = [self.Dz, 
                       self.Dxz, 
                       self.Fx, 
                       self.Gz]
    
    def compile(self,
                optimizers,
                losses,
                **kwargs):
        
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
                z_hat = self.Fx((X, ε), training=True)
                
                # Decode reconstructed signals X_hat
                X_hat = self.Gz((z_hat,n), training=True)
                
                # Discriminates real (X,z_hat) from false (X_hat,z_hat)
                Dxz_real = self.Dxz((X, z_hat), training=True)
                Dxz_fake = self.Dxz((X_hat, z_hat), training=True)

                # Compute Z adversarial loss (JS(x,z))
                AdvDlossXz = self.AdvDlossXz(Dxz_real, Dxz_fake)
                
                # Discriminates real z from false z_hat
                Dz_real = self.Dz(z_prior, training=True)
                Dz_fake = self.Dz(z_hat, training=True)
                
                # Compute Z adversarial loss (JS(z))
                AdvDlossZ = self.AdvDlossZ(Dz_real, Dz_fake)

                AdvDloss = AdvDlossXz+AdvDlossZ
                
            # Compute the discriminator gradient
            gradDxz_w, gradDz_w = tape.gradient(AdvDloss,
                                                (self.Dxz.trainable_variables, 
                                                 self.Dz.trainable_variables),
                                                unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Update discriminators' weights
            self.DxzOpt.apply_gradients(
                zip(gradDxz_w,
                    self.Dxz.trainable_variables)
                )
            self.DzOpt.apply_gradients(
                zip(gradDz_w,
                    self.Dz.trainable_variables)
                )

        self.loss_val["AdvDlossXz"] = AdvDlossXz
        self.loss_val["AdvDlossZ"] = AdvDlossZ
            
        # Train the generative part
        for _ in range(self.nGenerator):
            
            # Tape gradients
            with tf.GradientTape(persistent=True) as tape:
                
                # Encode real signals X
                z_hat = self.Fx((X, ε), training=True)

                # Decode reconstructed signals X_hat
                X_hat = self.Gz((z_hat, n), training=True)

                # Discriminates real (X,z_hat) from false (X_hat,z_hat)
                Dxz_fake = self.Dxz((X_hat, z_hat), training=True)

                # Compute Z adversarial loss (JS(x,z))
                AdvGlossXz = self.AdvGlossXz(None, Dxz_fake)

                # Discriminates real z from false z_hat
                Dz_fake = self.Dz(z_hat, training=True)

                # Compute Z adversarial loss (JS(z))
                AdvGlossZ = self.AdvGlossZ(None, Dz_fake)
                                
                # Compute reconstruction loss
                RecXloss = self.RecXloss(X, X_hat)
                
                # Total generator loss
                # Compute total generator loss
                AdvGloss = AdvGlossXz+AdvGlossZ#+RecXloss

            # Get the gradients w.r.t the generator loss
            gradFx_w, gradGz_w = tape.gradient(AdvGloss,
                                               (self.Fx.trainable_variables,
                                                self.Gz.trainable_variables),
                                               unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Update the weights of the generator using the generator optimizer
            self.FxOpt.apply_gradients(zip(gradFx_w,self.Fx.trainable_variables))
            self.GzOpt.apply_gradients(zip(gradGz_w,self.Gz.trainable_variables))
            
        self.loss_val["AdvGlossXz"] = AdvGlossXz
        self.loss_val["AdvGlossZ"] = AdvGlossZ
        # self.loss_val["RecXloss"] = RecXloss
            
        return
    
    @tf.function
    def train_step(self, XC):
        if isinstance(XC, tuple):
            X, metadata = XC
            damage_class, magnitude, damage_index = metadata

        self.batchSize = tf.shape(X)[0]

        for _ in range(self.nXRepX):
            XZXout = self.train_XZX(X, damage_class)
        
        # for _ in range(self.nRepXRep):  
        #     ZXZout = self.train_ZXZ(X, damage_class)
        

        # (Dx_fake,Dx_real) = ZXZout

        (Dxz_fake,Dz_fake,Dxz_real,Dz_real) = XZXout
        
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
        ε = self.px_flat.sample(self.batchSize)
        n = self.pz_flat.sample(self.batchSize)
        
        z_hat = self.Fx((X, ε), training=False)
        X_hat = self.Gz((z_hat, n), training=False)

        # Updates the metrics tracking the loss
        self.RecXloss(X, X_hat)
        # Update the metrics.
        self.RecGlossX_tracker.update_state(X, X_hat)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"RecXloss": RecGlossX_tracker.result()}


    def call(self, X):
        # Compute predictions
        ε = self.px_flat.sample(self.batchSize)
        n = self.pz_flat.sample(self.batchSize)
        z_hat = self.Fx((X, ε), training=False)
        X_hat = self.Gz((z_hat, n), training=False)
        return X_hat, z_hat

    def plot(self,X,c):
        ε = self.px_flat.sample(self.batchSize)
        n = self.pz_flat.sample(self.batchSize)
        z = self.pz.sample(self.batchSize)

        z_hat = self.Fx((X, ε), training=False)
        X_hat = self.Gz((z_hat, n), training=False)
        X_tilde = self.Gz((z, n), training=False)
        return X_hat, z_hat, X_tilde

    def label_predictor(self, X, c):
        ε = self.px_flat.sample(self.batchSize)
        n = self.pz_flat.sample(self.batchSize)
        z = self.pz.sample(self.batchSize)
        
        z_hat = self.Fx((X, ε), training=False)
        X_tilde = self.Gz((z,n), training=False)
        z_tilde = self.Fx((X_tilde, ε), training=False)
        return z_tilde, z_hat
    

    def reconstruct(self, X):
        ε = self.px_flat.sample(self.batchSize)
        n = self.pz_flat.sample(self.batchSize)
        return self.Gz((self.Fx(X, ε, training=False), n), training = False)
    
    def generate(self, c_fake_new):
        z = self.pz.sample(self.batchSize)
        n = self.pz_flat.sample(self.batchSize)
        return self.Gz((z, n), training=False)

    # BN : do not apply batchnorm to the generator output layer and the discriminator input layer
    def BuildFx(self):
        if self.AEtype=="CNN":
            return self.BuildFx_CNN()
        elif self.AEtype=="RES":
            return self.BuildFx_RES()
    
    def BuildFx_RES(self):
        
        X = kl.Input(shape = self.Xshape, 
                     name="X")
        ε = kl.Input(shape = self.Xshape, 
                     name="ε")
        # c = kl.Input(shape=self.latentCdim,
        #              name="c")
        # p = kl.Input(shape=self.nParams,
        #              name="p")
        
        # Initial CNN layer
        layer = 0
        h = il.FResBlock(out_channels=self.nZfirst,
                         dilation_rate=3, 
                         name='FRB0',
                         )(X)
        h = il.FiLM(self.nZfirst,
                    name="FiLM4e_2")((h, ε))

        # CNN layers
        layer = 1
        h = il.FResBlock(self.nZfirst*(2**(layer)),
                         name='FRB{:>d}'.format(layer),
                         dilation_rate=2)(h)
        h = il.FiLM(self.nZfirst*(2**(layer)),
                    name="FiLM4e_2")((h, ε))
        
        layer = 2
        h = il.FResBlock(self.nZfirst*(2**(layer)),
                         name='FRB{:>d}'.format(layer), 
                         dilation_rate=1)(h)
        h = il.FiLM(self.nZfirst*(2**(layer)),
                    name="FiLM4e_2")((h, ε))

        layer = 3
        h = kl.Flatten(name="FxFLN{:>d}".format(layer))(h)
        z = tfal.SpectralNormalization(kl.Dense(self.latentZdim),
                                       name="FxD{:>d}".format(layer))(h)
        # n = InstanceNormalization(axis=3)(h)

        Fx = tf.keras.Model(inputs=X, 
                            outputs=z, 
                            name="Fx")
        return Fx

    def BuildFx_CNN(self):
        """
            Fx encoder structure
        """
        # To build this model using the functional API

        # Input layer
        X = kl.Input(shape = self.Xshape, 
                     name="X")
        ε = kl.Input(shape = self.Xshape[0]*self.Xshape[1], 
                     name="ε")

        h_ε0 = LatentCodeConditioner(LCC=self.LCC,
                                     batch_size=self.batchSize,
                                     latent_length=self.Xshape[0],
                                     n_channels=self.nZfirst,
                                     SN=self.FxSN,
                                     name="FxLCC")(ε)
        m_ε = tf.keras.Model(ε, h_ε0)
        if self.FxSN:
            # Initial CNN layer
            h_X0 = tfal.SpectralNormalization(kl.Conv1D(self.nZfirst, 
                                                        self.kernel, 
                                                        1, 
                                                        padding="same",
                                                        data_format="channels_last", 
                                                        name="FxCNN0"))(X)
            m_X = tf.keras.Model(X, h_X0)
            h_X = kl.Add()([m_X.output, m_ε.output])
            
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
                # h_X = tfal.InstanceNormalization(axis=-1,
                #                                  epsilon=1e-6,
                #                                  center=True,
                #                                  scale=True,
                #                                  beta_initializer="random_uniform",
                #                                  gamma_initializer="random_uniform")
            h_X = kl.LeakyReLU(alpha=0.1, 
                               name="FxA{:>d}".format(l+1))(h_X)
            z = kl.Dropout(self.dpout, 
                           name="FxDO{:>d}".format(l+1))(h_X)
        else:        
            # Initial CNN layer        
            h_X0 = kl.Conv1D(self.nZfirst,
                             self.kernel,
                             1,
                             padding="same",
                             data_format="channels_last",
                             name="FxCNN0")(X)
            m_X = keras.Model(X, h_X0)
            h_X = kl.Add()([m_X.output, m_ε.output])
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

        F_x = tf.keras.Model(inputs=[m_X.input, m_ε.input], 
                             outputs=z,
                             name="Fx")
        return F_x

    def BuildGz(self):
        """
            Conv1D Gz structure CNN
        """
    
        z = kl.Input(shape=self.Zshape,
                     name="z")
        n = kl.Input(shape=self.Zshape[0]*self.Zshape[1],
                     name="n")
        h_n0 = LatentCodeConditioner(LCC=self.LCC,
                                     batch_size=self.batchSize,
                                     latent_length=self.Zshape[0],
                                     n_channels=self.nZchannels,
                                     SN=self.GzSN,
                                     name="GzLCC")(n)
        m_n = tf.keras.Model(n,h_n0)
        if self.GzSN:
            h_z0 = tfal.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels,
                                                                 self.kernel, 
                                                                 1, 
                                                                 padding="same", 
                                                                 data_format="channels_last"))(z)
            m_z = tf.keras.Model(z,h_z0)
            h_z = kl.Add()([m_z.output, m_n.output])
            h_z = kl.BatchNormalization(axis=-1, 
                                        momentum=0.95)(h_z)
            h_z = kl.LeakyReLU(alpha=0.1)(h_z)
        
            for l in range(1, self.nAElayers-1):
                h_z = tfal.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels//self.stride**(l+1),
                                                                    self.kernel, 
                                                                    self.stride, 
                                                                    padding="same", 
                                                                    use_bias=False))(h_z)
                h_z = kl.BatchNormalization(axis=-1, 
                                            momentum=0.95)(h_z)
                h_z = kl.LeakyReLU(alpha=0.1)(h_z)

            layer = self.nAElayers-1

            h_z = tfal.SpectralNormalization(kl.Conv1DTranspose(self.nZchannels//self.stride**(l+1),
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
            h_z0 = kl.Conv1DTranspose(self.nZchannels, 
                                      self.kernel, 
                                      1, 
                                      padding="same", 
                                      data_format="channels_last")(z)
            m_z = tf.keras.Model(z,h_z0)
            h_z = kl.Add()([m_z.output, m_n.output])
            h_z = kl.BatchNormalization(axis=-1, 
                                        momentum=0.95)(h_z)
            h_z = kl.LeakyReLU(alpha=0.1)(h_z)

            for l in range(1, self.nAElayers-1):
                h_z = kl.Conv1DTranspose(self.nZchannels//self.stride**(l+1),
                                         self.kernel, 
                                         self.stride, 
                                         padding="same", 
                                         use_bias=False)(h_z)
                h_z = kl.BatchNormalization(axis=-1, 
                                            momentum=0.95)(h_z)
                h_z = kl.LeakyReLU(alpha=0.1)(h_z)

            layer = self.nAElayers-1

            h_z = kl.Conv1DTranspose(self.nZchannels//self.stride**(l+1),
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
        G_z = tf.keras.Model(inputs=[m_z.input, m_n.input], 
                             outputs = X, 
                             name="Gz")
        return G_z

    def BuildDz(self):
        """
            Conv1D discriminator structure
        """
        l = 0
        z = kl.Input(shape=self.Zshape,name="z")
        
        if self.DzSN:
            h_z = tfal.SpectralNormalization(kl.Conv1D(self.Zsize*self.stride**(-(l+1)),
                                                       self.kernel,
                                                       self.stride,
                                                       padding="same",
                                                       data_format="channels_last",
                                                       name="DzCNN0"))(z)
            h_z = kl.LeakyReLU(alpha=0.1,
                               name="DzA0")(h_z)
            h_z = kl.Dropout(self.dpout)(h_z)

            for l in range(1,self.nDlayers):
                h_z = tfal.SpectralNormalization(kl.Conv1D(self.Zsize*self.stride**(-(l+1)),
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
            h_z = kl.Conv1D(self.Zsize*self.stride**(-(l+1)),
                            self.kernel,
                            self.stride,
                            padding="same",
                            data_format="channels_last",
                            name="DzCNN0")(z)
            h_z = kl.LeakyReLU(alpha=0.1,
                             name="DzA0")(h_z)
            h_z = kl.Dropout(self.dpout)(h_z)

            for l in range(1,self.nDlayers):
                h_z = kl.Conv1D(self.Zsize*self.stride**(-(l+1)),
                                self.kernel,
                                self.stride,
                                padding="same",
                                data_format="channels_last",
                                name="DzCNN{:>d}".format(l+1))(h_z)
                h_z = kl.LeakyReLU(alpha=0.1,
                                   name="DzA{:>d}".format(l+1))(h_z)
                h_z = kl.Dropout(self.dpout)(h_z)

            l = self.nDlayers    
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
            h_X = tfal.SpectralNormalization(kl.Conv1D(f, 
                                                       k, 
                                                       strides=1,
                                                       padding="same",
                                                       data_format="channels_last",
                                                       name="DxCNN0"))(X)
            h_X = kl.LeakyReLU(alpha=0.1, 
                               name="DxA0")(h_X)
            h_X = kl.Dropout(self.dpout)(h_X)

            # Branch Dz
            f = int(self.Zsize*1**(-(l+1)))
            h_z = tfal.SpectralNormalization(kl.Conv1D(f, 
                                                       k, 
                                                       strides=1,
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
