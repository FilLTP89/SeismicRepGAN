import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Layer, Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout 
from tensorflow.keras.layers import Lambda, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding1D
from tensorflow.keras.layers import LeakyReLU, ReLU, Softmax
from tensorflow.keras.layers import UpSampling1D, Conv1D, Conv1DTranspose
from tensorflow.keras.optimizers import Adam, RMSprop
from numpy.linalg import norm
import MDOFload as mdof
import matplotlib.pyplot as plt
import visualkeras

class RandomWeightedAverage(Layer):
    """Provides a (random) weighted average between real and generated signal samples"""
    def _merge_function(self, inputs, **kwargs):
        alpha = tf.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def WassersteinDiscriminatorLoss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
	return fake_loss - real_loss


# Define the loss functions for the generator.
def WassersteinGeneratorLoss(fake_img):
    return -tf.reduce_mean(fake_img)


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))


def GaussianNLL(true, pred):
    """
     Gaussian negative loglikelihood loss function 
    """
    n_dims = int(int(pred.shape[1])/2)
    mu = pred[:, 0:n_dims]
    logsigma = pred[:, n_dims:]
    
    mse = -0.5*K.sum(K.square((true-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)


class RepGAN(Model):

    def __init__(self,options):
		super(RepGAN, self).__init__()
    	"""
            Setup
          """
        self.__dict__.update(options)

        assert self.nZchannels >= 1
        assert self.nZchannels >= self.stride**self.nlayers
        assert self.latentZdim >= self.Xsize//(self.stride**self.nlayers)

        """
            Build the discriminators
        """
        self.Dx = self.build_Dx()
        self.Dc = self.build_Dc()
        self.Ds = self.build_Ds()
        self.Dn = self.build_Dn()
        """
            Build Fx/Gz (generators)
        """
        self.Fx = self.build_Fx()
        self.Gz = self.build_Gz()

    def compile(self,optimizers,losses):
        super(RepGAN, self).compile()
		"""
            Optimizers
        """
        self.__dict__.update(optimizers)
        
        """
        	Losses
        """
        self.__dict__.update(losses)

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, realX, realC):
        if isinstance(realX, tuple):
            realX = realX[0]

        # Get the batch size
        batchSize = tf.shape(realX)[0]
        if self.batchSize != batchSize:
        	self.batchSize = batchSize

        #------------------------------------------------
        #           Construct Computational Graph
        #               for the Discriminator
        #------------------------------------------------

        # Freeze generators' layers while training critics
        self.Fx.trainable = False
        self.Gz.trainable = False
        self.Dx.trainable = True
        self.Dc.trainable = True
        self.Ds.trainable = True
        self.Dn.trainable = True

        for _ in range(self.n_critic):
        	
        # The generator takes the signal, encodes it and reconstructs it
        # from the encoding
        # Real c,s,n

	    # realC = tf.zeros(shape=(self.batchSize,self.latentCdim))
     	# rand_idx = np.random.randint(0,self.latentCdim,self.batchSize)
     	# realC[np.arange(self.batchSize),rand_idx]=1.0

	     	with tf.GradientTape() as tape:

	        	realS = tf.random.normal(loc=0.0,scale=0.5,shape=(self.batchSize,self.latentSdim))
				realN = tf.random.normal(loc=0.0,scale=0.3,shape=(self.batchSize,self.latentNdim))

		        # # Generate fake latent code from real signals
		        (fakeC,fakeS,fakeN) = self.Fx(realX) # encoded z = Fx(X)

		        fakeX = self.Gz(realZ) # fake X = Gz(Fx(X))
		        

		        # Discriminator determines validity of the real and fake X
		        fakeXcritic = self.Dx(fakeX)
		        realXcritic = self.Dx(realX)

		        # Discriminator determines validity of the real and fake C
		        fakeCcritic = self.Dc(fakeC)
		        realCcritic = self.Dc(realC)

		        # Discriminator determines validity of the real and fake S
		        fakeScritic = self.Ds(fakeS)
		        realScritic = self.Ds(realS)

		        # Discriminator determines validity of the real and fake N
		        fakeNcritic = self.Dn(fakeN)
		        realNcritic = self.Dn(realN)


	        # Calculate the discriminator loss using the fake and real logits
			AdvDloss = self.AdvDloss(realXcritic,fakeXcritic)*self.PenAdvXloss +
	        	self.AdvDloss(realCcritic,fakeCcritic)*self.PenAdvCloss +
	        	self.AdvDloss(realScritic,fakeScritic)*self.PenAdvSloss +
	        	self.AdvDloss(realNcritic,fakeNcritic)*self.PenAdvNloss

	       	# Get the gradients w.r.t the discriminator loss
	       	gradDx = tape.gradient(AdvDloss, self.Dx.trainable_variables)
	       	gradDc = tape.gradient(AdvDloss, self.Dc.trainable_variables)
	       	gradDs = tape.gradient(AdvDloss, self.Ds.trainable_variables)
	       	gradDn = tape.gradient(AdvDloss, self.Dn.trainable_variables)

	       	# Update the weights of the discriminator using the discriminator optimizer
        	self.DxOpt.apply_gradients(zip(gradDx,self.Dx.trainable_variables))
        	self.DcOpt.apply_gradients(zip(gradDc,self.Dc.trainable_variables))
        	self.DsOpt.apply_gradients(zip(gradDs,self.Ds.trainable_variables))
        	self.DnOpt.apply_gradients(zip(gradDn,self.Dn.trainable_variables))

        	# Clip critic weights
            for l in self.Dx.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                l.set_weights(weights)
            for l in self.Dc.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                l.set_weights(weights)
            for l in self.Ds.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                l.set_weights(weights)
            for l in self.Dn.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                l.set_weights(weights)

        #----------------------------------------
        #      Construct Computational Graph
        #               for Generator
        #----------------------------------------

        # Freeze critics' layers while training generators
        self.Fx.trainable = True
        self.Gz.trainable = True
        self.Dx.trainable = False
        self.Dc.trainable = False
        self.Ds.trainable = False
        self.Dn.trainable = False

        with tf.GradientTape() as tape:
	        # Fake
	        (fakeC,fakeS,fakeN) = self.Fx(realX) # encoded z = Fx(X)
	        fakeX = self.Gz(realZ) # fake X = Gz(Fx(X))
	        
	        # Discriminator determines validity of the real and fake X
	        fakeXcritic = self.Dx(fakeX)

	        # Discriminator determines validity of the real and fake C
	        fakeCcritic = self.Dc(fakeC)

	        # Discriminator determines validity of the real and fake S
	        fakeScritic = self.Ds(fakeS)

	        # Discriminator determines validity of the real and fake N
	        fakeNcritic = self.Dn(fakeN)

	        # Reconstruction
	        fakeZ = Concatenate([fakeC,fakeS,fakeN])
	        recX  = self.Gz(fakeZ)
	        (recC,recS,_)  = self.Fx(fakeX)

	    AdvGLoss = self.AdvGloss(fakeXcritic)*self.PenAdvXloss + 
	    	self.AdvGloss(fakeCcritic)*self.PenAdvCloss +
	    	self.AdvGloss(fakeScritic)*self.PenAdvSloss +
	    	self.AdvGloss(fakeNcritic)*self.PenAdvNloss +
	    	self.RecXloss(recX)*self.PenRecXloss +
	    	self.RecCloss(recC)*self.PenRecCloss + 
	    	self.RecSloss(recS)*self.PenRecSloss

	    # Get the gradients w.r.t the generator loss
        gradFx = tape.gradient(AdvGLoss, self.Fx.trainable_variables)
        gradGz = tape.gradient(AdvGLoss, self.Gz.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.FxOpt.apply_gradients(zip(gradFx,self.Fx.trainable_variables))
        self.GzOpt.apply_gradients(zip(gradGz,self.Gz.trainable_variables))

        return {"AdvDLoss": AdvDLoss, "AdvGLoss": AdvGLoss}


    def build_Fx(self):
        """
            Conv1D Fx structure
        """
        
        X = Input(shape=self.Xshape)

        # for n in range(self.nlayers):
        #     X = Conv1D((self.latentZdim/self.Zsize)*self.stride**(-self.nlayers+n),
        #         self.kernel,self.stride,padding="same",
        #         input_shape=self.Xshape,data_format="channels_last")(X)
        #     X = BatchNormalization(momentum=0.95)(inputs=X)
        #     X = LeakyReLU(alpha=0.1)(X)
        #     X = Dropout(rate=0.2)(X)
        # z = Flatten()(X)
        # z = Dense(self.latentZdim)(z)
        # z = BatchNormalization(momentum=0.95)(inputs=z)
        # z = LeakyReLU(alpha=0.1)(z)


        model = Sequential()
        for n in range(self.nlayers):
            model.add(Conv1D((self.latentZdim/self.Zsize)*self.stride**(-self.nlayers+n),
                self.kernel,self.stride,padding="same",
                input_shape=self.Xshape,data_format="channels_last"))
            model.add(BatchNormalization(momentum=0.95))
            model.add(LeakyReLU(alpha=0.1))
            model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(self.latentZdim))
        model.add(BatchNormalization(momentum=0.95))
        model.add(LeakyReLU(alpha=0.1))
        model.summary()
        
        z = model(X)
        
               

        # sampleS = Sequential()
        # sampleS.add(Lambda(lambda t: t,(self.latentSidx[0]),(self.latentSidx[-1])))

        # sampleC = Sequential()
        # sampleC.add(Lambda(lambda t: t,(self.latentCidx[0]),(self.latentCidx[-1])))

        # sampleN = Sequential()
        # sampleN.add(Lambda(lambda t: t,(self.latentNidx[0]),(self.latentNidx[-1])))


        # variable s 
        # MuS = Sequential()
        # LVS = Sequential()
        # MuS.add(Dense(self.latentSdim))
        # MuS.add(BatchNormalization(momentum=0.95))
        # LVS.add(Dense(self.latentSdim))
        # LVS.add(BatchNormalization(momentum=0.95))

        # mu = MuS(z)
        # lv = LVS(z)

        h = Dense(self.latentSdim)(z)
        Zmu = BatchNormalization(momentum=0.95)(inputs=h)

        h = Dense(self.latentSdim)(z)
        Zlv = BatchNormalization(momentum=0.95)(inputs=h)

        s = Sampling()([Zmu,Zlv])

        # variable c
        sampleC = Sequential()
        sampleC.add(Dense(self.latentCdim))
        sampleC.add(BatchNormalization(momentum=0.95))
        sampleC.add(Softmax())
        c = sampleC(z)
        
        # variable n
        sampleN = Sequential()
        sampleN.add(Dense(self.latentNdim))
        sampleN.add(BatchNormalization(momentum=0.95))
        n = sampleN(z)

        # concatenation of variables c, s and n
        # z = Concatenate()([c,s,n])


        return keras.Model(X,(c,s,n),name="Fx")

    def build_Gz(self):
        """
            Conv1D Gz structure
        """
        model = Sequential()          
        model.add(Dense(self.Zsize*self.nZchannels,input_dim=self.latentZdim,
            use_bias=False))
        model.add(Reshape((self.Zsize,self.nZchannels)))
        model.add(BatchNormalization(momentum=0.95))
        model.add(ReLU())
        for n in range(self.nlayers):
            model.add(Conv1DTranspose(self.latentZdim//self.stride**n,
                self.kernel,self.stride,padding="same"))
            model.add(BatchNormalization(momentum=0.95))
            model.add(ReLU())
        
        model.add(Conv1DTranspose(self.nXchannels,self.kernel,1,padding="same"))
        model.summary()

        z = Input(shape=(self.latentZdim,))
        X = model(z)

        return keras.Model(z,X,name="Gz")
        
    def build_Dx(self):
        """
            Conv1D discriminator structure
        """
        model = Sequential()
        model.add(Conv1D(32,self.kernel,self.stride,
            input_shape=self.Xshape,padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(64,self.kernel,self.stride,padding="same"))
        model.add(ZeroPadding1D(padding=((0,1))))
        model.add(BatchNormalization(momentum=0.95))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(128,self.kernel,self.stride,padding="same"))
        model.add(BatchNormalization(momentum=0.95))
        model.add(LeakyReLU(alpha=0.0))
        model.add(Dropout(0.25))
        model.add(Conv1D(256,self.kernel,strides=1,padding="same"))
        model.add(BatchNormalization(momentum=0.95))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1,activation='sigmoid'))
        # model.add(Conv1D(64,self.kernel,self.stride,
        #     input_shape=self.Xshape,padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv1D(128,self.kernel,self.stride,
        #     input_shape=self.Xshape,padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.95))
        # model.add(Dense(1024,activation='LeakyReLU'))
        # model.add(BatchNormalization(momentum=0.95))
        # model.add(Dense(1,activation='sigmoid'))

        model.summary()

        X = Input(shape=(self.Xshape))
        Dx = model(X)

        return keras.Model(X,Dx,name="Dx")


    def build_Dc(self):
        """
            Dense discriminator structure
        """
        model = Sequential()
        model.add(Dense(3000)) 
        model.add(LeakyReLU())
        model.add(Dense(3000)) 
        model.add(LeakyReLU())
        model.add(Dense(1))  

        model.summary()

        c = Input(shape=(self.latentCdim,))
        Dc = model(c)

        return keras.Model(c,Dc,name="Dc")

    def build_Dn(self):
        """
            Dense discriminator structure
        """
        model = Sequential()
        model.add(Dense(3000)) 
        model.add(LeakyReLU())
        model.add(Dense(3000)) 
        model.add(LeakyReLU()) 
        model.add(Dense(1))  

        model.summary()

        n = Input(shape=(self.latentNdim,))
        Dn = model(n)

        return keras.Model(n,Dn,name="Dn")

    def build_Ds(self):
        """
            Dense discriminator structure
        """
        model = Sequential()
        model.add(Dense(3000)) 
        model.add(LeakyReLU())
        model.add(Dense(3000)) 
        model.add(LeakyReLU())
        model.add(Dense(1))  

        model.summary()

        s = Input(shape=(self.latentSdim,))
        Ds = model(s)

        return keras.Model(s,Ds,"Ds")


if __name__ == '__main__':

	options = {}
	options['stride'] = 2
    options['kernel'] = 3
    options['nlayers'] = 5
    options['Xsize'] = 1024
    options['nXchannels'] = 2
    options['Xshape'] = (Xsize, nXchannels)

    options['latentZdim'] = 2048
    options['Zsize'] = Xsize//(stride**nlayers)
    options['nZchannels'] = latentZdim//Zsize

    options['latentCidx'] = list(range(5))
    options['latentSidx'] = list(range(5,7))
    options['latentNidx'] = list(range(7,latentZdim))
    options['latentCdim'] = len(latentCidx)
    options['latentSdim']= len(latentSidx)
    options['latentNdim'] = len(latentNidx)
    options['batchSize'] = 128
    options['n_critic'] = 5
    options['clipValue'] = 0.01
    # options['data_root_ID'] = '/gpfs/workdir/invsem07/damaged_1_8P' 
    # options['ID_string'] = 'U'
    # options['ID_pb_string'] = 'BC'
    # options['case'] = 'train_model'


    optimizers = {}
    optmizers['DxOpt'] = RMSprop(lr=0.00005)
    optmizers['DcOpt'] = RMSprop(lr=0.00005)
    optmizers['DsOpt'] = RMSprop(lr=0.00005)
    optmizers['DnOpt'] = RMSprop(lr=0.00005)
    optmizers['FxOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    optmizers['GzOpt'] = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    losses = {}
	losses['AdvDloss'] = WassersteinDiscriminatorLoss
	losses['AdvGloss'] = WassersteinGeneratorLoss
	losses['RecSloss'] = GaussianNLL
	losses['RecXloss'] = keras.losses.MeanSquareError()
	losses['RecCloss'] = keras.losses.BinaryCrossentropy()
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
	GiorgiaGAN.compile(optimizers,losses)

	# Start training the model.
	# GiorgiaGAN.fit(train_images, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk],
	# 	epochs=4000, batchSize=32, save_interval=50)