## -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
__author__ = "Filippo Gatti, Gerardo Granados"
__copyright__ = "Copyright 2022, CentraleSupélec (LMPS UMR CNRS 9026)"
__credits__ = ["Filippo Gatti, Gerardo Granados"]
__license__ = "GPL"
__version__ = "1.0.1"
__Maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow.keras.metrics as km
import tensorflow.keras.constraints as kc
import tensorflow.keras.activations as ka
import tensorflow_addons.layers as tfal



class Attention(tf.Module):
    def __init__(self, 
                 in_ch, 
                 name='attention'):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = int(in_ch)
        self.theta = tfal.SpectralNormalization(kl.Conv1D(self.ch // 8,
                                                          kernel_size=1,
                                                          padding='valid',
                                                          use_bias=False),
                                                name=name+'_conv_theta')
        self.phi = tfal.SpectralNormalization(kl.Conv1D(self.ch // 8,
                                                        kernel_size=1,
                                                        padding='valid',
                                                        use_bias=False),
                                              name=name+'_conv_phi')
        self.max_phi = kl.MaxPool1D((2,2),name=name+'_max_phi')
        self.g = tfal.SpectralNormalization(kl.Conv1D(self.ch//2,
                                                      kernel_size=1,
                                                      padding='valid',
                                                      use_bias=False),
                                                      name=name+'_conv_g')
        self.max_g = kl.MaxPool1D((2,2),name=name+'_max_g')
        self.o = tfal.SpectralNormalization(kl.Conv1D(self.ch,
                                                      kernel_size=1,
                                                      padding='valid',
                                                      use_bias=False),
                                                      name=name+'_conv_o')
        # Learnable gain parameter
        self.gamma = tf.Variable(0., 
                                 trainable=True)
        self.softmax = kl.Softmax(axis=-1,
                                  name = name +'_softmax_map')

    #Forward
    def __call__(self, x):
        # print('I\'m using self-attention!')
        #Use attention only on 64x64 resolution.
        # Apply convs to (B X H X W X Ch)
        theta = self.theta(x)
        phi = self.max_phi(self.phi(x))
        g = self.max_g(self.g(x))
        # Perform reshapes (B X H X W X Ch) --> (B X Ch X H X W) -->  (B X Ch/8 X H*W) ,
        # X.shape is still shape (B X H X W X Ch)!
        theta = tf.reshape(tf.transpose(theta, [0,3,1,2]),shape=(-1, self.ch // 8, x.shape[1] * x.shape[2]))
        phi = tf.reshape(tf.transpose(phi, [0,3,1,2]),shape=(-1, self.ch // 8, x.shape[1] * x.shape[2]//4))
        g = tf.reshape(tf.transpose(g, [0,3,1,2]),shape=(-1, self.ch // 2, x.shape[1] * x.shape[2]//4))
        # Matmul par batch and softmax to get attention map (B X Ch/8 X H*W) x (B X H*W X Ch/8)
        # beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        beta = self.softmax(tf.matmul(tf.transpose(theta,[0, 2, 1]), phi))
        # Attention map times g path (o --> channel at last for conv o and adapt to X shape)
        # o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        o = self.o(tf.reshape(tf.matmul(g, tf.transpose(beta,[0, 2, 1])),shape=(-1, x.shape[1], x.shape[2], self.ch // 2)))
        return self.gamma * o + x

# Class-conditional Instance normalization
# output size is the number of channels, input size is for the linear layers
# https://github.com/ajbrock/BigGAN-PyTorch
class ccbn(tf.Module):
    def __init__(self,
                 n_bias_scale, 
                 eps=1e-5,
                 momentum=0.99, 
                 name = 'CCN'):
        super(ccbn, self).__init__()
        self.n_bias_scale = n_bias_scale
        # Prepare gain and bias layers
        self.gain = tfal.SpectralNormalization(kl.Dense(self.n_bias_scale),
                                               name = name + 'gain')
        self.bias = tfal.SpectralNormalization(kl.Dense(self.n_bias_scale),
                                               name = name + 'bais')
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        self.norm = tfal.InstanceNormalization(axis=3,
                                               epsilon=self.eps,
                                               center = False,scale=False,
                                               name = name +'norm')
    #Forward
    def __call__(self, x, y):
        # Calculate class-conditional gains and biases
        #TO-DO check output size (HxWxCH on tf) (-1 axis is channel size for this layer)
        gain = tf.reshape(tf.convert_to_tensor(1,tf.float32) + self.gain(y), shape=(-1, 1, 1, self.n_bias_scale))
        bias = tf.reshape(self.bias(y), shape = (-1, 1, 1,self.n_bias_scale))
        out = self.norm(x)
        return out * gain + bias

#ResNet kl.Layer Generator
class GResBlock(tf.Module):
    def __init__(self, 
                 out_ch,
                 activation=ka.relu, 
                 name='GResnet',
                 dilation_rate=1, 
                 with_noise=False):
        super(GResBlock, self).__init__()
        self.out_channels = out_ch
        self.activation = activation
        self.dilation_rate = dilation_rate
        # Reach 2**n resolution on filters when dilation is set
        self.padding = self.dilation_rate
        self.with_noise = with_noise
        # class conditional IN layers (on ft for activation)
        self.ccin1 = ccbn(self.out_channels*2,
                          name = name +'_CIN1') # Input for previous layer: ch_in*2
        self.ccin2 = ccbn(self.out_channels,
                          name = name +'_CIN2')   # 2nd conv on res layer: ch_in
        # class conditional ST layers
        self.st1 = pSpatialTransformerGenerator(name = name +'_ST1')
        self.st2 = pSpatialTransformerGenerator(name = name +'_ST2')
        # self.stx = pSpatialTransformer(name = name +'_STX')
        # Zero Padding layer (keep 2**n resolution)
        self.pad1 = kl.ZeroPadding1D(padding = self.padding,
                                     name = name +'_PAD1')
        self.pad2 = kl.ZeroPadding1D(padding = self.padding, 
                                     name = name +'_PAD2')
        # Conv layers
        self.conv1 = tfal.SpectralNormalization(kl.Conv1D(self.out_channels,
                                                          3,
                                                          dilation_rate=self.dilation_rate,
                                                          use_bias = False),
                                                name = name +'_Conv1')
        self.conv2 = tfal.SpectralNormalization(kl.Conv1D(self.out_channels,
                                                          3,
                                                          dilation_rate=self.dilation_rate,
                                                          use_bias = False),
                                                name = name +'_Conv2')
        self.convx = tfal.SpectralNormalization(kl.Conv1D(self.out_channels,
                                                          1,
                                                          padding='valid',
                                                          use_bias = False),
                                                name = name +'_Convx')
        #Random noise kl.Layers (weigthed)
        if with_noise:
            self.nb1 = WeightedNoise(name = name +'_NoiseB1')
            self.nb2 = WeightedNoise(name = name +'_NoiseB2')
        # self.nbx = WeightedNoise(name = name +'_NoiseBX')
        # Down sampling 2D
        self.upsamplex = kl.UpSampling1D(size=2,
                                         name = name +'_UPSX')
        self.upsample1 = kl.UpSampling1D(size=2,
                                         name = name +'_UPS1')
    #Forward
    def __call__(self, x, y):
        # branch upsampling+class+zkip
        h = self.activation(self.st1(self.ccin1(x, y),y))
        # h = self.activation(self.ccin1(x, y))
        h = self.upsample1(h)
        h = self.conv1(self.pad1(h))
        if self.with_noise:
            h = self.nb1(h)
        h = self.activation(self.st2(self.ccin2(h, y),y))
        # h = self.activation(self.ccin2(h, y))
        h = self.conv2(self.pad2(h))
        if self.with_noise:
            h = self.nb2(h)
        # branch x
        x = self.upsamplex(x)
        x = self.convx(x)
  
        # adding branches
        return h + x

class FiLM(kl.Layer):
    """Features-wise layer normalization.

    It follows:
        
        f(x) = gain * norm(x) + bais 
        
    Input shape
    
        [NxWxHxC]
        
    Output shape
    
        Same shape as the input.
        
    Arguments
    
        n_bias_scale: number of channel (pairs of parameters to learn by FCNN)

    References
       
        Implementation on Sequential Model
    
        conv = FiLM()(conv)
    
    """
    def __init__(self, 
                 n_bias_scale, 
                 eps=1e-5, 
                 name = 'CCBN',**kwargs):
        super(FiLM, self).__init__(**kwargs)
        self.supports_masking = True
        self.n_bias_scale = n_bias_scale
        # Prepare gain and bias layers
        self.gain = kl.Dense(n_bias_scale,
                             name = name + 'Dgain')
        self.bias = kl.Dense(n_bias_scale,
                             name = name + 'Ggain')
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.norm = kl.BatchNormalization(epsilon=self.eps,
                                          center = False,
                                          scale=False,
                                          name = name +'BN')
            
    # def build(self, input_shape):
    #     self.shape = input_shape[-1] 
    #     self.built = True
    #     super(FiLM, self).build(input_shape)
    
    #Forward
    @tf.function
    def call(self, hε):
        # Calculate class-conditional gains and biases
        #check output size (HxWxCH on tf) (-1 axis is channel size for this layer)
        ftX = hε[0]
        ε = hε[1]
        one = tf.convert_to_tensor(1.0, tf.float32)
        # import pdb
        # pdb.set_trace()
        gain = tf.reshape(one + self.gain(ε), 
                          shape=(-1, 1, self.n_bias_scale))
        bias = tf.reshape(self.bias(ε), 
                          shape = (-1, 1, self.n_bias_scale))
        out = self.norm(ftX)
        return out * gain + bias
    
    # def get_config(self):
    #     base_config = super(FiLM, self).get_config()
    #     return dict(list(base_config.items()))
    
    # def compute_output_shape(self, input_shape):
    #     return input_shape
    
#ResNet kl.Layer Filter (Encoder) -same than D but BatchNorm and dilatation rate
class FResBlock(tf.Module):
    def __init__(self, 
                 out_channels,
                 activation=ka.relu,
                 dilation_rate=1,
                 name='FResBlock',
                 **kwargs):
        super(FResBlock, self).__init__(name, **kwargs)

        # Reach 2**n resolution on filters when dilation is set
        padding = dilation_rate
        # BN layers (on ft for activation)
        self.in1 = tfal.InstanceNormalization(name = name +'_IN1',
                                              axis=-1)
        self.in2 = tfal.InstanceNormalization(name = name +'_IN2',
                                              axis=-1)
        # # class conditional ST layers
        # self.st1 = pSpatialTransformerGenerator(name = name +'_ST1')
        # self.st2 = pSpatialTransformerGenerator(name = name +'_ST2')
        # Zero Padding layer (keep 2**n resolution)
        self.pad1 = kl.ZeroPadding1D(padding = padding, 
                                     name = name +'_PAD1')
        self.pad2 = kl.ZeroPadding1D(padding = padding, 
                                     name = name +'_PAD2')
        
        self.act1 = activation
        self.act2 = activation
        # Conv layers
        self.conv1 = tfal.SpectralNormalization(kl.Conv1D(filters=out_channels,
                                                          kernel_size=3,
                                                          dilation_rate=dilation_rate),
                                                name = name +'_Conv1')
        self.conv2 = tfal.SpectralNormalization(kl.Conv1D(filters=out_channels,
                                                          kernel_size=3,
                                                          dilation_rate=dilation_rate),
                                                name = name +'_Conv2')
        self.convx = tfal.SpectralNormalization(kl.Conv1D(filters=out_channels,
                                                          kernel_size=1,
                                                          padding='valid'),
                                                name = name +'_Convx')
        # Average pooling layer (down sampling)
        self.poolconv = kl.AveragePooling1D(pool_size=2,
                                            name = name +'_AvgPool_conv')
        self.poolx = kl.AveragePooling1D(pool_size=2,
                                         name = name +'_AvgPool_X')

    #Forward
    # @tf.function
    def __call__(self, X):
        # branch upsampling+class+zkip
        h = self.act1(self.in1(X))
        h = self.conv1(self.pad1(h))
        h = self.act2(self.in2(h))
        h = self.conv2(self.pad2(h))
        # h = self.poolconv(h)
        # branch x
        fx = self.convx(X)
        # fx = self.poolx(X)
        # adding branches
        return h + fx

#ResNet kl.Layer Discriminator (same as Encoder but no ST, need to make the diff btw class)
class DResBlock(tf.Module):
    def __init__(self,
                 out_ch,
                 activation=ka.relu,
                 name='FResBlock',
                 dilation_rate=1):
        super(DResBlock, self).__init__()
        self.out_channels = out_ch
        self.activation = activation
        self.dilation_rate = dilation_rate
        # Reach 2**n resolution on filters when dilation is set
        self.padding = self.dilation_rate
        # BN layers (on ft for activation)
        self.bn1 = tfal.InstanceNormalization(name = name +'_IN1',
                                              axis=3)
        self.bn2 = tfal.InstanceNormalization(name = name +'_IN2',
                                              axis=3)
        # # class conditional ST layers
        # self.st1 = pSpatialTransformer(name = name +'_ST1')
        # self.st2 = pSpatialTransformer(name = name +'_ST2')

        # Zero Padding layer (keep 2**n resolution)
        self.pad1 = kl.ZeroPadding1D(padding = self.padding)
        self.pad2 = kl.ZeroPadding1D(padding = self.padding)
        # Conv layers
        self.conv1 = tfal.SpectralNormalization(kl.Conv1D(self.out_channels,
                                                          3,
                                                          dilation_rate=self.dilation_rate),
                                                name = name +'_Conv1')
        self.conv2 = tfal.SpectralNormalization(kl.Conv1D(self.out_channels,
                                                          3,
                                                          dilation_rate=self.dilation_rate),
                                                name = name +'_Conv2')
        self.convx = tfal.SpectralNormalization(kl.Conv1D(self.out_channels,
                                                          1,
                                                          padding='valid'),
                                                name = name +'_Convx')
        # Average pooling layer (down sampling)
        self.pool = kl.AveragePooling1D(pool_size=2,
                                        name = name +'_AvgPool')

    #Forward
    def __call__(self,x):
        # branch upsampling+class+zkip
        h = self.activation(self.bn1(x))
        h = self.conv1(self.pad1(h))
        h = self.activation(self.bn2(h))
        h = self.conv2(self.pad2(h))
        h = self.pool(h)    
        # branch x
        x = self.convx(x)
        x = self.pool(x)
        # adding branches
        return h + x

class ResBlockDense(tf.Module):
    def __init__(self, 
                 units, 
                 dp = 0.0,
                 name = 'ResBlockkl.Dense',
                 act='relu'):
        super().__init__()
        with tf.name_scope('ResBlockkl.Dense'):
            self.dense1 = tfal.SpectralNormalization(kl.Dense(units=units),
                                                     name = name + '1')
            self.dense2 = tfal.SpectralNormalization(kl.Dense(units=units),
                                                     name = name + '2')
            self.densex = tfal.SpectralNormalization(kl.Dense(units=units),
                                                     name = name + 'x')
            # self.bn1 = kl.BatchNormalization(name = name +'_BN1')
            # self.bn2 = kl.BatchNormalization(name = name +'_BN2')
            self.dp1 = kl.Dropout(dp)
            self.dp2 = kl.Dropout(dp)
            if act == 'relu':
                self.activation1 = kl.ReLU(name=name+"act1")
                self.activation2 = kl.ReLU(name=name+"act2")
                self.activationx = kl.ReLU(name=name+"actx")
            elif act == 'leaky':
                self.activation1 = kl.LeakyReLU(name=name+"act1")
                self.activation2 = kl.LeakyReLU(name=name+"act2")
                self.activationx = kl.LeakyReLU(name=name+"actx")

    #@tf.function
    # It follows https://github.com/LEGO999/BigBiGAN-TensorFlow2.0/
    # p-relu instead of leakyrelu
    def __call__(self, x, training = False):
        h = self.dense1(x)
        h = self.dp1(h, training = training)
        h = self.activation1(h)
        h = self.dense2(h)
        h = self.dp2(h, training = training)
        h = self.activation2(h)
        x = self.densex(x)
        x = self.activationx(x)
        return h + x

class DClassProjection(tf.Module):
    def __init__(self,
               ch,
               activation=ka.relu):
        super(DClassProjection, self).__init__()
        self.activation = activation
        self.embedding = kl.Embedding(input_dim=2,
                                      output_dim=ch*512,
                                      input_length=None,
                                      name="D_kl.Embeddings")
    #self.dense_proyetor = tfal.SpectralNormalization(kl.Dense(512),name= 'DProyectorkl.Dense')

    def __call__(self, inputs):
        Px, h, c = inputs
        h = tf.math.reduce_sum(self.activation(h), [1, 2])
        return Px + tf.math.reduce_sum(self.embedding(tf.math.argmax(c,axis=-1))*h,
                                       axis=-1, 
                                       keepdims=True)

class WeightedNoise(kl.Layer):
    """Add noise on for filter resolution.
    It follows:
        f(x) = x + noise x weight
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        B: initializer function for applies learned per-channel scaling factors to the noise input

    # References
        - https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py

    # Implementation on Sequencial Model

    conv = WeightedNoise()(conv)

    """

    def __init__(self, 
                 B_initializer='zeros',
                 noise_initializer = 'random_normal',
                 **kwargs):
        super(WeightedNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.B_initializer = tf.keras.initializers.get(B_initializer)
        # fix noise for this layer
        self.noise_initializer = tf.keras.initializers.get(noise_initializer)


    def build(self, input_shape):
        param_shape = input_shape[-1]

        #one B gain per ch
        self.B = self.add_weight(shape=(param_shape),
                                      name='B',
                                      initializer=self.B_initializer,
                                   trainable=True)

        #one featere shape noise mask for all ch
        self.noise = self.add_weight(shape=(input_shape[1],input_shape[2],1),
                                      name='noise',
                                      initializer=self.noise_initializer,
                                       trainable=False)

        self.built = True
        super(WeightedNoise, self).build(input_shape)

    @tf.function
    def call(self, x):
        self.noise = tf.random.normal(tf.shape(self.noise))
        return x + self.noise * tf.reshape(tf.cast(self.B, x.dtype), [1,  1, 1, -1])


    def get_config(self):
        config = {
            'B': self.B_initializer,
            'noise': self.noise_initializer,
        }
        base_config = super(WeightedNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape