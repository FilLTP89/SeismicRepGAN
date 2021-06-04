
        # The batch size is always omitted since only the shape of each sample is specified. 
        #Alternative: https://github.com/tensorflow/tensorflow/issues/37942 batch_input_shape=self.batchXshape,

        # for n in range(self.nCnnLayers):
        #     X = Conv1D((self.latentZdim/self.Zsize)*self.stride**(-self.nCnnLayers+n),
        #         self.kernel,self.stride,padding="same",
        #         input_shape=self.Xshape,data_format="channels_last")(X)
        #     X = BatchNormalization(momentum=0.95)(inputs=X)
        #     X = LeakyReLU(alpha=0.1)(X)
        #     X = Dropout(rate=0.2)(X)
        # z = Flatten()(X)
        # z = Dense(self.latentZdim)(z)
        # z = BatchNormalization(momentum=0.95)(inputs=z)
        # z = LeakyReLU(alpha=0.1)(z)

        # nXchannels = deepcopy(self.nXchannels)
        # input_shape=self.Xshape,
        # batch_input_shape=self.batchXshape,

        



        
        # z = model(X)

             
               

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