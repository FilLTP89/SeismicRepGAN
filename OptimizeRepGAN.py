# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from sklearn import metrics
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix
# from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error
# from sklearn.model_selection import GridSearchCV

        # def create_model():

        #     """
        #         Conv1D Fx structure
        #     """
        #     # To build this model using the functional API

        #     # Input layer
        #     X = Input(shape=self.Xshape,name="X")

        #     # Initial CNN layer
        #     layer = -1
        #     # h = Conv1D(self.nZfirst*self.nSchannels*self.stride**(-self.nAElayers+layer+1),
        #     h = Conv1D(self.nZfirst[i], #*self.nSchannels*self.stride**(-self.nAElayers+layer+1),
        #             self.kernel[k],1,padding="same", # self.kernel,self.stride,padding="same",
        #             data_format="channels_last",name="FxCNN0")(X)
        #     h = BatchNormalization(momentum=0.95,name="FxBN0")(h)
        #     h = LeakyReLU(alpha=0.1,name="FxA0")(h)
        #     h = Dropout(0.2,name="FxDO0")(h)

        #     # Common encoder CNN layers
        #     for layer in range(self.nAElayers[j]):
        #         # h = Conv1D(self.nZfirst*self.nSchannels*self.stride**(-self.nAElayers+layer+1),
        #         h = Conv1D(self.nZfirst[i]*self.stride**(layer+1),
        #             self.kernel[k],self.stride,padding="same",
        #             data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
        #         h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
        #         h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
        #         h = Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)

        #     # Last common CNN layer (no stride, same channels) before branching
        #     layer = self.nAElayers[j]
        #     # h = Conv1D(self.nZfirst*self.nSchannels*self.stride**(-self.nAElayers+layer+1),
        #     h = Conv1D(self.nZchannels,
        #         self.kernel,1,padding="same",
        #         data_format="channels_last",name="FxCNN{:>d}".format(layer+1))(h)
        #     h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
        #     h = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)
        #     z = Dropout(0.2,name="FxDO{:>d}".format(layer+1))(h)
        #     # z ---> Zshape = (Zsize,nZchannels)

        #     layer = 0
        #     if 'dense' in self.branching:
        #         # Flatten and branch
        #         h = Flatten(name="FxFL{:>d}".format(layer+1))(z)
        #         h = Dense(self.latentZdim,name="FxFW{:>d}".format(layer+1))(h)
        #         h = BatchNormalization(momentum=0.95,name="FxBN{:>d}".format(layer+1))(h)
        #         zf = LeakyReLU(alpha=0.1,name="FxA{:>d}".format(layer+1))(h)

        #         # variable s
        #         # s-average
        #         h = Dense(self.latentSdim,name="FxFWmuS")(zf)
        #         Zmu = BatchNormalization(momentum=0.95)(h)

        #         # s-log std
        #         h = Dense(self.latentSdim,name="FxFWlvS")(zf)
        #         Zlv = BatchNormalization(momentum=0.95)(h)

        #         # variable c
        #         h = Dense(self.latentCdim,name="FxFWC")(zf)
        #         Zc = BatchNormalization(momentum=0.95,name="FxBNC")(h)

        #         # variable n
        #         Zn = Dense(self.latentNdim,name="FxFWN")(zf)

        #     elif 'conv' in self.branching:
        #         # variable s
        #         # s-average
        #         Zmu = Conv1D(self.nZchannels*self.Sstride**(layer+1),
        #             self.Skernel,self.Sstride,padding="same",
        #             data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(z)
        #         Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(Zmu)
        #         Zmu = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(Zmu)
        #         Zmu = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(Zmu)

        #         # s-log std
        #         Zlv = Conv1D(self.nZchannels*self.Sstride**(layer+1),
        #             self.Skernel,self.Sstride,padding="same",
        #             data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(z)
        #         Zlv = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zlv)
        #         Zlv = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zlv)
        #         Zlv = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zlv)

        #         # variable c
        #         Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
        #                 self.Ckernel,self.Cstride,padding="same",
        #                 data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(z)
        #         Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
        #         Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
        #         Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)

        #         # variable n
        #         Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
        #                 self.Nkernel,self.Nstride,padding="same",
        #                 data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(z)
        #         Zn = BatchNormalization(momentum=0.95,name="FxBNN{:>d}".format(layer+1))(Zn)
        #         Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
        #         Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)

        #         # variable s
        #         for layer in range(1,self.nSlayers):
        #             # s-average
        #             Zmu = Conv1D(self.nZchannels*self.Sstride**(layer+1),
        #                 self.Skernel,self.Sstride,padding="same",
        #                 data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(Zmu)
        #             Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(Zmu)
        #             Zmu = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(Zmu)
        #             Zmu = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(Zmu)

        #             # s-log std
        #             Zlv = Conv1D(self.nZchannels*self.Sstride**(layer+1),
        #                 self.Skernel,self.Sstride,padding="same",
        #                 data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(Zlv)
        #             Zlv = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zlv)
        #             Zlv = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zlv)
        #             Zlv = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zlv)

        #         # variable c
        #         for layer in range(1,self.nClayers):
        #             Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
        #                 self.Ckernel,self.Cstride,padding="same",
        #                 data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(Zc)
        #             Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
        #             Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
        #             Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)

        #         # variable n
        #         for layer in range(1,self.nNlayers):
        #             Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
        #                 self.Nkernel,self.Nstride,padding="same",
        #                 data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(Zn)
        #             Zn = BatchNormalization(momentum=0.95,name="FxBNN{:>d}".format(layer+1))(Zn)
        #             Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
        #             Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)

        #         # variable s
        #         # layer = self.nSlayers
        #         # Zmu = Conv1D(self.nZchannels*self.Sstride**(layer+1),
        #         #     self.Skernel,self.Sstride,padding="same",
        #         #     data_format="channels_last",name="FxCNNmuS{:>d}".format(layer+1))(Zmu)
        #         # Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS{:>d}".format(layer+1))(Zmu)
        #         # Zmu = LeakyReLU(alpha=0.1,name="FxAmuS{:>d}".format(layer+1))(Zmu)
        #         # Zmu = Dropout(0.2,name="FxDOmuS{:>d}".format(layer+1))(Zmu)
        #         Zmu = Flatten(name="FxFLmuS{:>d}".format(layer+1))(Zmu)
        #         Zmu = Dense(self.latentSdim,name="FxFWmuS")(Zmu)
        #         Zmu = BatchNormalization(momentum=0.95,name="FxBNmuS")(Zmu)

        #         # s-log std
        #         # Zlv = Conv1D(self.nZchannels*self.Sstride**(layer+1),
        #         #         self.Skernel,self.Sstride,padding="same",
        #         #     data_format="channels_last",name="FxCNNlvS{:>d}".format(layer+1))(Zlv)
        #         # Zlv = BatchNormalization(momentum=0.95,name="FxBNlvS{:>d}".format(layer+1))(Zlv)
        #         # Zlv = LeakyReLU(alpha=0.1,name="FxAlvS{:>d}".format(layer+1))(Zlv)
        #         # Zlv = Dropout(0.2,name="FxDOlvS{:>d}".format(layer+1))(Zlv)
        #         Zlv = Flatten(name="FxFLlvS{:>d}".format(layer+1))(Zlv)
        #         Zsigma = Dense(self.latentSdim,activation=tf.keras.activations.sigmoid,
        #             name="FxFWlvS")(Zlv)
        #         # Zlv = Dense(self.latentSdim,name="FxFWlvS")(Zlv)
        #         # Zlv = BatchNormalization(momentum=0.95,name="FxBNlvS")(Zlv)
        #         # CLIP Zlv
        #         Zsigma_clip = tf.clip_by_value(Zsigma,-1.0,1.0)

        #         # variable c
        #         layer = self.nClayers
        #         # Zc = Conv1D(self.nZchannels*self.Cstride**(layer+1),
        #         #         self.Ckernel,self.Cstride,padding="same",
        #         #         data_format="channels_last",name="FxCNNC{:>d}".format(layer+1))(Zc)
        #         # Zc = BatchNormalization(momentum=0.95,name="FxBNC{:>d}".format(layer+1))(Zc)
        #         # Zc = LeakyReLU(alpha=0.1,name="FxAC{:>d}".format(layer+1))(Zc)
        #         # Zc = Dropout(0.2,name="FxDOC{:>d}".format(layer+1))(Zc)
        #         Zc = Flatten(name="FxFLC{:>d}".format(layer+1))(Zc)
        #         Zc = Dense(self.latentCdim,name="FxFWC")(Zc)
        #         Zc = BatchNormalization(momentum=0.95,name="FxBNC")(Zc)

        #         # variable n
        #         layer = self.nNlayers
        #         # Zn = Conv1D(self.nZchannels*self.Nstride**(layer+1),
        #         #     self.Nkernel,self.Nstride,padding="same",
        #         #     data_format="channels_last",name="FxCNNN{:>d}".format(layer+1))(Zn)
        #         # Zn = BatchNormalization(momentum=0.95,name="FxBNN{:>d}".format(layer+1))(Zn)
        #         # Zn = LeakyReLU(alpha=0.1,name="FxAN{:>d}".format(layer+1))(Zn)
        #         # Zn = Dropout(0.2,name="FxDON{:>d}".format(layer+1))(Zn)
        #         Zn = Flatten(name="FxFLN{:>d}".format(layer+1))(Zn)
        #         Zn = Dense(self.latentNdim,name="FxFWN")(Zn)

        #     # variable s
        #     s = SamplingFxS()([Zmu,Zsigma_clip])
        #     QsX = Concatenate(axis=-1)([Zmu,Zsigma])

        #     # variable c
        #     c   = Softmax(name="FxAC")(Zc)
        #     QcX = Softmax(name="QcAC")(Zc)

        #     # variable n
        #     n = BatchNormalization(momentum=0.95,name="FxBNN")(Zn)
            

        #     Fx = keras.Model(X,[s,c,n],name="Fx")
        #     Qs = keras.Model(X,QsX,name="Qs")
        #     # Qs = keras.Model(X,[Zmu,Zsigma],name="Qs")
        #     Qc = keras.Model(X,QcX,name="Qc")

        #     Fx.compile(optimizers,losses)

        #     return Fx

        #     # GiorgiaGAN = RepGAN(options)
        #     # GiorgiaGAN.compile(optimizers,losses)

        #     #return GiorgiaGAN

        
        
        # model = KerasClassifier(build_fn = create_model, batch_size=128, epochs=10)
        # #now write out all the parameters you want to try out for the grid search
        # nZfirst = [8, 16, 32]
        # nAElayers = [5, 10, 15]
        # kernel = [1, 2, 3]
        # #activation = ['relu', 'tanh', 'sigmoid']
        # #learn_rate = [0.00005, 0.0002, 0.00001]
        # #optimizer = ['Adam', 'RMSprop']
        # #param_grid = dict(activation=activation, learn_rate=learn_rate, optimizer=optimizer)
        # param_grid = dict(nZfirst=nZfirst, kernel=kernel)
        # #param_grid= {'kernel': ('linear', 'rbf'),'C': [1, 10, 100]}
        # grid = GridSearchCV(estimator=model, param_grid=param_grid)
        # result = grid.fit(fakeX,realX)
        # best_params = result.best_params_
        # best_params.to_csv('Best parameters.csv', index= True)
        # print("[INFO] grid search best parameters: {}".format(best_params))


        # tuner = kt.Hyperband(build_model, objective="AdvDlossX", max_epochs=30, hyperband_iterations=2)

        # tuner.search_space_summary()

        # tuner.search(Xtrn,epochs=options["epochs"],validation_data=Xvld,callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],)

        # bestModel = tuner.get_best_models(1)[0]

        # bestHyperparameters = tuner.get_best_hyperparameters(1)[0]

