# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("Num GPUs:", len(physical_devices))
##tf.config.experimental.disable_mlir_graph_optimization()
#print("Num GPUs Available: ", len(tf.test.gpu_device_name()))
#from tensorflow.python.eager.context import get_config
# tf.compat.v1.disable_eager_execution()
#tf.debugging.set_log_device_placement(True)

#
#print(c)
#
#gpus = tf.config.list_physical_devices('GPU')
#if gpus:
#  # Restrict TensorFlow to only use the first GPU
#  try:
#    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#  except RuntimeError as e:
#    # Visible devices must be set before GPUs have been initialized
#    print(e)
# @tf_export('config.experimental.disable_mlir_bridge')
# def disable_mlir_bridge():
#   ##Disables experimental MLIR-Based TensorFlow Compiler Bridge.
#   context.context().enable_mlir_bridge = False

# tp.plot_model(self.Fx, filename='Fx_color.png', display_params=True, display_shapes=True)
# tp.plot_model(self.Qs, filename='Qs_color.png', display_params=True, display_shapes=True)
# tp.plot_model(self.Qc, filename='Qc_color.png', display_params=True, display_shapes=True)
# # tp.plot_model(self.Gz, filename='Gz_color.png', display_params=True, display_shapes=True)
# tp.plot_model(self.Dn, filename='Dn_color.png', display_params=True, display_shapes=True)
# tp.plot_model(self.Ds, filename='Ds_color.png', display_params=True, display_shapes=True)
# tp.plot_model(self.Dc, filename='Dc_color.png', display_params=True, display_shapes=True)
# tp.plot_model(self.Dx, filename='Dx_color.png', display_params=True, display_shapes=True)

# Fx_onnx = keras2onnx.convert_keras(self.Fx, "Fx")
# keras2onnx.save_model(Fx_onnx, 'Fx.onnx')
# visualkeras.layered_view(self.Fx, to_file='Fx.png', legend=True, font=font)
   # model.summary()


   #      dot_img_file = 'Dx.png'
   #      tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)
#h = BatchNormalization(momentum=hp.Float('BN_1',min_value=0.0,max_value=1,
        #    default=0.95,step=0.05),name="FxBN0")(h)
        #h = Dropout(rate=hp.Float('dropout_1',min_value=0.0,max_value=0.4,
        #    default=0.2,step=0.05),name="FxDO0")(h)