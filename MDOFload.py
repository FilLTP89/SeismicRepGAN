# MDOFload

# load data function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
from os.path import join as opj
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(**kwargs):
	load_data.__globals__.update(kwargs)
	"""
		Load time histories and parameters:
			data_root_ID
			case
			which_params
			n_param 
			ID_string
			ID_pb_string
			n_channels
			seq_len_input
			seq_len
			seq_sampling
	"""
	param_path = opj(data_root_ID,"parameters_{:>s}.csv".format(case))                                      
	load_param = np.genfromtxt(param_path)

	# Parse model's parameters
	for i0 in range(n_param):
		for i1 in range(load_param.shape[0]):
			if which_params[i0] == i1:
				to_be_load = load_param[i1,:]
				# Standardize the param to be loaded
				mean_p     = [np.mean(to_be_load)]
				std_p      = [np.std(to_be_load)]
				to_be_load = (to_be_load - mean_p) / std_p
				mean_p     = np.expand_dims(mean_p, axis=1)
				std_p      = np.expand_dims(std_p, axis=1)
				to_be_load = np.expand_dims(to_be_load, axis=1)
				if i0 == 0:
					params     = to_be_load
					param_mean = mean_p
					param_std  = std_p
				else:
					params     = np.concatenate((params, to_be_load), axis=1)
					param_mean = np.concatenate((param_mean, mean_p), axis=1)
					param_std  = np.concatenate((param_std,  std_p),  axis=1)
		
	# Parse time series 
	for i1 in range(nchannels):                                                    # load time series

		if len(ID_string) == 0:
			data_path  = opj(data_root_ID,"{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(ID_pb_string,case,which_channels[i1]))
		else:
			data_path  = opj(data_root_ID,"{:>s}_{:>s}_damaged_concat_{:>s}_gdl_{:>d}.csv".format(ID_pb_string,ID_string,case,which_channels[i1]))
		

		print("Sensor to be loaded: {:d}".format(i1))
		X_single_dof = np.genfromtxt(data_path)
		print("Loaded sensor: {:d}".format(i1))

		X_single_dof.astype(np.float32)
		if i1 == 0:
			n_instances = len(X_single_dof) // seq_len_input
			#n_instances = int(n_instances)
			print("n_instances: {:d}".format(n_instances))
			X = np.zeros((n_instances, seq_len, nchannels))

		i4 = 0
		for i3 in range(n_instances):                                                 # subsample the time series if required
			X_single_label = X_single_dof[i4 : (i4 + seq_len_input)]
			X_to_pooler = X_single_label[seq_len_start:(seq_len_start+seq_len*seq_sampling):seq_sampling]
			X[i3, 0 : (seq_len), i1] = X_to_pooler
			i4 = i4 + seq_len_input
	
	# Standardize the data
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	
	# Split between train and validation set (time series and parameters are splitted in the same way)
	X_trn, X_vld, params_trn, params_vld = train_test_split(X, params[:X.shape[0],:], random_state=5)

	# Sampling of categorical variables c
	c = np.zeros((batch_size, latent_c_dim))
	rand_idx = np.random.randint(0, latent_c_dim, batch_size)
	c[np.arange(batch_size), rand_idx]=1.0

	# Split between train and validation set of categorical variables c
	c_trn, c_vld = train_test_split(c, random_state=5)

	# Sampling of continuous variables s
	s = np.random.uniform(low=-1.0, high=1.0000001, size=(batch_size, latent_s_dim))

	# Split between train and validation set of continuous variables s
	s_trn, s_vld = train_test_split(s, random_state=5)

	#Sampling of noise n
	noise = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, latent_n_dim))

	# Split between train and validation set of noise
	n_trn, n_vld = train_test_split(noise, random_state=5)
	
	return X_trn,  X_vld, params_trn, params_vld, c_trn, c_vld, s_trn, s_vld, n_trn, n_vld