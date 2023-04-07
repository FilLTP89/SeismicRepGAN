## -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
__author__ = "Giorgia Colombera, Filippo Gatti"
__copyright__ = "Copyright 2021, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Giorgia Colombera,Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__Maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

import os
import argparse
import tensorflow as tf
import h5py

def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CreateData",
                        action='store_true',
                        default=False,
                        help='Create data flag')
    parser.add_argument("--rawdata_dir",
                        nargs="+",
                        default=["PortiqueElasPlas_N_2000_index", "PortiqueElasPlas_E_2000_index"], help="Data root folder")
    parser.add_argument("--store_dir",
                        default="input_data",
                        help="Data root folder")
    parser.add_argument("--checkpoint_dir",
                        default='checkpoint',
                        help="Checkpoint")
    parser.add_argument("--checkpoint_step",
                        type=int,
                        default=500,
                        help="Checkpoint epochs")
    parser.add_argument("--results_dir",
                        default='results',
                        help="Checkpoint")
    parser.add_argument("--idChannels",
                        type=int,
                        nargs='+',
                        default=[1, 2, 3, 4],
                        help="Channel 1")
    parser.add_argument("--nParams",
                        type=int,
                        default=2,
                        help="Number of parameters")
    parser.add_argument("--case",
                        type=str,
                        default="train_model",
                        help="case")
    parser.add_argument("--avu",
                        type=str,
                        nargs='+',
                        default="U",
                        help="case avu")
    parser.add_argument("--pb",
                        type=str,
                        default="DC",
                        help="case pb")
    parser.add_argument('--dtm',
                        type=float,
                        default=0.04,
                        help='time-step [s]')
    parser.add_argument("--cuda", 
                        action='store_true', 
                        default=False, 
                        help='Use cuda powered GPU')
    parser.add_argument("--epochs",
                        type=int,
                        default=2000,
                        help='Number of epochs')
    parser.add_argument("--batchSize",
                        type=int,
                        default=50,
                        help='input batch size')
    parser.add_argument("--Xsize",
                        type=int,
                        default=2048,
                        help='Data space size')
    parser.add_argument("--nX",
                        type=int,
                        default=4000,
                        help='Number of signals')
    parser.add_argument("--nXchannels",
                        type=int,
                        default=4,
                        help="Number of data channels")
    parser.add_argument("--AEtype",
                        type=str,
                        default="RES",
                        help="Type of AE (CNN or RES)")
    parser.add_argument("--normalization",
                        type=str,
                        default="IN",
                        help="Instance Normalization (IN), Adaptive IN (AdaIN) or BatchNormalizaion (BN)")
    parser.add_argument("--nAElayers",
                        type=int,
                        default=3,
                        help='Number of AE CNN layers')
    parser.add_argument("--nDlayersX",
                        type=int, 
                        default=3,
                        help='Number of D CNN layers for X')
    parser.add_argument("--nDlayersz", 
                        type=int, 
                        default=3,
                        help='Number of D CNN layers for z')
    parser.add_argument("--nDlayers",
                        type=int,
                        default=5,
                        help='Number of D CNN layers')
    parser.add_argument("--N",
                        type=int,
                        default=2,
                        help="Number of experiments")
    parser.add_argument("--signal",
                        type=int,
                        default=2,
                        help="Types of signals")
    parser.add_argument("--kernel",
                        type=int,
                        default=3,
                        help='CNN kernel size')
    parser.add_argument("--stride",
                        type=int,
                        default=2,
                        help='CNN stride')
    parser.add_argument("--nZfirst",
                        type=int,
                        default=8,
                        help="Initial number of channels")
    # parser.add_argument("--branching",
    #                     type=str,
    #                     default='conv',
    #                     help='conv or dens')
    # parser.add_argument("--latentSdim",
    #                     type=int,
    #                     default=2,
    #                     help="Latent space s dimension")
    parser.add_argument("--latentCdim",
                        type=int,
                        default=2,
                        help="Number of classes")
    # parser.add_argument("--latentNdim",
    #                     type=int,
    #                     default=512,
    #                     help="Latent space n dimension")
    parser.add_argument("--latentZdim",
                        type=int,
                        default=512,
                        help="Latent space z dimension")
    # parser.add_argument("--nSlayers",
    #                     type=int,
    #                     default=1,
    #                     help='Number of S-branch CNN layers')
    # parser.add_argument("--nClayers",
    #                     type=int,
    #                     default=1,
    #                     help='Number of C-branch CNN layers')
    # parser.add_argument("--nNlayers",
    #                     type=int,
    #                     default=1,
    #                     help='Number of N-branch CNN layers')
    # parser.add_argument("--Skernel",
    #                     type=int,
    #                     default=3,
    #                     help='CNN kernel of S-branch branch')
    # parser.add_argument("--Ckernel",
    #                     type=int,
    #                     default=3,
    #                     help='CNN kernel of C-branch branch')
    # parser.add_argument("--Nkernel",
    #                     type=int,
    #                     default=3,
    #                     help='CNN kernel of N-branch branch')
    # parser.add_argument("--Sstride",
    #                     type=int,
    #                     default=2,
    #                     help='CNN stride of S-branch branch')
    # parser.add_argument("--Cstride",
    #                     type=int,
    #                     default=2,
    #                     help='CNN stride of C-branch branch')
    # parser.add_argument("--Nstride",
    #                     type=int,
    #                     default=2,
    #                     help='CNN stride of N-branch branch')
    
    parser.add_argument("--DxTrainType", 
                        type=str, 
                        default='WGAN',
                        help='Train Dx with GAN, WGAN or WGANGP')
    # parser.add_argument("--DcTrainType", 
    #                     type=str, 
    #                     default='WGAN',
    #                     help='Train Dc with GAN, WGAN or WGANGP')
    # parser.add_argument("--DsTrainType", 
    #                     type=str, 
    #                     default='WGAN',
    #                     help='Train Ds with GAN, WGAN or WGANGP')
    # parser.add_argument("--DnTrainType", 
    #                     type=str, 
    #                     default='WGAN',
    #                     help='Train Dn with GAN, WGAN or WGANGP')
    parser.add_argument("--DxzTrainType",
                        type=str,
                        default='WGAN',
                        help='Train Dxz with GAN, WGAN or WGANGP')
    parser.add_argument("--DzTrainType",
                        type=str,
                        default='WGAN',
                        help='Train Dz with GAN, WGAN or WGANGP')
    # parser.add_argument("--DxSN", 
    #                     action='store_true',
    #                     default=False, 
    #                     help='Spectral normalization in Dx')
    parser.add_argument("--DxzSN",
                        action='store_true',
                        default=False,
                        help='Spectral normalization in Dc, Ds, Dn')
    parser.add_argument("--DzSN", 
                        action='store_true',
                        default=False, 
                        help='Spectral normalization in Dc, Ds, Dn')
    parser.add_argument("--FxSN", 
                        action='store_true',
                        default=False, 
                        help='Spectral normalization in Fx')
    parser.add_argument("--GzSN", 
                        action='store_true',
                        default=False, 
                        help='Spectral normalization in Gz')
    parser.add_argument("--DxLR", 
                        type=float, 
                        default=0.0001,
                        help='Learning rate for Dx [GAN=0.0002/WGAN=0.001]')
    # parser.add_argument("--DcLR", 
    #                     type=float, 
    #                     default=0.0001,
    #                     help='Learning rate for Dc [GAN=0.0002/WGAN=0.001]')    
    # parser.add_argument("--DsLR", 
    #                     type=float, 
    #                     default=0.0001,
    #                     help='Learning rate for Ds [GAN=0.0002/WGAN=0.001]')
    # parser.add_argument("--DnLR", 
    #                     type=float, 
    #                     default=0.0001,
    #                     help='Learning rate for Dn [GAN=0.0002/WGAN=0.001]')
    parser.add_argument("--DzLR",
                        type=float,
                        default=0.0001,
                        help='Learning rate for Dz [GAN=0.0002/WGAN=0.001]')
    parser.add_argument("--FxLR", 
                        type=float, 
                        default=0.0001,
                        help='Learning rate for Fx [GAN=0.0002/WGAN=0.0001]')
    parser.add_argument("--GzLR", 
                        type=float, 
                        default=0.0001,
                        help='Learning rate for Gz [GAN=0.0002/WGAN=0.0001]')
    parser.add_argument("--PenAdvXloss",
                        type=float,
                        default=1.0,
                        help="Penalty coefficient for Adv X loss")
    # parser.add_argument("--PenAdvCloss",
    #                     type=float,
    #                     default=1.0,
    #                     help="Penalty coefficient for Adv C loss")
    # parser.add_argument("--PenAdvSloss",
    #                     type=float,
    #                     default=1.0,
    #                     help="Penalty coefficient for Adv S loss")
    # parser.add_argument("--PenAdvNloss",
    #                     type=float,
    #                     default=1.0,
    #                     help="Penalty coefficient for Adv N loss")
    parser.add_argument("--PenAdvZloss",
                        type=float,
                        default=1.0,
                        help="Penalty coefficient for Adv z loss")
    parser.add_argument("--PenRecXloss",
                        type=float,
                        default=1.0,
                        help="Penalty coefficient for Rec X loss")
    # parser.add_argument("--PenRecCloss",
    #                     type=float,
    #                     default=1.0,
    #                     help="Penalty coefficient for Rec C loss")
    # parser.add_argument("--PenRecSloss",
    #                     type=float,
    #                     default=1.0,
    #                     help="Penalty coefficient for Rec S loss")
    parser.add_argument("--PenGPXloss",
                        type=float,
                        default=10.0,
                        help="Penalty coefficient for WGAN GP X loss")
    # parser.add_argument("--PenGPCloss",
    #                     type=float,
    #                     default=10.0,
    #                     help="Penalty coefficient for WGAN GP C loss")
    # parser.add_argument("--PenGPSloss",
    #                     type=float,
    #                     default=10.0,
    #                     help="Penalty coefficient for WGAN GP S loss")
    # parser.add_argument("--PenGPNloss",
    #                     type=float,
    #                     default=10.0,
    #                     help="Penalty coefficient for WGAN GP N loss")
    parser.add_argument("--PenGPZloss",
                        type=float,
                        default=10.0,
                        help="Penalty coefficient for WGAN GP z loss")
    # parser.add_argument("--DclassLR",
    #                     type=float, 
    #                     default=0.0002,
    #                     help='Learning rate for Qc [GAN=0.0002/WGAN=0.00002]')
    parser.add_argument("--nCritic",
                        type=int,
                        default=1,
                        help='number of discriminator training steps')
    parser.add_argument("--nGenerator", 
                        type=int, 
                        default=1,
                        help='number of generator training steps')
    parser.add_argument("--nXRepX",
                        type=int,
                        default=1,
                        help='number of discriminator training steps')
    parser.add_argument("--nRepXRep",
                        type=int,
                        default=5,
                        help='number of discriminator training steps')
    parser.add_argument("--LCC", 
                        type=str, 
                        default="LDC",
                        help='Latent Code Conditioning: [LCD] Latent-Dependent Conditioning; [LIC] Location-Invariant Conditioning')
    parser.add_argument("--clipValue",
                        type=float,
                        default=0.01,
                        help='clip weight for WGAN')
    parser.add_argument("--dpout",
                        type=float,
                        default=0.25,
                        help='Dropout ratio')
    # parser.add_argument("--sigmas2",
    #                     default='linear',
    #                     help="Last sigmas2 activation layer")
    # parser.add_argument("--sdouble_branch",
    #                     action='store_true',
    #                     default=False,
    #                     help="Split the s branch into two, one for average and one for logvar")
    # parser.add_argument("--skip",
    #                     action='store_true',
    #                     default=False,
    #                     help="Add skip connections from F to G")
    parser.add_argument("--trainVeval",
                        type=str,
                        default="TRAIN",
                        help="Train or Eval")
    options = parser.parse_args().__dict__

    options['batchXshape'] = (options['batchSize'],options['Xsize'],options['nXchannels'])
    options['Xshape'] = (options['Xsize'],options['nXchannels'])

    # options['latentZdim'] = options['latentSdim']+options['latentNdim']+options['latentCdim']

    options['Zsize'] = options['Xsize']//(options['stride']**options['nAElayers'])
    options['nZchannels'] = options['nZfirst']*(options['stride']**options['nAElayers']) # options['latentZdim']//options['Zsize']
    options['Zshape'] = (options['Zsize'],options['nZchannels'])

    options["AEtype"] = options["AEtype"].upper()
    # options['nSchannels'] = options['nZchannels']*options['Sstride']**options['nSlayers']
    # options['nCchannels'] = options['nZchannels']*options['Cstride']**options['nClayers']
    # options['nNchannels'] = options['nZchannels']*options['Nstride']**options['nNlayers']
    # options['Ssize'] = int(options['Zsize']*options['Sstride']**(-options['nSlayers']))
    # options['Csize'] = int(options['Zsize']*options['Cstride']**(-options['nClayers']))
    # options['Nsize'] = int(options['Zsize']*options['Nstride']**(-options['nNlayers']))
    # options['Nshape'] = (options['Nsize'], options['nNchannels'])
    # #options['nDlayers'] = min(options['nDlayers'],int(tf.math.log(options['Xsize'],options['stride'])))

    if not os.path.exists(options['checkpoint_dir']):
        os.makedirs(options['checkpoint_dir'])

    options["DeviceName"] = tf.test.gpu_device_name() if not options['cuda'] else "/cpu:0"
    options["LCC"] = options["LCC"].upper()
    
    return options
        
def DumpModels(model, results_dir):
    
    # filepath = os.path.join(results_dir, "{:>s}".format(model.name))
    # tf.keras.models.save_model(model=model,filepath=filepath,save_format="tf")
    # loaded_module = tf.keras.models.load_model(
    #     filepath, custom_objects=None, compile=True, options=None)
    for m in model.models:
        filepath= os.path.join(results_dir, "{:>s}.h5".format(m.name))
        fidh5 = h5py.File(filepath,'w')
        tf.keras.models.save_model(model=m,filepath=fidh5,save_format="h5")
        fidh5.close()
        
        tf.keras.utils.plot_model(m,to_file="{:>s}.png".format(os.path.join(results_dir,m.name)))
    
    return