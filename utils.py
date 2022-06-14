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

def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=2000,help='Number of epochs')
    parser.add_argument("--Xsize",type=int,default=2048,help='Data space size')
    parser.add_argument("--nX",type=int,default=4000,help='Number of signals')
    parser.add_argument("--nXchannels",type=int,default=4,help="Number of data channels")
    parser.add_argument("--nAElayers",type=int,default=3,help='Number of AE CNN layers')
    parser.add_argument("--nDlayers",type=int,default=5,help='Number of D CNN layers')
    parser.add_argument("--N",type=int,default=2,help="Number of experiments")
    parser.add_argument("--signal",type=int,default=2,help="Types of signals")
    parser.add_argument("--kernel",type=int,default=3,help='CNN kernel size')
    parser.add_argument("--stride",type=int,default=2,help='CNN stride')
    parser.add_argument("--nZfirst",type=int,default=8,help="Initial number of channels")
    parser.add_argument("--branching",type=str,default='conv',help='conv or dens')
    parser.add_argument("--latentSdim",type=int,default=2,help="Latent space s dimension")
    parser.add_argument("--latentCdim",type=int,default=2,help="Number of classes")
    parser.add_argument("--latentNdim",type=int,default=512,help="Latent space n dimension")
    parser.add_argument("--nSlayers",type=int,default=3,help='Number of S-branch CNN layers')
    parser.add_argument("--nClayers",type=int,default=3,help='Number of C-branch CNN layers')
    parser.add_argument("--nNlayers",type=int,default=3,help='Number of N-branch CNN layers')
    parser.add_argument("--Skernel",type=int,default=3,help='CNN kernel of S-branch branch')
    parser.add_argument("--Ckernel",type=int,default=3,help='CNN kernel of C-branch branch')
    parser.add_argument("--Nkernel",type=int,default=3,help='CNN kernel of N-branch branch')
    parser.add_argument("--Sstride",type=int,default=2,help='CNN stride of S-branch branch')
    parser.add_argument("--Cstride",type=int,default=2,help='CNN stride of C-branch branch')
    parser.add_argument("--Nstride",type=int,default=2,help='CNN stride of N-branch branch')
    parser.add_argument("--batchSize",type=int,default=50,help='input batch size')    
    parser.add_argument("--nCritic",type=int,default=1,help='number of discriminator training steps')
    parser.add_argument("--nXRepX",type=int,default=1,help='number of discriminator training steps')
    parser.add_argument("--nRepXRep",type=int,default=5,help='number of discriminator training steps')
    parser.add_argument("--nGenerator",type=int,default=1,help='number of generator training steps')
    parser.add_argument("--clipValue",type=float,default=0.01,help='clip weight for WGAN')
    parser.add_argument("--dataroot", nargs="+", default=["/gpfs/workdir/colombergi/GiorgiaGAN/PortiqueElasPlas_N_2000_index",
                        "/gpfs/workdir/colombergi/GiorgiaGAN/PortiqueElasPlas_E_2000_index"],help="Data root folder") 
    parser.add_argument("--dataroot_index", nargs="+", default=["/gpfs/workdir/colombergi/GiorgiaGAN/PortiqueElasPlas_N_2000_index",
                        "/gpfs/workdir/colombergi/GiorgiaGAN/PortiqueElasPlas_E_2000_index"],help="Data root folder") 
    parser.add_argument("--idChannels",type=int,nargs='+',default=[1,2,3,4],help="Channel 1")
    parser.add_argument("--nParams",type=str,default=2,help="Number of parameters")
    parser.add_argument("--case",type=str,default="train_model",help="case")
    parser.add_argument("--avu",type=str,nargs='+',default="U",help="case avu")
    parser.add_argument("--pb",type=str,default="DC",help="case pb")#DC
    parser.add_argument("--CreateData",action='store_true',default=False,help='Create data flag')
    parser.add_argument("--cuda",action='store_true',default=False,help='Use cuda powered GPU')
    parser.add_argument('--dtm',type=float,default=0.04,help='time-step [s]')
    parser.add_argument("--checkpoint_dir",default='/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/03_06',help="Checkpoint")
    parser.add_argument("--results_dir",default='/gpfs/workdir/colombergi/GiorgiaGAN/results',help="Checkpoint")
    parser.add_argument("--discriminator",default='WGAN',help="Type of Dz")
    parser.add_argument("--σs2",default='sigmoid',help="Last σs2 activation layer")
    options = parser.parse_args().__dict__

    options['batchXshape'] = (options['batchSize'],options['Xsize'],options['nXchannels'])
    options['Xshape'] = (options['Xsize'],options['nXchannels'])

    options['latentZdim'] = options['latentSdim']+options['latentNdim']+options['latentCdim']

    options['Zsize'] = options['Xsize']//(options['stride']**options['nAElayers'])
    options['nZchannels'] = options['nZfirst']*(options['stride']**options['nAElayers']) # options['latentZdim']//options['Zsize']
    options['nZshape'] = (options['Zsize'],options['nZchannels'])

    options['nSchannels'] = options['nZchannels']*options['Sstride']**options['nSlayers']
    options['nCchannels'] = options['nZchannels']*options['Cstride']**options['nClayers']
    options['nNchannels'] = options['nZchannels']*options['Nstride']**options['nNlayers']
    options['Ssize'] = int(options['Zsize']*options['Sstride']**(-options['nSlayers']))
    options['Csize'] = int(options['Zsize']*options['Cstride']**(-options['nClayers']))
    options['Nsize'] = int(options['Zsize']*options['Nstride']**(-options['nNlayers']))

    #options['nDlayers'] = min(options['nDlayers'],int(tf.math.log(options['Xsize'],options['stride'])))

    if not os.path.exists(options['checkpoint_dir']):
        os.makedirs(options['checkpoint_dir'])

    return options