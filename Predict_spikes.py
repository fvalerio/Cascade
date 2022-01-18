

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script to predict spiking activity from calcium imaging data

The function "load_neurons_x_time()" loads the input data as a matrix. It can
be modified to load npy-files, mat-files or any other standard format.

The line "spike_prob = cascade.predict( model_name, traces )" performs the
predictions. As input, it uses the loaded calcium recordings ('traces') and
the pretrained model ('model_name'). The output is a matrix with the inferred spike rates.

"""



"""

Import python packages

"""

import os, sys
if 'Demo scripts' in os.getcwd():
    sys.path.append( os.path.abspath('..') ) # add parent directory to path for imports
    os.chdir('..')  # change to main directory
print('Current working directory: {}'.format( os.getcwd() ))

from cascade2p import checks
checks.check_packages()

import numpy as np
import scipy.io as sio
# import ruamel.yaml as yaml
# import matplotlib.pyplot as plt

from cascade2p import cascade # local folder
# from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth
# from scipy.io import savemat, loadmat

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

cases = ['rbp10'] # ['rbp6_3', 'rbp11', 'rbp12', 'rbp13', 'rbp16']
pt = 'Z:\\Scanbox\\Data\\'
pt = 'Z:\\Scanbox\\Data\\'
model_name = 'Global_EXC_15Hz_smoothing200ms'
# cascade.download_model( model_name,verbose = 1)
for icase in cases:
    path = pt + icase
    for filename in os.listdir(path):    
        if filename != 'matchedRoi'and filename != 'hide' :
            
            file_load = path + '\\' + filename + '\\suite2p\\plane0\\CASCADE\\Suite2p4Cascade.mat'
            traces = sio.loadmat(file_load)['df']
            spike_prob = cascade.predict( model_name, traces )
            file_save =  path + '\\' + filename + '\\suite2p\\plane0\\CASCADE\\Cascade.npy'
            np.save(file_save, spike_prob, allow_pickle=True)
            

