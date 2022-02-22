# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:03:49 2021

@author: asdg
"""

#---for convolution layer-----
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU



#---------for LSTM-----------

import tensorflow as tf
from numpy import array
from tensorflow.keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
import keras

import xlrd
import numpy as np
import matplotlib.pyplot as plt
import math
from random import *
from math import log 
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
params = {'backend': 'ps',
          'axes.labelsize': 20,
          'legend.fontsize': 16,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True};
#--------------80% train data, 20% test data----------
T_pert=np.zeros((184,40));
loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/temperature_perturbations_hc_22_mar_2000.xls')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        T_pert[i][j]=sheet.cell_value(i,j);
        
        
        

