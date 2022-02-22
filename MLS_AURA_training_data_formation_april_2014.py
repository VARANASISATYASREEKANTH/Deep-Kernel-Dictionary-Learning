# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:24:13 2021

@author: asdg
"""
#import pandas as pd
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
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn import preprocessing
from keras.utils.vis_utils import plot_model

plt.rcParams.update({'font.size':22})
params = {'backend': 'ps',
          'axes.labelsize': 22,
          'legend.fontsize': 22,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22,
          'text.usetex': True};
#----------------definition of convlstm---------------------
#--------------96% training data, 4% test data----------
#--------------training data perturbations---------------------

#--------Rayleigh Lidar training data
'''loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/training_data/wave_breaking_train_data_ver2.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
T_pert=np.zeros((sheet.nrows,sheet.ncols));
T_dash_pert=np.zeros((sheet.nrows,sheet.ncols));
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        T_dash_pert[i][j]=sheet.cell_value(i,j)

for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        T_pert[i][j]=T_dash_pert[i][j]-np.mean(T_dash_pert[i][:]);
'''
#----Microwave limb sounder AURA data (13N to 14N, 79E to 80E)---
#-----0m to 110km, with resolution=300m---
data_preprocessed=np.zeros((3459,58));
loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/Aura_MLS/09_01_2015/mls_10.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
data1=np.zeros((sheet.nrows,sheet.ncols));
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        data1[i][j]=sheet.cell_value(i,j)
a=0,
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        if (data1[i][0]>10 and data1[i][0]<20) and (data1[i][1]>50 and data1[i][1]<80):
            data_preprocessed[i][j]=data1[i][j];
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        