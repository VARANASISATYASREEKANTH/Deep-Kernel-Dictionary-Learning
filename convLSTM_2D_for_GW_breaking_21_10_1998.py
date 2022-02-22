# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:57:59 2021

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

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
#import segmentation_models as sm



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

loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/training_data/wave_breaking_train_data_ver2.xlsx')
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




#--------------test data perturbations---------------------

loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/test_data_21_10_1998_ver2.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
test_data_pert=np.zeros((sheet.nrows,sheet.ncols));
test_data_dash_pert=np.zeros((sheet.nrows,sheet.ncols));
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        test_data_dash_pert[i][j]=sheet.cell_value(i,j)

for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        test_data_pert[i][j]=test_data_dash_pert[i][j]-np.mean(test_data_dash_pert[i][1:1663]);




######################################################################
#---------------------training the model(CNN+LSTM)--------------------
n_in=1;
#-----reshaping train and test data----
data =T_pert.reshape(1,26,1, 128, 64)#training
data_test=test_data_pert.reshape(1, 26,1, 128, 64)# reshaping test data


#---------------------ConvLSTM model----------------------
seq_model = Sequential()

seq_model.add(ConvLSTM2D(filters=64, kernel_size=(4, 4),
                   input_shape=( 26, 1, 128, 64),
                   padding='same', return_sequences=True))
seq_model.add(BatchNormalization())

seq_model.add(ConvLSTM2D(filters=64, kernel_size=(4, 4),
                   padding='same', return_sequences=True))
seq_model.add(BatchNormalization())

seq_model.add(ConvLSTM2D(filters=64, kernel_size=(4, 4),
                   padding='same', return_sequences=True))
seq_model.add(BatchNormalization())

seq_model.add(ConvLSTM2D(filters=64, kernel_size=(4, 4),
                   padding='same', return_sequences=True))
seq_model.add(BatchNormalization())

#seq_model.add(Conv3D(filters=1, kernel_size=(4, 4, 4),
               #activation='sigmoid',
               #padding='same', data_format='channels_last'))
seq_model.compile(optimizer='adam', loss='mse',metrics=['accuracy','mse', 'mae', 'mape'])
seq_model.summary();    
keras.utils.plot_model(seq_model, "ConvLSTM_model_rayleigh_lidar_1.pdf", show_shapes=True);    
y=np.zeros((128,26));
print(y);
y=y.reshape(1,26,1, 128, 1);
model_hist=seq_model.fit(data, y,epochs=50);
model_loss=seq_model.history;
model_metrics=seq_model.metrics_names;

print(model_metrics);
yhat_ConvLSTM= seq_model.predict(data_test, verbose=1);
yhat_ConvLSTM=yhat_ConvLSTM.reshape(128,1664);

#yhat_ConvLSTM=yhat_ConvLSTM/np.max(yhat_ConvLSTM);
#-----------model training with train data-------------------  
#--------------contour plot------------------
fig = plt.figure(figsize=(10,8))
plt.figure(1)
ax=plt.subplot(1,1,1)
start, stop, n_values = 1, 1664, 1664
start1, stop1, n_values1 = 42, 80, 128
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y=np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y, (yhat_ConvLSTM),cmap='jet')
plt.colorbar(cp)
#plt.clim(0, 1)
ax.set_title('FFT' )
ax.set_xlabel('Time(minutes) \n (a)')
ax.set_ylabel('Height(km)')
ax.set_xticks(np.arange(0, 360, 90))
ax.set_yticks(np.arange(30, 90, 10))
plt.xlim(0,1664);
plt.ylim(40,80);