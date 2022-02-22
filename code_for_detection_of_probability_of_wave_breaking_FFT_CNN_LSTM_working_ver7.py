# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 09:49:42 2021

@author: asdg
"""
#---for convolution layer-----
import keras
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
plt.rcParams.update({'font.size':28})
params = {'backend': 'ps',
          'axes.labelsize': 26,
          'legend.fontsize': 26,
          'xtick.labelsize': 26,
          'ytick.labelsize': 26,
          'text.usetex': True};


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

loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/test_data_03_01_2014_ver2.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
test_data_pert=np.zeros((sheet.nrows,sheet.ncols));
test_data_dash_pert=np.zeros((sheet.nrows,sheet.ncols));
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        test_data_dash_pert[i][j]=sheet.cell_value(i,j)

for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        test_data_pert[i][j]=test_data_dash_pert[i][j]-np.mean(test_data_dash_pert[i][:]);





#-----------------training the model(CNN+LSTM)-----------------


#---------CNN----------
#-----reshaping data----
data =T_pert.reshape(1, 128, 64, 26)#training
data_test=test_data_pert.reshape(1, 128, 64, 1)# reshaping test data



model=Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(128,64,26),padding='same'))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.011))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))



#model.add(Flatten())
#model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.summary()
keras.utils.plot_model(model, "my_first_model.png");
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True) 

#---------------assigning random weights to the model--------
initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
weights=initializer(shape=(128, 64));

#----------------conforming the model weights----------------
model_weights=model.get_weights()
#-----------------------see the dimensions of the tensor-------
model.compile(optimizer='adam', loss='mse',metrics=['accuracy','mse', 'mae', 'mape'])
model_hist=model.fit(out_predicted, out_predicted, epochs=10, verbose=0)
model_loss=model_hist.history;
model_metrics=model.metrics_names;
print(model_metrics);

#-------------------output of CNN----------------------------
out_predicted = model.predict(data_test);

for i in range(0,1,15):
    y_out_0=np.transpose(out_predicted[0][:][:][0]);
    y_out_1=np.transpose(out_predicted[0][:][:][1]);
    y_out_2=np.transpose(out_predicted[0][:][:][2]);
    y_out_3=np.transpose(out_predicted[0][:][:][3]);
    y_out_4=np.transpose(out_predicted[0][:][:][4]);
    y_out_5=np.transpose(out_predicted[0][:][:][5]);
    y_out_6=np.transpose(out_predicted[0][:][:][6]);
    y_out_7=np.transpose(out_predicted[0][:][:][7]);
    y_out_8=np.transpose(out_predicted[0][:][:][8]);
    y_out_9=np.transpose(out_predicted[0][:][:][9]);
    y_out_10=np.transpose(out_predicted[0][:][:][10]);
    y_out_11=np.transpose(out_predicted[0][:][:][11]);
    y_out_12=np.transpose(out_predicted[0][:][:][12]);
    y_out_13=np.transpose(out_predicted[0][:][:][13]);
    y_out_14=np.transpose(out_predicted[0][:][:][14]);
    y_out_15=np.transpose(out_predicted[0][:][:][15]);

#y_out=np.concatenate((y_out_0,y_out_1,y_out_2,y_out_3,y_out_4,y_out_5,y_out_6, y_out_7,y_out_8,y_out_9,y_out_10,y_out_11,y_out_12,y_out_13,y_out_14,y_out_15),axis=1);
y_out1=np.concatenate((y_out_0,y_out_1,y_out_2,y_out_3,y_out_4,y_out_5,y_out_6, y_out_7),axis=1)
y_normalized=preprocessing.normalize(y_out1);
#--------------calculation of model_metrics-----------------

y_out_reshaped=y_out1.reshape(1, 128, 64, 1)#reshaping output
tst_data_reshape=data_test.reshape(1,8,8,128)
print(np.shape(y_out_reshaped))
print(np.shape(data_test));

#--------------plotting the model---------------
'''keras.utils.plot_model(model, "my_first_model.png");
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)''' 

























