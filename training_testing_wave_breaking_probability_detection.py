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
plt.rcParams.update({'font.size':21})
params = {'backend': 'ps',
          'axes.labelsize': 21,
          'legend.fontsize': 21,
          'xtick.labelsize': 21,
          'ytick.labelsize': 21,
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




###############################################################
#-----------------training the model(CNN+LSTM)-----------------


#---------CNN----------
#-----reshaping data----
data =T_pert.reshape(1, 128, 64, 26)#training
data_test=test_data_pert.reshape(1, 128, 64, 1)# reshaping test data


model=Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(128,64,1),padding='same'))
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

y=np.zeros(1)
#training the model
model.compile(optimizer='adam', loss='mse',metrics=['accuracy','mse', 'mae', 'mape'])
model_hist=model.fit(data_test,y, epochs=50)


#evaluating model with test data
test_scores = model.evaluate(data_test, y, verbose=2)

#----model performance metrics
model_loss=model_hist.history;
model_metrics=model.metrics_names;
keras.utils.plot_model(model, "my_first_model.png");
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True) 
print(model_metrics);

#plt.plot(model_hist.history['loss'])
plt.plot(model_hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


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

y_out=np.concatenate((y_out_0,y_out_1,y_out_2,y_out_3,y_out_4,y_out_5,y_out_6, y_out_7,y_out_8,y_out_9,y_out_10,y_out_11,y_out_12,y_out_13,y_out_14,y_out_15),axis=1);
y_normalized=preprocessing.normalize(y_out);


##########################################################################
#--------------calculation of model_metrics for training data -----------------
model1=Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(128,64,26),padding='same'))
model1.add(LeakyReLU(alpha=0.01))
model1.add(MaxPooling2D((2, 2),padding='same'))
model1.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model1.add(LeakyReLU(alpha=0.01))
model1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model1.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model1.add(LeakyReLU(alpha=0.011))                  
model1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))



#model.add(Flatten())
#model.add(Dense(128, activation='linear'))
model1.add(LeakyReLU(alpha=0.1))                  
model1.summary()
plot_model(model1, to_file='train_model_plot.png', show_shapes=True, show_layer_names=True)
#---------------assigning random weights to the model--------
initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
weights=initializer(shape=(128, 64));

#----------------conforming the model weights----------------
model_weights=model1.get_weights()
#-----------------------see the dimensions of the tensor-------

y=np.zeros(1)
#training the model
model1.compile(optimizer='adam', loss='mse',metrics=['accuracy','mse', 'mae', 'mape'])
model1_hist_train=model1.fit(data,y, epochs=50)


#evaluating model with test data
test_scores_train= model1.evaluate(data, y, verbose=2)

#----model performance metrics
model_loss_train=model1_hist_train.history;
model_metrics_train=model1.metrics_names;
keras.utils.plot_model(model1, "CNN_Model_WB.pdf");
keras.utils.plot_model(model1, "CNN_Model_WB.pdf", show_shapes=True) 


#--------------plotting the model metrics---------------
fig = plt.figure(figsize=(8,6));
plt.figure(1)
e_pochs=np.arange(0,50,1);
ax = plt.subplot(2,2,1)
#ax.plot(e_pochs,model_hist.history['loss'] ,color='b',label="Test Loss",linewidth='4');
ax.plot(e_pochs,model1_hist_train.history['loss'] ,color='b',label="Train Loss",linewidth='4');
ax.legend()
ax = fig.gca()

ax.set_xticks(np.arange(0, 51, 10))
ax.set_yticks(np.arange(0, 100000, 10000))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (a)');
plt.ylabel('Loss');
plt.xlim(0,50);
plt.ylim(0,90000)



ax = plt.subplot(2,2,2)
#ax.plot(e_pochs,model_hist.history['mae'] ,color='b',label="Test Accuracy",linewidth='4');
ax.plot(e_pochs,model1_hist_train.history['mape'] ,color='b',label="Train Accuracy",linewidth='4');
ax.legend()
ax = fig.gca()

ax.set_xticks(np.arange(0, 51, 10))
ax.set_yticks(np.arange(0, 80, 10))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (b)');
plt.ylabel('Accuracy');
plt.xlim(0,50);
plt.ylim(0,30);


ax = plt.subplot(2,2,3)
ax.plot(e_pochs,model_hist.history['loss'] ,color='b',label="Test Loss",linewidth='4');
#ax.plot(e_pochs,model1_hist_train.history['loss'] ,color='r',label="Train Loss",linewidth='4');
ax.legend()
ax = fig.gca()

ax.set_xticks(np.arange(0, 51, 10))
ax.set_yticks(np.arange(0, 100000, 10000))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (a)');
plt.ylabel('Loss');
plt.xlim(0,50);
plt.ylim(0,90000)



ax = plt.subplot(2,2,4)
ax.plot(e_pochs,model_hist.history['mae'] ,color='b',label="Test Accuracy",linewidth='4');
#ax.plot(e_pochs,model1_hist_train.history['mae'] ,color='r',label="Train Accuracy",linewidth='4');
ax.legend()
ax = fig.gca()

ax.set_xticks(np.arange(0, 51, 10))
ax.set_yticks(np.arange(0, 80, 10))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (b)');
plt.ylabel('Accuracy');
plt.xlim(0,50);
plt.ylim(0,30);


















