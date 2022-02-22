# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 09:49:42 2021

@author: asdg
"""
#------for convolution layer-----
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
#----------for LSTM-----------
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
plt.rcParams.update({'font.size':10})
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'legend.fontsize':10,
          'xtick.labelsize':10,
          'ytick.labelsize':10,
          'text.usetex': True};
#--------------80% train data, 20% test data----------

loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/pocker_flat_lidar/meridonal_wind_18_10_2018_ver2.xls')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
m_vel=np.zeros((sheet.nrows,sheet.ncols))
m_vel_pert=np.zeros((sheet.nrows,sheet.ncols))
m_vel_fft=np.zeros((sheet.nrows,sheet.ncols))
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        m_vel[i][j]=sheet.cell_value(i,j)
        
for k in range(667):
    for m in range(141):
        m_vel_pert[k][m]=m_vel[k][m]-np.mean(m_vel[k][1:141])
m_vel_fft=np.abs(np.fft.fft2(m_vel_pert, s=None, axes=(- 2, - 1), norm=None));



from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
from tensorflow.keras import layers
from tensorflow.keras import initializers
#define input data
data1 = np.abs(np.fft.fft(m_vel_fft));

data =( data1.reshape(1, 667, 142, 1))#training

#--------create model
model = Sequential()
model.add(Conv2D(1, (3,3),activation='linear', input_shape=(667, 142, 1)))

#---adding several layers----
'''
model=Sequential()
model.add(Conv2D(8, kernel_size=(3, 3),activation='linear',input_shape=(667,142,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(16, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
#model.add(Dense(num_classes, activation='softmax'))

#define a vertical line detector'''
'''detector = [[[[0]],[[1]],[[0]],[[1]],[[0]],[[1]],[[0]],[[1]]],
            [[[1]],[[0]],[[1]],[[1]],[[0]],[[1]],[[1]],[[0]]],
            [[[1]],[[1]],[[0]],[[1]],[[0]],[[1]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]'''
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
values = initializer(shape=(667, 142))
#store the weights in the model
#model.set_weights(weights)

print(values);
#confirm they were stored
model_weights=model.get_weights()


out=np.zeros((667,64));
#apply filter to input data
yhat = model.predict(data)#provide the testing data
for j in range(1,64,1):
    for k in range(1,64,1):
        out[j][k]=yhat[0][j][k][0];
'''for r in range(yhat.shape[1]):
	# print each column in the row
	out=yhat[0][r][:][0]; '''
    
print(yhat);





##################################################
#-------------------------------------------------
#------------------LSTM network-------------------
#------------define input sequence----------------
seq_in=np.abs(out[1][:]);
#-----reshape input into [samples, timesteps, features]---
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))
#--------prepare output sequence-------------------
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1
#-------------define encoder---------------
visible=Input(shape=(n_in,1))
encoder=LSTM(100, activation='relu')(visible)
#-----------define reconstruct decoder---------
decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)
#-------------define predict decoder----------
decoder2 = RepeatVector(n_out)(encoder)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)
#-----------tie it together-------------
model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse',metrics=['accuracy','mse', 'mae', 'mape'])
plot_model(model, show_shapes=True, to_file='composite_lstm_autoencoder.png')
#-------------fit model-----------
model_hist=model.fit(seq_in, [seq_in,seq_out], epochs=10, verbose=0)
#-----------demonstrate prediction--------
yhat_lstm_out=np.zeros(( 667, 142));
for i in range(1,667,1):
    seq_in=out[i][:];
    print(seq_in);
    yhat = model.predict(out[i][:], verbose=1)
    for j in range(1,142,1):
        yhat_lstm_out[i][j]=np.transpose(yhat[0][j][0]);










#-----calculation of model validation and accuracy----------
print('')
model_metrics=model.metrics_names
model_loss=model_hist.history;
#out=np.fft.ifft(yhat);
#-----------------------------------------------              
#----meridonal_contour------------  
fig = plt.figure(figsize=(10,10))
plt.figure(1)
ax=plt.subplot(2,2,1)
#start, stop, n_values = 4, 600, 140
#start1, stop1, n_values1 = , 80, 184
y_vals = np.linspace(80, 100, 667)
x_vals = np.linspace(1, 600, 142)
X, Y = np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y, m_vel_pert, cmap='jet')
plt.colorbar(cp)
plt.clim(-100, 100)
ax.set_title('Meridonal Wind Perturbations(m/s)' )
ax.set_xlabel('Time(minutes) \n (a)')
ax.set_ylabel('Height(km)')
ax.set_xticks(np.arange(0, 660, 60))
ax.set_yticks(np.arange(80, 105, 5))
plt.xlim(0,600);
plt.ylim(80,100);
        

ax=plt.subplot(2,2,2)
#start, stop, n_values = 4, 600, 140
#start1, stop1, n_values1 = , 80, 184
y_vals = np.linspace(80, 100, 665)
x_vals = np.linspace(1, 600, 139)
X, Y = np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y, out, cmap='jet')
plt.colorbar(cp)
plt.clim(-10000, 10000)
ax.set_title('CNN' )
ax.set_xlabel('Time(minutes) \n (a)')
ax.set_ylabel('Height(km)')
ax.set_xticks(np.arange(0, 660, 60))
ax.set_yticks(np.arange(80, 105, 5))
plt.xlim(0,600);
#plt.ylim(80,100);              
        
        
        
        
        
ax=plt.subplot(2,2,3)
#start, stop, n_values = 4, 600, 140
#start1, stop1, n_values1 = , 80, 184
y_vals = np.linspace(80, 100, 667)
x_vals = np.linspace(1, 600, 142)
X, Y = np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y, m_vel_pert, cmap='jet')
plt.colorbar(cp)
plt.clim(-100, 100)
ax.set_title('LSTM' )
ax.set_xlabel('Time(minutes) \n (a)')
ax.set_ylabel('Height(km)')
ax.set_xticks(np.arange(0, 660, 60))
ax.set_yticks(np.arange(80, 105, 5))
plt.xlim(0,600);
#plt.ylim(80,100);       
        
        
        
    
ax=plt.subplot(2,2,4)
#start, stop, n_values = 4, 600, 140
#start1, stop1, n_values1 = , 80, 184
y_vals = np.linspace(80, 100, 667)
x_vals = np.linspace(1, 600, 142)
X, Y = np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y, m_vel_pert, cmap='jet')
plt.colorbar(cp)
plt.clim(-100, 100)
ax.set_title('CNN+LSTM' )
ax.set_xlabel('Time(minutes) \n (a)')
ax.set_ylabel('Height(km)')
ax.set_xticks(np.arange(0, 660, 60))
ax.set_yticks(np.arange(80, 105, 5))
plt.xlim(0,600);
#plt.ylim(80,100);          
        
        
        
        
        
        
