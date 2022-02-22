# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 09:49:42 2021

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
#--------------80% train data, 20% test data----------
vel_pert=np.zeros((1754,142));
vel_dash_pert=np.zeros((1754,142));
loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/meridonal_wind_18_10_2018.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        vel_dash_pert[i][j]=sheet.cell_value(i,j)
        #T_pert[i][j]= T_pert[i][j]-sheet.cell_value(i,61);
    
        #if ((T_pert[i][j]>400) or (T_pert[i][j]<-400)):
            
            #T_pert[i][j]=40;
            
            
            
#-----------pre processing------


for i in range(1,1754,1):
    for j in range(1,142,1):
        vel_pert[i][j]=vel_dash_pert[i][j]-vel_dash_pert[i][140]
        '''if T_pert[i][j]>40:
            T_pert[i][j]=40;
        if T_pert[i][j]<-40:
            T_pert[i][j]=-40;'''

#-------performing the FFT Transform-----------------        
        
vel_per_fft=(np.fft.fft2(vel_pert, s=None, axes=(- 2, - 1), norm=None)); 
      
vel_per_fft_abs=np.abs(np.fft.fft2(vel_pert, s=None, axes=(- 2, - 1), norm=None));
from scipy import signal
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=8):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#---------------filtering----------------------
vel_filtered=butter_bandpass_filter(vel_per_fft_abs, 1, 3,10, order=8)
vel_filtered=butter_bandpass_filter(np.transpose(vel_filtered), 1.8, 2,10, order=8)
vel_filtered=np.transpose(vel_filtered);
#--------------------Wave Breaking PFL------------------------- 



fig = plt.figure(figsize=(5,20))
plt.figure(1)
ax=plt.subplot(2,2,1)
start, stop, n_values = 1, 142, 142
start1, stop1, n_values1 = 70.053, 122.643, 1754
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y = np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y, np.normalize(vel_filtered),cmap='jet')
plt.colorbar(cp);
#plt.clim(-100, 100)
ax.set_title('FFT' );
ax.set_xlabel('Time(minutes) \n (a)');
ax.set_ylabel('Height(km)');
#ax.set_xticks(np.arange(0, 240, 60));
#ax.set_yticks(np.arange(80, 90, 10));
#plt.clim(0,5000)
#plt.ylim(80, 90);

