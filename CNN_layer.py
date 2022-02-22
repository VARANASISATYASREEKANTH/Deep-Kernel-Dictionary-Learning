# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:33:54 2021

@author: asdg
"""
#------1D convolution layer--------  
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
plt.rcParams.update({'font.size':20})
params = {'backend': 'ps',
          'axes.labelsize': 20,
          'legend.fontsize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True};


from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D


T_pert=np.zeros((184,40));
loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/temperature_perturbations_hc_22_mar_2000.xls')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        T_pert[i][j]=sheet.cell_value(i,j);


#define input data
data1 = [[2, 3, 4, 1, 1, 0, 0, 0],
		[0, 3, 6, 1, 1, 8, 3, 2],
		[3, 5, 7, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = np.fft.fft(T_pert);

data = data.reshape(1, 184, 40, 1)#training

#create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(4, 4, 1)))


#define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]

#store the weights in the model
model.set_weights(weights)


#confirm they were stored
model_weights=model.get_weights()


out=np.zeros((182,38));
#apply filter to input data
yhat = model.predict(data)#provide the testing data
for j in range(1,182,1):
    for k in range(1,38,1):
        out[j][k]=yhat[0][j][k][0];
'''for r in range(yhat.shape[1]):
	# print each column in the row
	out=yhat[0][r][:][0]; '''
    
print(yhat);