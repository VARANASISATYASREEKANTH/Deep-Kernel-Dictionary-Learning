# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 10:46:46 2021

@author: asdg
"""

#####################################################################
#----------------------------LSTM autoencoder------------------------
# lstm autoencoder reconstruct and predict sequence
from numpy import array
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
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

loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/test_data/3_jan_2014/y_out_03_01_2014.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
test_data_pert=np.zeros((sheet.nrows,sheet.ncols));
test_data_dash_pert=np.zeros((sheet.nrows,sheet.ncols));
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        test_data_dash_pert[i][j]=sheet.cell_value(i,j)

seq_in=np.zeros(128);
seq_in=array([40.5006,
63.5296	,
113.219	,
0.10103	,
110.435	,
78.4455	,
37.7941	,
85.2746	,
-0.00438969,
-0.470995,
207.192	,
-0.544608,
24.4891,
183.038,
25.6426,
53.3864,
51.635,
69.7868,
21.5968,
176.811,
176.113,
255.337,
66.9308,
136.636,
53.9459	,
-0.0791578,
202.23	,
20.2804	,
77.224	,
31.9757	,
56.9025	,
-0.303029,
113.413	,
108.996	,
47.7453	,
29.1974	,
102.546	,
254.489	,
47.4846	,
8.21992	,
67.5193	,
44.9223	,
32.9224	,
234.854	,
-0.329959,
-0.306543,
125.768	,
-0.781176,
79.0747	,
209.461	,
160.34	,
56.654	,
108.95	,
57.3484	,
31.8995	,
153.633	,
169.104	,
60.9912	,
43.7753	,
80.1722	,
22.7595	,
76.4158	,
212.531	,
82.3213	,
51.0142	,
59.8258	,
57.2794	,
8.85655	,
-0.0624068,
109.441	,
298.517	,
98.7884	,
50.0498	,
-0.00959143,
30.9748	,
-0.281629,
51.005	,
163.248	,
145.935	,
-0.250398,
60.0401	,
-0.120299,
3.31207	,
125.608	,
1.50908	,
56.8624	,
140.782	,
29.7454	,
-0.0192172,
104.692,
46.8,
-0.327449,
-0.276155,
120.34	,
94.45	,
9.80447	,
39.7884	,
14.7714	,
20.0305	,
4.71481	,
-0.063011,
199.708	,
4.6031	,
-0.148795,
175.065	,
77.4196	,
140.132	,
55.9727	,
115.491	,
62.2757	,
-0.384805,
111.654	,
-0.436272,
-0.176741,
59.7676,
163.522,
22.0553,
182.122,
132.646,
25.68,
34.0619,
38.4882,
123.86,
103.779,
17.5397,
67.9144,
143.485,
175.239])

# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))
# prepare output sequence
seq_out = seq_in[:, 1:, :]#-----output_predicted_data
n_out = n_in - 1;
# define encoder
visible = Input(shape=(n_in,1))
encoder = LSTM(100, activation='relu')(visible)
# define reconstruct decoder
decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)
# define predict decoder
decoder2 = RepeatVector(n_out)(encoder)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)
#-------------------tie it together--------------
model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse')
plot_model(model, show_shapes=True, to_file='composite_lstm_autoencoder.png')
#-----------------fit model---------------------
hist=model.fit(seq_in, [seq_in,seq_out], epochs=100, verbose=0)
# demonstrate prediction
yhat = model.predict(seq_in, verbose=0)#---provide the test data
print(yhat);




#-----------------list all data in history----------------
print(hist.history.keys())
'''# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss'''
plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
























