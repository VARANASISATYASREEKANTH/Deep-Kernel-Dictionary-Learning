clc; 
clear all; 
close all;
AA=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\programs\anamoly_detection_LSTM_RNN\pocker_flat_lidar\density_data_breaking_event.xlsx');
den_fft=abs(fft(AA(1:1645,1:44),1645));
clim([0,500])