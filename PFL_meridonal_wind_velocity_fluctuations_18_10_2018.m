clc; 
clear all; 
close all;
AA=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\programs\anamoly_detection_LSTM_RNN\test_data_21_10_1998.xls');
m_vel=zeros(184,90)
ht=AA(:,1);%%%height
AA(:,1)=[];%%%height in first column is remov
for i=1:1:184
m_vel(i,2:90)=AA(i,2:90)-mean(AA(i,2:90));
end
k=abs(fft(m_vel(5:184,2:90)));