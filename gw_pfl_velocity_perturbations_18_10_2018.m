clc; 
clear all; 
close all;
AA=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\programs\anamoly_detection_LSTM_RNN\meridonal_wind_18_10_2018.xlsx');

ht=AA(:,1);%%%height
AA(:,1)=[];%%%height in first column is removing
time=0:4.2:600;%%%time in minute
ht1=70.053:0.03:122.613;%%%% height
%%%%plotting temperature data
figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [30 20]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 30 20]);

subplot(221)
contourf(time(1:141),ht1(1:1753),AA(1:1753,1:141),'linestyle','none','levelstep',1);
colormap('jet');
caxis([-100 100]);
colorbar;
xticks([0:60:645])
xlim([0 240]);
%ylim([80 95]);
title('Temperature (K)');
ylabel('Altitude (km)');
xlabel('Time(min)');
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',28);
hcb.Title.String = "";
%%%%considering 30-60km for gw analysis
%%%%second order polynomial fit in timewise for each height
for i=1:length(ht)
    TA2=AA(i,:);
[p2 s]=polyfit(time(1:141),TA2,2);
polu2=polyval(p2,time(1:141));
clear p2; clear s;
Tp1(i,:)=TA2-polu2;
clear polu2 TA2;   
end
%%%%second order polynomial fit in heightwise for each time
for i=1:length(time(1:141))
TA2=Tp1(1:1753,i);
[p2 s]=polyfit(ht1,TA2,2);
polu2=polyval(p2,ht1);
clear p2; clear s;
Ttt(:,i)=polu2;
Tp2(:,i)=TA2-polu2';
clear polu2 TA2;
end
%%%plotting obtained perturbation
z=normalize(abs(fft(Tp2)));