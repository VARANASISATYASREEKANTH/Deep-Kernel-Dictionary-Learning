clc; 
clear all; 
close all;
AA=xlsread('test_data_22_03_2000.xlsx');

ht=AA(:,1);%%%height
AA(:,1)=[];%%%height in first column is removing
time=0:4:356;%%%time in minute
ht1=25.1:0.3:80;%%%% height
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
contourf(time+4,ht1,AA(1:184,:),'linestyle','none','levelstep',1);
colormap('jet');
caxis([140 300]);
colorbar;
xticks([0:90:360])
xlim([0 360]);
ylim([30 80]);
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
[p2 s]=polyfit(time+4,TA2,2);
polu2=polyval(p2,time);
clear p2; clear s;
Tp1(i,:)=TA2-polu2;
clear polu2 TA2;   
end
%%%%second order polynomial fit in heightwise for each time

ht1=25.1:0.3:80;
for i=1:length(time)
TA2=Tp1(1:184,i);
[p2 s]=polyfit(ht1,TA2,2);
polu2=polyval(p2,ht1);
clear p2; clear s;
Ttt(:,i)=polu2;
Tp2(:,i)=TA2-polu2';
clear polu2 TA2;
end
%%%plotting obtained perturbation
subplot(222)
contourf(time+4,ht1,Tp2,'linestyle','none','levelstep',0.1);
colormap('jet');
set(gca,'linewidth',2,'fontsize',28);
xticks([0:90:360])
xlim([0 360]);
caxis([-20 20]);
colorbar;
ylim([30 80]);
title('Temperature perturbation (K)');
ylabel('Altitude (km)');
xlabel('Time(min)');
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',28);
hcb.Title.String = "";
%%% doing fft analysis in timewise
smt=4;%%%minute
smf=1/smt;
for i=1:length(ht1)%%%%heightwise
xx=length(Tp2(i,:));
TT = (0:xx-1)*smt;    
NFFT = 2^nextpow2(xx); % Next power of 2 from length of y
Y1 = fft(Tp2(i,:),NFFT)/xx;
f= smf/2*linspace(0,1,NFFT/2+1);
pr=1./f;
Y1=Y1';
f= smf/2*linspace(0,1,NFFT/2+1);
pr=1./f;
YY1(i,:)=Y1(1:length(f));
end
%%%%% plotting frequency spectra
subplot(223)
contourf(1./f(2:end),ht1,2.*abs(YY1(:,2:end)),'linestyle','none','levelstep',0.1);
colormap('jet');
caxis([0 20]);
colorbar;
ylim([30 80]);
xticks([0:90:360])
xlim([0 360]);

%xlim([8 20]);
title('Amplitude of Frequency Spectrum (K)');
ylabel('Altitude (km)');
xlabel('Period(min)');

set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',28);
hcb.Title.String = "";
%%%doing fft in height domain
smt=0.3;%%%in km
smf=1/smt;
for i=1:length(time)%%%%timewise
xx=length(Tp2(:,i));
TT = (0:xx-1)*smt;    
NFFT = 2^nextpow2(xx); % Next power of 2 from length of y
Y1 = fft(Tp2(i,:),NFFT)/xx;
f= smf/2*linspace(0,1,NFFT/2+1);
pr=1./f;
Y1=Y1';
f= smf/2*linspace(0,1,NFFT/2+1);
pr=1./f;
YY2(i,:)=Y1(1:length(f));
end
%%%%%%plotting wavenumber spectra
subplot(224)
contourf(time+4,1./f(2:end),2.*abs(YY2(:,2:end))','linestyle','none','levelstep',0.1);
colormap('jet');
caxis([0 4]);
colorbar;
ylim([0 15]);
xticks([0:90:360])
xlim([0 360]);
set(gca,'linewidth',2,'fontsize',28);
ylabel('Vertical wavelength(km)');
xlabel('Time(min)');
title('Amplitude of Wavenumber spectrum (K)');
set(gca,'Fontweight','bold');
hcb.Title.String = "";
%%%%%%bpf 5-8 km bandpass filtering 
     pr1=11;pr2= 14;%--------------------------------------------------------------------------------
    fcut1=1./pr2;fcut2=1./pr1;
    [a,b]=butter(2,[(2*fcut1)/smf  (2*fcut2)/smf],'Bandpass');
for i=1:length(time)
TQB(:,i)=filter(a,b,Tp2(:,i));

end




%%%%5-8km band pass filterd
figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [20 30]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 20 30]);

subplot(211)
contourf(time+4,ht1,TQB,'linestyle','none','levelstep',0.1);
colormap('jet');
caxis([-4 4]);
colorbar;
ylim([30 80]);
xticks([0:60:240])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',24);
title('\lambda_z=[4.5km, 4.8km]');
set(gca,'Fontweight','bold');
ylabel('Altitude(km)');
xlabel('Time(min)');
hcb.Title.String = "";
%%%%%15-40min bandpass filtering
    pr1=15;pr2=30;%--------------------------------------------------------------------------------
    fcut1=1./pr2;fcut2=1./pr1;
    [a,b]=butter(2,[(2*fcut1)/smf  (2*fcut2)/smf],'Bandpass');
for i=1:length(ht1)
TQA(i,:)=filter(a,b,TQB(i,:));

end
%%%%%%%5-8km band pass filterd+15-40min bandpass filtering 
subplot(212)
contourf(time+4,ht1,TQA,'linestyle','none','levelstep',0.1);
colormap('jet');
caxis([-0.4 0.4]);
colorbar;
ylim([30 80]);
ylabel('Altitude(km)');
xlabel('Time(min)');
xticks([0:60:240])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',24);
title('\lambda_z=[4.5km, 4.8km],T=[15min, 80min]');
set(gca,'Fontweight','bold');
hcb.Title.String = "";

%---------------------------------------------------------------------------



%display of a_e
ht1=25.1:0.3:80;
for i=1:length(time)
TA2=Tp1(1:184,i);
[p2 s]=polyfit(ht1,TA2,2);
polu2=polyval(p2,ht1);
clear p2; clear s;
Ttt(:,i)=polu2;
Tp2(:,i)=TA2-polu2';
clear polu2 TA2;
end
k=mean(transpose(AA));

for i=1:1:184
    for j=1:1:60
    ae(i,j)=(2*3.14*9.8*abs(TQA(i,j)))./(4.2e3*1e-4*transpose(k(i)));
    end
end
%----------------------------------
abs_ae=abs(ae);
%(bla - min(bla)) / ( max(bla) - min(bla) )
norm_ae=abs_ae./max(abs_ae);
figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [15 5]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 15 5]);
%set(gcf,'position',[633.8000000000001,177.8,638.4000000000002,514.4000000000001]);
hcb.Title.String = "";


 
subplot(1,3,1)
contourf(time+4,ht1,norm_ae);
colormap('jet');
colorbar;
caxis([0, 1])
ylim([30 80]);
xticks([0:60:240])
xlim([0 240]);
title('a_e(Hz)');
ylabel('Altitude (km)');
xlabel('Time(min)');
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',24);
hcb.Title.String = "";
%-------INTRINSIC_FREQUENCY
f=3.330e-3; %inertial frequency=2*omega*sin(latitude) 
N=1e-2;% Brunt vaisala frequency

for i=1:1:184
    for j=1:1:60
w_intrinsic(i,j)=(f*(2-(norm_ae(i,j))))/(2*sqrt(1-(norm_ae(i,j))));
    end
end



subplot(1,3,2)
contourf(time+4,ht1,w_intrinsic);
xticks([0:60:240])
xlim([0 240]);
colormap('jet');
%caxis([4e-3,16e-3])
colorbar;
caxis([4e-3, 16e-3])
ylim([30 80]);
title('f_{Instrinsic}(Hz)');
ylabel('Altitude (km)');
xlabel('Time(min)');
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',24);
f_period=2.08e-3*ones(184,60);

subplot(1,3,3)
f_dop=f_period-w_intrinsic/(2);
f_dop(isinf(f_dop))=0;
contourf(time+4,ht1,f_dop);
colormap('jet');
colorbar;
caxis([-0.05, 0])
ylim([30 80]);
xticks([0:60:240])
xlim([0 240]);
title('f_{Doppler}(Hz)');
ylabel('Altitude (km)');
xlabel('Time(min)');
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',24);


%---HORIZONTAL WAVELENGTH------
lambda_z=4e3;%meters

H_WAVE_LENGTH=2*lambda_z*N*(sqrt(1-ae))./(f*ae);

%----------------------------------------------
figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [15 5]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 15 5]);
%set(gcf,'position',[633.8000000000001,177.8,638.4000000000002,514.4000000000001]);

hspd=zeros(60,128);
hs =0.0033;   
for i=1:1:60
nfft = 2^nextpow2(length(Tp2(:,i)));
Pxx = abs(fft(Tp2(:,i),nfft)).^2/length(Tp2(:,i))/hs;
Hpsd = dspdata.psd(Pxx(1:length(Pxx)/2),'Fs',hs);  
hspd(i,:)=(Hpsd.Data);
end
%hspd_normalize=normalize(hspd);
%plot(hspd_normalize(1:5,1), hspd_normalize(1:5,3))
max_value=max(hspd);
k1=hspd(1,:)./hspd(45,:);
k2=hspd(60,:)./hspd(45,:);
for j=1:1:128
    k2(i)=k2(128-i);
    k1(i)=k1(128-i);
end
scatter( -k1(1:128)+50 ,k2(1:128));
xlim([0,50]);
ylim([-100,100]);



q=(pwelch(Tp1(:,1:60)));
m_o=pwelch(Tp1(10,1:60))/pwelch(Tp1(2,1:60));
m_m=pwelch(Tp1(110,1:60))/pwelch(Tp1(2,1:60));
figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [20 10]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 20 10]);

[m,txt,raw]=xlsread(strcat('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3_study_of_gravity_waves_with_rayleigh_lidar\working_programs\programs\MScDL_gw_analysis_estimation\mra_analysis\testing_data\mo_mm.xlsx'));






subplot(1,1,1)
plot(1./m_o(125:129,67), 1./m_m(125:129,67) ,'b','linewidth',3);
hold on
plot(1./(m_o(125:129,67)), 1./(m_m(125:129,67)/1.75) ,'r','linewidth',3);
%hold on
%plot((y_dash+50),x_dash-50,'r','linewidth',4);
set(gca,'linewidth',4,'fontsize',24);
xlim([0.4,1.2]);
ylim([0,5]);

xlabel('Source Wavenumber[m_0/m_c]');
ylabel('Doppler Shift corrected Wavenumber[m/m_c]');
title('Effect of reduction of Doppler Shift on vertical Wavenumber');


%---study on reducing background effect on vertical wave number---
q=(pwelch(Tp1(:,1:60)));
m_with_wind_x=20*(pwelch(Tp1(10,1:60))./pwelch(Tp1(90,1:60)));
m_with_wind_z=20*(pwelch(Tp1(129,1:60))./pwelch(Tp1(90,1:60)));

Tp1_th=Tp1;
for i=1:1:184
    for j=1:1:60
        if(Tp1_th(i,j)>2)
            Tp1_th(i,j)=1.5;
        else
        end
        
        
    end
end
m_without_wind_x=20*(pwelch(Tp1_th(10,1:60))./pwelch(Tp1_th(90,1:60)));
m_without_wind_z=20*(pwelch(Tp1_th(129,1:60))./pwelch(Tp1_th(90,1:60)));






%plot(2.5*m_with_wind_x(1:40,1),2.5*m_with_wind_z(1:40,1));
%hold on
%plot(2.5*m_without_wind_x(1:40,1),2.5*m_without_wind_z(1:40,1));


figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [5 10]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 5 10]);

subplot(1,1,1)
plot(m_with_wind_x(1:40,1), m_with_wind_z(1:40,1),'b','linewidth',2);
hold on
plot(m_without_wind_x(1:40,1), m_without_wind_z(1:40,1)  ,'r','linewidth',2);
%hold on
%plot((y_dash+50),x_dash-50,'r','linewidth',4);
set(gca,'linewidth',2,'fontsize',24);
xlim([0,5]);
ylim([0,40]);
xlabel('Source Wavenumber[m_0/m_c]');
ylabel('Doppler Shift corrected Wavenumber[m/m_c]');
title('Effect of reduction of Doppler Shift on vertical Wavenumber');














