function y=filter_1(data,fs,cut,stp)
% this is a function that do filtering
fs=fs; % sample frequency 1/5/60 for per 5 mins' sampling 
cut=cut; % cutting frequency 5e
stp=stp; % cutting steepness of cutting, 0.95 can be fine

y=lowpass(data,cut,fs,'Steepness',stp); % filter data



