function y=plotfft(channel1,fs)
% this is a function that plot the fft
    T=1/fs;    % Sampling period
    windowsize=1; % moving average window size
    L=length(channel1); % Length of signal
    channel1=detrend(channel1); % detrend
    Y=fft(channel1); % Calculating fft
    P2=abs(Y/L); % Compute the magnitude
    P1=P2(1:L/2+1); % Single-side spectrum
    P1(2:end-1)=2*P1(2:end-1); % Double the amptitude
    f=fs*(0:(L/2))/L; % Frequency axis 0 to Nyquist freq
    P1=movmean(P1,windowsize);
    plot(f(1:floor(length(channel1)*0.1)), P1(1:floor(length(channel1)*0.1)))
    xlabel('Frequency (Hz)')
    ylabel('Amplitude')
    set(findall(gcf,'-property','FontSize'),'FontSize',6)
end