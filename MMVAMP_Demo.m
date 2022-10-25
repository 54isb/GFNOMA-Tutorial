%MMV-AMP Demo 
%Based on L. Liu and W. Yu, “Massive Connectivity With Massive MIMO - Part I: Device Activity Detection and Channel Estimation,” IEEE Trans. Signal Process., vol. 66, no. 11, pp. 2933–2946, 2018.
%By Koji Ishibashi
%
clear all; close all; clc;
tic

% Simulation parameters 
N = 200; %Number of users
M = 60; %Length of sequence
K = 20; %Number of active users
J = 1; %Number of antennas at the BS
SNR = -10:2:20; % SNR range (e.g. -10:2:20)
Ite = 100000; % Iteration of Monte Carlo simulations (e.g. 1e+4)
Tmax = 100; % Maximum iteration of MMV-AMP

% Initialization
nvar = 1/M * 10.^(-SNR./10); % Noise variance
epsilon = K/N; %Activity ratio

%Measurement matrix based on partial DFT (normalized)
DFT_M = dftmtx(N); % Generate NxN DFT matrix
A = DFT_M(sort(randperm(N,M)),:); %Choose M rows
A = 1./vecnorm(A) .* A; %Normalize columns

%Performance metrics
MD=zeros(length(SNR),1); %Miss detection 
FA=zeros(length(SNR),1); %False alarm 
NMSE=zeros(length(SNR),1); %NMSE of the channel estimation

%K-Sparse Vector 
S = zeros(N,1);
S(1:K,:)=ones(K,1);

% Loop for SNR
parfor sn = 1:length(SNR)
   % Loop for Monte Carlo Simulations
   for i = 1:Ite
       % Active User Selection
       S_t = S(randperm(N)); % Choose K active users randomly
       Active_Set = find(S_t); % Indices of active users
       
       % Channel Generation
       H = sqrt(0.5)*(randn(N,J)+1j*randn(N,J)) .* repmat(S_t,1,J); 
        
       % Received Signal
       Y = A*H + sqrt(0.50*nvar(sn))*(randn(M,J)+1j*randn(M,J)); 
       
       % MMV-AMP
       [Xhat,est_Active_set,mse]=MMVAMP(Y,A,Tmax,epsilon);
   
       %Miss Detection
       MD(sn) = MD(sn)+(K - sum(ismember(find(est_Active_set),Active_Set)))/(K*Ite);
       %Flase Alarm
       FA(sn) = FA(sn)+sum(~ismember(find(est_Active_set),Active_Set))/((N-K)*Ite);
       %NMSE Calculation
       NMSE(sn) = NMSE(sn) + 1/Ite * norm(Xhat - H,'fro')^2/norm(H,'fro')^2;
   end
end
    
% Plot MD/FA Probabilities
figure(1)
semilogy(SNR,MD,'-or')
hold on
semilogy(SNR,FA,'-^k')
grid on
ylim([1e-4 1e-0])
xlim([-10 20])
xlabel('SNR[dB]')
ylabel('Probability')
legend('Miss detection','False alarm')
% Plot NMSE Performance
figure(2)
semilogy(SNR,NMSE,'-or')
hold on 
grid on
ylim([1e-4 1e-0])
xlim([-10 20])
xlabel('SNR[dB]')
ylabel('NMSE')
legend('NMSE')

toc