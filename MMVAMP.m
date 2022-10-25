% MMV-AMP Based on L. Liu and W. Yu, “Massive Connectivity With Massive MIMO - Part I: Device Activity Detection and Channel Estimation,” IEEE Trans. Signal Process., vol. 66, no. 11, pp. 2933–2946, 2018.
% By Koji Ishibashi
% NOTE : Without loss of generality, beta_i is set as 1.
% A : sensing matrix
% Y : observation
% M : length of sequence
% N : number of users 
% J : number of antennas at the BS
% Tmax : maximum number of iterations
% epsilon : activity ratio
function [Xhat,activeset,mse]=MMVAMP(Y,A,Tmax,epsilon)

%Initialization
[M,J] = size(Y);
N = size(A,2); 
Xhat = zeros(N,J); % estimate of X
mse = zeros(Tmax,1); %Mean-squared error of every iteration
coef = (1-epsilon)/epsilon; %Coefficient c.t. activity ratio
R = Y; % Residual matrix

for t=1:Tmax %iteration for MMV-AMP
    mse(t) = 1/(J*M) * norm(R,'fro')^2;
    Temp = R.'*conj(A)+Xhat.'; %Temporal estimate
    
    % Denoising 
    Phi = 1./(1 + coef * exp(-J * (((1/mse(t) - 1/(mse(t)+1))/J * vecnorm(Temp,2,1).^2) - log(1+1/mse(t)))));
    Xhat = (1/(1+mse(t)) * Phi .* Temp).';
    
    %Residual Calculation (Using Tensor)
    mat2 = repmat(permute(Temp,[1 3 2]), 1, size(Xhat,2));
    R = Y - A*Xhat + N/M * R * mean(1/mse(t) * (mse(t)/(1+mse(t)) * permute(Phi,[1,3,2]) .* reshape(repmat(eye(J),1,N),[J,J,N]) + (permute(Phi,[1,3,2]) .* (1-permute(Phi,[1,3,2])))/(1+mse(t))^2 .* mat2.* permute(conj(mat2),[2 1 3])),3);
end

%Active User Detection
Temp = R.'*conj(A)+Xhat.';
activeset = (vecnorm(Temp,2,1).^2 > (J/(1/mse(Tmax) - 1/(mse(Tmax)+1)) * log(1 + 1/mse(Tmax)) .* ones(1,N))).';