import numpy as np
N = 1000 #number of users
M = 200  #number of observations (sequence length)
K = 80  #number of active users
J = 1  #number of antennas
L = 100000 #number of monte carlo averaging 
epsilon = K/N #activity ratio 
omega = N/M #overload ratio
itermax = 20 #number of iterations for MMV-AMP
SNR = 10 #SNR

sig2 = 1.0/M * 1.0/np.power(10.0,SNR/10.0) #noise variance
tau2 = sig2 + omega * epsilon

for t in range(itermax): #iteration count
    x = np.sqrt(0.5)*np.sqrt(tau2)*(np.random.normal(size=(L,1)) + np.random.normal(size=(L,1)) * 1j)
    Km = (int)(L/N * K)
    x[0:Km] += np.sqrt(0.5)*(np.random.normal(size=(Km,1)) + np.random.normal(size=(Km,1)) * 1j) 

    psi=np.log(1.0 + 1.0/tau2)
    pit= 1/J * (1.0/tau2 - 1.0/(1.0+tau2)) * np.power(np.abs(x),2)
    phi = 1.0/(1.0 + (1.0-epsilon)/epsilon * np.exp(-1 * (pit - psi)))
    tmp = phi * (1-phi) * 1/np.power(1+tau2,2) * np.power(np.abs(x),2)
    theta = tmp.mean() * 1/J
    tau2 = sig2 + omega*epsilon*tau2/(1+tau2) + omega*theta
    print(t, tau2)


    