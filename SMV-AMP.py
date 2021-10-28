#Multi-Measurement Vector Approximate Message Passing (AMP) 
#J = 1 , beta = 1 (i.e., SMV)
#Code by K. Ishibashi, 2021
import numpy as np
N = 100 #number of users
M = 80  #number of observations (sequence length)
K = 2  #number of active users
epsilon = K/N #activity ratio 
itermax = 100 #number of iterations for MMV-AMP
SNR = 20 #SNR

#MMSE Denoiser
def denoiser(x,tau2):
    psi=np.log(1.0 + 1.0/tau2)
    pit= 1/1 * (1.0/tau2 - 1.0/(1.0+tau2)) * np.power(np.abs(x.A),2)
    phi = 1.0/(1.0 + (1.0-epsilon)/epsilon * np.exp(-1 * (pit - psi)))
    return phi*(1.0/(1.0+tau2))*x.A, 1/tau2 * ((phi * tau2)/(1+tau2) + (phi*(1-phi))/((1+tau2)**2) * x.A * np.conj(x.A))

#Threshold function
def AUD(u, theshold):
    return np.piecewise(u,[u<theshold,u>=theshold],[0.0,1.0])

#Calculation of MSE
def mean_squared_error(y, t):
    return np.sum(np.abs(y-t)**2)/np.size(y)

sig2 = 1.0/M * 1.0/np.power(10.0,SNR/10.0) #noise variance

#N signals w/ K nonzero elements
x = np.zeros((N,1)) + 1j*np.zeros((N,1)) #generate an array w/ N zeros
x[0:K] = np.sqrt(0.5)*(np.random.normal(size=(K,1)) + np.random.normal(size=(K,1)) * 1j) #put k nonzero elements
np.random.shuffle(x) #shuffle it
actX = AUD(np.abs(x),1e-9)

#Observation matrix (M x N), iid Gaussian w/ mean 0, var 1/M
A = np.asmatrix(np.sqrt(1.0/(2.0*M)) * (np.random.normal(size=(M,N)) + np.random.normal(size=(M,N)) * 1j))

#(M x J) observation signals 
y = np.dot(A,x) + np.sqrt(0.5 * sig2)*(np.random.normal(size=(M,1)) + np.random.normal(size=(M,1)) * 1j)

#Appproximate Message Passing ***************
#Initialization
xhat = np.zeros((1,N)) #hat{x}
R = y #residual

for t in range(itermax): #iteration count
    tau2 = 1.0/(1*M) * np.power(np.linalg.norm(R),2)
    nxhat, etap = denoiser(R.T @ A.T.H + xhat,tau2)
    nxR = y - A @ np.asmatrix(nxhat).T + N/M * R * etap.mean()
    R = nxR
    xhat = nxhat
    
#Active User Detection
hatactX = np.reshape(AUD(np.power(np.abs(R.T @ A.T.H + xhat),2.0), 1/(1.0/tau2 - 1.0/(tau2 + 1.0)) * np.log(1+1.0/tau2)),(N,1))

#SHOW AUD RESULTS
print('true:',np.reshape(actX,(1,N)))
print('est :',np.reshape(hatactX,(1,N)))

