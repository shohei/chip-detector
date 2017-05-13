#coding:utf-8
import numpy as np
from sklearn import datasets
import math
from numpy import dot
from numpy.linalg import inv
from numpy.linalg import det 

iris = datasets.load_iris()
data = iris.data
target = iris.target

d = 4 #特徴量の次元
K = 3 #クラス数

def Gaussian(x, mu,sigma):
    N = 1/((2*math.pi)**(1/K))*(det(sigma))**0.5*exp(-(1/2.0)*dot((dot((x-mu).T,inv(sigma))),(x-mu)))
    return N

"""initialize"""
gamma = np.zeros((len(data),K))
for g in gamma:
    g0 = random.random()
    g1 = random.random(g0) 
    g2 = 1-g0-g1
    g = np.array([g1,g2,g3])

pi = np.array([1/3.0,1/3.0,1/3.0]) #混合比

"""E-step"""
idx_0 = np.where(gamma==0)
x_0 = data[idx_0,:]
mu_0 = sum(x_0) / len(x_0) 
#mu_0s = np.zeros((len(x_0),1))*mu_0
S0 = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        S0[i,j] = sum(dot((x_0[:,i]-mu_0[:,i]),(x_0[:,j]-mu_0[:,j])))/len(x_0)

idx_1 = np.where(gamma==1)
x_1 = data[idx_1,:]
mu_1 = sum(x_1) / len(x_1) 
#mu_1s = np.zeros((len(x_1),1))*mu_1
S1 = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        S1[i,j] = sum(dot((x_1[:,i]-mu_1[:,i]),(x_1[:,j]-mu_1[:,j])))/len(x_1)

idx_2 = np.where(gamma==2)
x_2 = data[idx_2,:]
mu_2 = sum(x_2) / len(x_2) 
#mu_2s = np.zeros((len(x_2),1))*mu_2
S2 = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        S2[i,j] = sum(dot((x_2[:,i]-mu_2[:,i]),(x_2[:,j]-mu_2[:,j])))/len(x_2)


for i in range(len(data)):
    xi = x[i,:]
    gaussian_array = [Gaussian(xi,mu_0,S0),Gaussian(xi,mu_1,S1),Gaussian(xi,mu_2,S2)]
    for k in range(K):
        gamma[i][k] = (pi[k]*gaussian_array[k])/sum(dot(pi,gaussian_array))

"""M-step"""
N0 = sum((np.matrix(gamma)[:,0]))
N1 = sum((np.matrix(gamma)[:,1]))
N2 = sum((np.matrix(gamma)[:,2]))

mu_0 = 1/N0 * sum((np.matrix(gamma)[:,0])*x_0)
mu_1 = 1/N1 * sum((np.matrix(gamma)[:,1])*x_1)
mu_2 = 1/N2 * sum((np.matrix(gamma)[:,2])*x_2)
mu = np.matrix([mu_0,mu_1,mu_2])

S_0 =  sum(dot(dot((np.matrix(gamma)[:,0]),(x_0-mu_0)),(x_0-mu_0).T))/N0
S_1 =  sum(dot(dot((np.matrix(gamma)[:,1]),(x_1-mu_1)),(x_1-mu_1).T))/N1
S_2 =  sum(dot(dot((np.matrix(gamma)[:,2]),(x_2-mu_2)),(x_2-mu_2).T))/N2
S = [S_0,S_1,S_2]

N = N0+N1+N2
pi_0 = N0/N
pi_1 = N1/N
pi_2 = N2/N
pi = [pi_0,pi_1,pi_2]

x_tmp = np.concatenate([x_0,x_1],axis=0)
x = np.concatenate([x_tmp,x_2],axis=0)

zs = np.zeros((len(data),K))
for i in range(len(gamma)):
    g = gamma[i]
    k = np.where(g==max(g))[0].tolist()[0]
    zs[i][k] = 1 

L = 0
for i in range(len(zs)):
    for k in range(K):
        L = L + zs[i][k]*math.log(pi[k]) + zs[i][k]*(-d/2*math.log(2*math.pi)_- 1/2.0*math.log(det(S[k])) - 1/2.0*(dot(dot((x[i,k]-mu[k]).T,inv(S[k])),(x[i,k]-mu[k]))))



