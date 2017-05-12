#coding:utf-8
import numpy as np
from sklearn import datasets
import math

iris = datasets.load_iris()
data = iris.data
target = iris.target

d = 4 #特徴量の次元
K = 3 #クラス数

def Gaussian(x, mu,sigma):
    N = 1/((2*math.pi)**(1/K))*(np.linalg.det(sigma))**0.5*exp(-(1/2.0)*np.dot((np.dot((x-mu).T,np.linalg.inv(sigma))),(x-mu)))
    return N

"""initialize"""
gamma = np.zeros((len(data),1))
for g in gamma:
    g = int(math.floor(random.random()*3))

pi = np.array([1/3.0,1/3.0,1/3.0]) #混合比

"""E-step"""
idx_0 = np.where(gamma==0)
x_0 = data[idx_0,:]
mu_0 = sum(x_0) / len(x_0) 
#mu_0s = np.zeros((len(x_0),1))*mu_0
S0 = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        S0[i,j] = sum(np.dot((x_0[:,i]-mu_0[:,i]),(x_0[:,j]-mu_0[:,j])))/len(x_0)

idx_1 = np.where(gamma==1)
x_1 = data[idx_1,:]
mu_1 = sum(x_1) / len(x_1) 
#mu_1s = np.zeros((len(x_1),1))*mu_1
S1 = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        S1[i,j] = sum(np.dot((x_1[:,i]-mu_1[:,i]),(x_1[:,j]-mu_1[:,j])))/len(x_1)

idx_2 = np.where(gamma==2)
x_2 = data[idx_2,:]
mu_2 = sum(x_2) / len(x_2) 
#mu_2s = np.zeros((len(x_2),1))*mu_2
S2 = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        S2[i,j] = sum(np.dot((x_2[:,i]-mu_2[:,i]),(x_2[:,j]-mu_2[:,j])))/len(x_2)


for i in range(len(data)):
    xi = x[i,:]
    gaussian_array = [Gaussian(xi,mu_0,S0),Gaussian(xi,mu_1,S1),Gaussian(xi,mu_2,S2)]
    for k in range(K):
        gamma[i][k] = (pi[k]*gaussian_array[k])/sum(np.dot(pi,gaussian_array))









