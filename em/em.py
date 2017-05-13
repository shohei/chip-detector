#coding:utf-8
import numpy as np
from sklearn import datasets
import math
from numpy import dot
from numpy.linalg import inv
from numpy.linalg import det 
import random
from numpy.matlib import repmat
from numpy import matrix
import pdb
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = iris.data
target = iris.target

d = 4 #特徴量の次元
K = 3 #クラス数

L_last = 0 #直前ステップでの対数尤度
xs = data[:,0] # for scatter plot
ys = data[:,2]
plot_s = [30 for i in range(len(xs))] #scatter plot point size

"""functions"""
def Gaussian(x, mu,sigma):
    coef = 1/(((2*math.pi)**(1/K))*(det(sigma))**0.5)
    exp = math.e**(-(1/2.0)*dot((dot(matrix(x-mu),inv(sigma))),matrix(x-mu).T).item())
    N = coef * exp
    return N

def makeColor(labels):
    colors=[]
    for l in labels:
        if l==0:
            colors.append('r')
        elif l==1:
            colors.append('g')
        elif l==2:
            colors.append('b')
    return colors


def calcPrecision(target,results):
    success = 0
    fail = 0
    for i in range(len(target)):
        if target[i]==results[i]:
            success = success + 1
        else:
            fail = fail+1
    return success/float(success+fail)*100

"""initialize"""
gamma = np.zeros((len(data),K))
# for i in range(len(gamma)):
#     problist = [3.5/10.0, 3.4/10.0, 3.1/10.0]
#     rindex = int(math.floor(random.random()*3))
#     g0 = problist.pop(rindex)
#     rindex = int(math.floor(random.random()*2))
#     g1 = problist.pop(rindex)
#     g2 = problist[0] 
#     gamma[i][0] = g0
#     gamma[i][1] = g1
#     gamma[i][2] = g2
for i in range(len(gamma)):
    true_label = target[i]

    luck = random.random()
    if (luck<0.3):
        gamma[i][0] = 1/10.0 
        gamma[i][1] = 1/10.0 
        gamma[i][2] = 1/10.0 

        gamma[i][true_label] = 8/10.0
    else:
        nums = [0,1,2]
        idx1 = int(math.floor(random.random()*3))
        rand_idx1 = nums.pop(idx1)
        gamma[i][rand_idx1] = 8/10.0
        idx2 = int(math.floor(random.random()*2))
        rand_idx2 = nums.pop(idx2)
        gamma[i][rand_idx2] = 1/10.0
        rand_idx3 = nums[0]
        gamma[i][rand_idx3] = 1/10.0

pi = np.array([1/3.0,1/3.0,1/3.0]) #混合比

labels = np.array([g.index(max(g)) for g in gamma.tolist()])
initial_labels = np.array([g.index(max(g)) for g in gamma.tolist()])

idx_0 = np.where(labels==0)[0]
x_0 = data[idx_0,:]
mu_0 = sum(x_0) / len(x_0)
mu_0s = repmat(mu_0,len(x_0),1)
S0 = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        S0[i,j] = sum((x_0[:,i]-mu_0s[:,i])*(x_0[:,j]-mu_0s[:,j]))/len(x_0)

idx_1 = np.where(labels==1)[0]
x_1 = data[idx_1,:]
mu_1 = sum(x_1) / len(x_1)
mu_1s = repmat(mu_1,len(x_1),1)
S1 = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        S1[i,j] = sum((x_1[:,i]-mu_1s[:,i])*(x_1[:,j]-mu_1s[:,j]))/len(x_1)

idx_2 = np.where(labels==2)[0]
x_2 = data[idx_2,:]
mu_2 = sum(x_2) / len(x_2)
mu_2s = repmat(mu_2,len(x_2),1)
S2 = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        S2[i,j] = sum((x_2[:,i]-mu_2s[:,i])*(x_2[:,j]-mu_2s[:,j]))/len(x_2)

MAX_LOOP = 100000

success_array=[]

"""EM algorithm"""
for epoch in range(MAX_LOOP):
    """E-step"""
    for i in range(len(data)):
        xi = data[i,:]
        Gauss0 = Gaussian(xi,mu_0,S0)
        Gauss1 = Gaussian(xi,mu_1,S1)
        Gauss2 = Gaussian(xi,mu_2,S2)
        gaussian_array = np.array([Gauss0,Gauss1,Gauss2])
        for k in range(K):
            gamma[i][k] = (pi[k]*gaussian_array[k])/sum(pi*gaussian_array)
    
    """M-step"""
    N0 = sum(gamma[:,0]).item()
    N1 = sum(gamma[:,1]).item()
    N2 = sum(gamma[:,2]).item()
    
    mu_0  = 1/N0 * sum([gamma[i,0].item()*data[i,:] for i in range(len(data))])
    mu_1  = 1/N1 * sum([gamma[i,1].item()*data[i,:] for i in range(len(data))])
    mu_2  = 1/N2 * sum([gamma[i,2].item()*data[i,:] for i in range(len(data))])
    mu = np.array([mu_0,mu_1,mu_2])
    
    
    S0 = np.zeros((d,d))
    for i in range(len(data)):
        delta = gamma[i,0]*dot(matrix(data[i,:]-mu_0).T,matrix(data[i,:]-mu_0))/N0 
        S0 =  S0 + delta
    
    S1 = np.zeros((d,d))
    for i in range(len(data)):
        delta = gamma[i,1]*dot(matrix(data[i,:]-mu_1).T,matrix(data[i,:]-mu_1))/N1 
        S1 =  S1 + delta
    
    S2 = np.zeros((d,d))
    for i in range(len(data)):
        delta = gamma[i,2]*dot(matrix(data[i,:]-mu_2).T,matrix(data[i,:]-mu_2))/N2 
        S2 =  S2 + delta
    
    S = [S0,S1,S2]
    
    N = N0+N1+N2
    pi_0 = N0/N
    pi_1 = N1/N
    pi_2 = N2/N
    pi = np.array([pi_0,pi_1,pi_2])
    
    zs = np.zeros((len(data),K))
    for i in range(len(gamma)):
        g = gamma[i]
        k = np.where(g==max(g))[0].tolist()[0]
        if(labels[i] != k):
            print "#", #->update label
        labels[i] = k
        zs[i][k] = 1 
    
    L = 0
    for i in range(len(zs)):
        for k in range(K):
            L = L + zs[i][k]*math.log(pi[k]) + zs[i][k]*(-d/2*math.log(2*math.pi) -1/2.0*math.log(det(S[k])) -1/2.0*(dot(dot(matrix(data[i,k]-mu[k]),inv(S[k])),matrix(data[i,k]-mu[k]).T)))

    if(abs(L-L_last)<math.e**-19):
        print "converged"
        break
    
    L_last = L.item()

    rate = calcPrecision(target,labels)
    print "Likelihood:",L.item(),"rate:",rate
    
    success_array.append([epoch,L,rate])
    epoch = epoch + 1
   
print "initialized data",initial_labels
print "computed result",labels
print "correct data",target

# plt.subplot2grid((2,2),(0,0),rowspan=2)
plt.subplot(221)
plt.scatter(xs,ys,plot_s,makeColor(initial_labels))
plt.title('Initialized sample')

plt.subplot(223)
plt.scatter(xs,ys,plot_s,makeColor(labels))
plt.scatter(xs,ys,plot_s,makeColor(labels))
plt.title('Classification using EM algorithm')

res = np.array(success_array)
ts = res[:,0]
Ls = res[:,1]
Rates = res[:,2]

# plt.subplot2grid((2,2),(0,1))
plt.subplot(222)
plt.plot(ts,Ls,'b-')
plt.title('Likelihood')

# plt.subplot2grid((2,2),(1,1))
plt.subplot(224)
plt.plot(ts,Rates,'r-')
plt.title('Precision')


plt.show()


