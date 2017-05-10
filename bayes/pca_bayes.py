#coding:utf-8
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pdb
import math

iris = datasets.load_iris()

data = iris.data
target = iris.target

mu = sum(data)/len(data)
mus = np.ones((len(data),1))*mu

S = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        s = sum((data[:,i] - mus[:,i])*(data[:,j]-mus[:,j]))
        cov = s/len(data)
        S[i,j] = cov 

T=np.matrix(np.linalg.eig(S)[1])
T1=T[:,0]
T2=T[:,1]

p1 = np.zeros((len(data),1))
for i in range(len(data)):
    p1[i] = np.dot(np.asarray(T1.T),data[i,:])

p2 = np.zeros((len(data),1))
for i in range(len(data)):
    p2[i] = np.dot(np.asarray(T2.T),data[i,:])

pdata = np.concatenate((p1,p2),axis=1)

Pc1 = 1/3.0 # a priori probability
Pc2 = 1/3.0
Pc3 = 1/3.0

c1_indices = np.where(target==0)[0].tolist()
c2_indices = np.where(target==1)[0].tolist()
c3_indices = np.where(target==2)[0].tolist()

c1=[]
c2=[]
c3=[]
for idx in c1_indices:
    c1.append(np.array([p1[idx].item(),p2[idx].item()]))
for idx in c2_indices:
    c2.append(np.array([p1[idx].item(),p2[idx].item()]))
for idx in c3_indices:
    c3.append(np.array([p1[idx].item(),p2[idx].item()]))

c1a = np.array(c1)
c2a = np.array(c2)
c3a = np.array(c3)

mu1 = sum(c1) /len(c1)
mu2 = sum(c2) /len(c2)
mu3 = sum(c3) /len(c3)
mu1s = mu1*np.ones((len(c1a),1))
mu2s = mu2*np.ones((len(c2a),1))
mu3s = mu3*np.ones((len(c3a),1))

S1 = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        s = sum((c1a[:,i] - mu1s[:,i])*(c1a[:,j]-mu1s[:,j]))
        cov = s/len(data)
        S1[i,j] = cov 

S2 = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        s = sum((c2a[:,i] - mu2s[:,i])*(c2a[:,j]-mu2s[:,j]))
        cov = s/len(data)
        S2[i,j] = cov 

S3 = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        s = sum((c3a[:,i] - mu3s[:,i])*(c3a[:,j]-mu3s[:,j]))
        cov = s/len(data)
        S3[i,j] = cov 

def g1(x):
    A=np.matrix(x).T-np.matrix(mu1).T
    B=np.linalg.inv(S1)
    C=A.T
    detS1 = np.linalg.det(S1) 
    result = np.dot(np.dot(C,B),A) + np.log(detS1) - 2*math.log(Pc1)
    return result.item()
def g2(x):
    A=np.matrix(x).T-np.matrix(mu2).T
    B=np.linalg.inv(S2)
    C=A.T
    detS2 = np.linalg.det(S2) 
    result = np.dot(np.dot(C,B),A) + np.log(detS2) - 2*math.log(Pc2)
    return result.item()
def g3(x):
    A=np.matrix(x).T-np.matrix(mu3).T
    B=np.linalg.inv(S3)
    C=A.T
    detS3 = np.linalg.det(S3) 
    result = np.dot(np.dot(C,B),A) + np.log(detS3) - 2*math.log(Pc3)
    return result.item()

def classify(x):
    map1inv = g1(x)
    map2inv = g2(x)
    map3inv = g3(x)
    mapinvs = np.array([map1inv,map2inv,map3inv])
    result = np.where(mapinvs==min(mapinvs))[0].tolist()[0]
    return result

results = []
for d in pdata:
    res = classify(d)
    results.append(res) 

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

precision = calcPrecision(target,results)
print 'success rate: ',precision,'%'

plt.subplot(121)
xs = data[:,0]
ys = data[:,2]
s = [30 for i in range(len(xs))]
colors = makeColor(target)
plt.scatter(xs,ys,s,colors)
plt.title('original data')

plt.subplot(122)
colors = makeColor(results)
plt.scatter(xs,ys,s,colors)
plt.title('Bayes classifier with normal distribution')

plt.show()
