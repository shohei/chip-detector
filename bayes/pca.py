#coding:utf-8
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pdb

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

cs = []
for l in target:
    if l==0:
        cs.append('r')
    elif l==1:
        cs.append('g')
    elif l==2:
        cs.append('b')

s = 30*np.ones((len(data),1))

plt.scatter(p1,p2,s,cs)
plt.xlabel(u'第1主成分p1')
plt.ylabel(u'第2主成分p2')

plt.show()

