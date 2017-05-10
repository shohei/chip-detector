import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

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

S2 = np.cov(data)
# print "variance-covariance matrix",S
T=np.matrix(np.linalg.eig(S)[1])
D = T.I*S*T
T2=np.matrix(np.linalg.eig(S2)[1])
D2 = T2.I*S2*T2
print np.size(D2)

pdata = np.zeros((len(data),4))
for i in range(len(data)):
    pdata[i,:] = np.dot(T,np.matrix(data[i,:]).T).T

# print pdata
x0 = pdata[:,0]
x1 = pdata[:,1]
x2 = pdata[:,2]
x3 = pdata[:,3]

cs = []
for l in target:
    if l==0:
        cs.append('r')
    elif l==1:
        cs.append('g')
    elif l==2:
        cs.append('b')

s = 30*np.ones((len(data),1))

# plt.subplot(321)
# plt.scatter(x0,x1,s,cs)
# plt.subplot(322)
# plt.scatter(x0,x2,s,cs)
# plt.subplot(323)
# plt.scatter(x0,x3,s,cs)
# plt.subplot(324)
# plt.scatter(x1,x2,s,cs)
# plt.subplot(325)
# plt.scatter(x1,x3,s,cs)
# plt.subplot(326)
# plt.scatter(x2,x3,s,cs)
#
# plt.show()
#
