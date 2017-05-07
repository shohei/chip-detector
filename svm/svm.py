import numpy as np
from numpy.random import *
from pylab import *
import cvxopt
import pdb

c1 = (1,1)
c2 = (5,5)

c1x = c1[0]
c1y = c1[1]
c2x = c2[0]
c2y = c2[1]

xs = []
ys = []
ts = []

for i in range(20):
    xs.append(normal(c1x,1))
    ys.append(normal(c1y,1))
    ts.append(-1)
    
for i in range(20):
    xs.append(normal(c2x,1))
    ys.append(normal(c2y,1))
    ts.append(1)

def label_to_color(ts):
    cs = []
    for l in ts:
        if l==-1:
            cs.append('r')
        elif l==1:
            cs.append('b')
    return cs

Q = np.zeros((40,40))
for i in range(40):
    for j in range(40):
        xi = np.matrix([xs[i],ys[i]]).transpose() 
        xj = np.matrix([xs[j],ys[j]]).transpose()
        Q[i,j] = 0.5*ts[i]*ts[j]*xi.transpose()*xj


A = np.zeros((2,40))
for j in range(40):
    A[0,j] = ts[j]
    A[1,j] = -ts[j] 
c = np.ones((40,1))*(-1)
b = np.matrix([0,0]).transpose()
O = np.ones((2,2))

A_ = np.concatenate((np.concatenate((A,O),axis=1),np.concatenate((Q,A.transpose()),axis=1)),axis=0)
b_ = np.concatenate((b,c),axis=0)
# print "A_",A_
# print "b_",b_

x =  np.linalg.solve(A_,b_)
lmd = x[-1]
alpha = x[1:-2]
print alpha
print lmd

# P=cvxopt.matrix(P)
# q=cvxopt.matrix((np.ones(40)*-1).transpose())
# print P
# print q
# print np.ndim(P)
# print np.ndim(q)
# sol=cvxopt.solvers.qp(P,q)
#
# print sol

"""
subplot(211)
colors = label_to_color(labels)
scatter(xs,ys,c=colors)
title('dataset')

subplot(212)
colors = label_to_color(labels)
scatter(xs,ys,c=colors)
title('SVM')

show()
"""


