import numpy as np
from sklearn import datasets
from numpy import matrix as mat
import math
from numpy import dot

iris = datasets.load_iris()
data = iris.data
target = iris.target

def exp(x):
    return math.e**(x)

def pi(w,xi):
    return [math.e**(wk*xi) / sum(math.e**(w*x)) for wk in w]

def f_w(w,xi):
    f = 0
    for i in range(len(data)):
        x = data[i,:]
        t = np.array([0,0,0])
        t[target[i]]=1
        f +=  (pi(w,x)-t)*x
    return f

def fdash_w(wk,w,xi):
    fdash = 0
    for i in range(len(data)):
        A = exp(dot(mat(wk).T,xi)) 
        B = sum(dot(exp(mat(w).T,xi)))  
        fdash += (A*B - A**2) / B**2 * xi
    return fdash

MAX_LOOP = 1000
w = np.zeros(5)
for i in range(MAX_LOOP):
    xi = data[i,:]
    w_nxt = w - f_w(w,xi) / fdash_w(w)
    if(np.linalg.norm(w_nxt-w)<0.001):
        break
    w_nxt = w

print w



