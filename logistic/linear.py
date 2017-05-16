import numpy as np
from sklearn import datasets
from numpy import matrix as mat
import math

iris = datasets.load_iris()
data = iris.data
target = iris.target

w = np.zeros(5)
# w = np.array([100,100,100,100,100])

def pi(wk,w,xi,x):
    return math.e**(wk*xi) / sum(math.e**(w*x))

def f_w(w):
    f = 0

    for i in range(len(data)):
        x = data[i,:]
        t = np.array([0,0,0])
        t[target[i]]=1
        f +=  (pi(w,x)-t)*x
    return f

def fdash_w(w):
    fdash = 0

    return 

MAX_LOOP = 1000
for _ in range(MAX_LOOP):
    w_nxt = w - f_w(w) / fdash_w(w)
    if(np.linalg.norm(w_nxt-w)<0.001):
        break
    w_nxt = w

print w



