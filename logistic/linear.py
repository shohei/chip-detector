import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target

w = np.zeros(5)
# w = np.array([100,100,100,100,100])

def pi(w,x):
    return 

def f_w(w):
    f = 0
    for i in range(len(data)):
        x = data[i,:]
        t = target
        f += + (pi(w,x)-t)*x
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



