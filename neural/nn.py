# coding:utf-8
import numpy as np
from sklearn import datasets
import math

iris = datasets.load_iris()
data = iris.data
target = iris.target

I = 4 # Sepal width/height, Petal width/height
J = 4 # Arbitrary
K = 3 # Setosa, Virginica, Versicolor

eta = 0.01

def sigmoid(x):
    return 1/(1-math.e**(-beta*x))

def sigmoid_dash(x):
    return beta * sigmoid(x) * (1 - sigmoid(x))

def softmax(xk,x):
    return math.e**(xk) / sum(math.e**(x))

def softmax_dash(xi,xk,is_same_i_j):
    if is_same_i_j:
        return softmax(xi) * (1 - softmax(xi))
    else:
        return -softmax(xi) * softmax(xk)

# 出力層-隠れ層の計算
hk=[]
delta_k=[]
d_wkj=[]
for k in range(K):
    for j in range(J):
        hk[k] =  sum(wkj[k][j] * softmax(sum(wji[j][i] * xk[k]))) 
        delta_k[k] = (tk[k] - ok[k])*softmax_dash(hk[k]) 
        d_wkj[k][j] = eta * sum(delta_k[k]) * Vj[j] 

# 隠れ層-入力層を計算
hj=[]
delta_j=[]
d_wji=[]
for j in range(J):
    for i in range(I):
        hj[j] =  sum(wji * xi) 
        delta_j[j] =  softmax_dash(hj) * sum(delta_k * wkj) 
        d_wji[j][i] =  eta * sum(delta_j * xk) 




