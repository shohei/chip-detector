# coding:utf-8
import numpy as np
from sklearn import datasets
import math
import random
from numpy import matrix as mat
import pdb
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = iris.data
target = iris.target

I = 4+1 # Sepal width/height, Petal width/height (+ bias)
J = 4+1 # Arbitrary (+ bias)
K = 3 # Setosa, Virginica, Versicolor

eta = 0.05
beta = 1

def sigmoid(x): # x can be a vector
    return 1/(1-math.e**(-beta*x))

def sigmoid_dash(x):
    return beta * sigmoid(x) * (1 - sigmoid(x))

def softmax(xk,x):
    return math.e**(xk) / sum(math.e**(x))

def softmax_dash(xk,x):
    return softmax(xk,x) * (1 - softmax(xk,x))

wkj = np.zeros((K,J))
wji = np.zeros((J,I))
def initialize():
    for k in range(K):
        for j in range(J):
            wkj[k][j]  = random.random() 
    for j in range(J):
        for i in range(I):
            wji[j][i]  = random.random() 
initialize()

hj=np.zeros(J)
hk=np.zeros(K)
Vj=np.zeros(J)
ok=np.zeros(K)
def forward(xi):
    for j in range(J):
        hj[j] =  (np.dot(mat(wji[j,:]),mat(xi).T)).item() # scalar
        Vj[j] = sigmoid(hj[j])
    for k in range(K):
        hk[k] = (sum(np.dot(mat(wkj[k,:]),mat(Vj).T))).item()
    for k in range(K):
        ok[k] = softmax(hk[k].item(),hk)
    return ok
    
delta_k=np.zeros((K,1))
delta_j=np.zeros((J,1))
d_wkj=np.zeros((K,J)) #JからKへの重み
d_wji=np.zeros((J,I))
def backpropagation(ti):
    for k in range(K):
        delta_k[k] = (ti[k] - ok[k])*softmax_dash(hk[k],hk)
        for j in range(J):
            d_wkj[k][j] = eta * delta_k[k] * Vj[j]
            wkj[k][j] = wkj[k][j] + d_wkj[k][j]
    for j in range(J):
        delta_j[j] = sigmoid_dash(hj[j]) * np.dot(mat(delta_k).T,mat(wkj[:,j]).T)
        for i in range(I):
            d_wji[j][i] = eta * delta_j[j] * xi[i]
            wji[j][i] = wji[j][i] + d_wji[j][i]

def makeTarget(t):
    ti=[0,0,0]
    ti[t] = 1 # t=0,1,2
    return ti

# main routine for neural network
for i in range(len(data)):
    xi = np.concatenate((np.array([1]),data[i,:]))
    forward(xi)
    ti = makeTarget(target[i])
    backpropagation(ti)

# Testing result
labels=[]
for i in range(len(data)):
    test = np.concatenate((np.array([1]),data[i,:]))
    ok = forward(test)
    print ok
    label = np.where(np.array(ok)==max(ok))[0].tolist()[0]
    labels.append(label)

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

xs = data[:,0] # for scatter plot
ys = data[:,2]
plot_s = [30 for i in range(len(xs))] #scatter plot point size
print labels
plt.scatter(xs,ys,plot_s,makeColor(labels))
plt.title("Classification using neural network")
# plt.show()



