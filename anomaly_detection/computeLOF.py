import numpy as np
from sklearn import datasets

irisdata = datasets.load_iris().data
xs = irisdata[:,0]
ys = irisdata[:,2]


def computeLRD(k,coord):
    pass

def computeLOF(k,coord):
    LRD = computeLRD(k,coord)
    sumLRD = 
    LOF = (1/k)*sumLRD/LRD


for i in len(xs):
    x = xs[i]
    y = ys[i]
    LOF1 = computeLOF(1,(x,y))
    LOF2 = computeLOF(2,(x,y))
    LOF3 = computeLOF(3,(x,y))
    LOF4 = computeLOF(4,(x,y))
    LOF5 = computeLOF(5,(x,y))







