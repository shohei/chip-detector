import numpy as np
from sklearn import datasets
import math
from pylab import *

iris = datasets.load_iris()
irisdata = iris.data
irislabel = iris.target

def calcDistance(point,target):
    xp=point[0]
    yp=point[1]
    xt=target[0]
    yt=target[1]
    dist = math.sqrt((xp-xt)**2+(yp-yt)**2)
    return dist

def computeLabel(k,min_j_arr):
    if k==1:
        label = irislabel[min_j_arr[0]]
    elif k==3:
        candidates = [irislabel[min_j_arr[0]],irislabel[min_j_arr[1]],irislabel[min_j_arr[2]]]
        label = np.bincount(candidates).argmax() # decide by majority
    elif k==5:
        candidates = [irislabel[min_j_arr[0]],irislabel[min_j_arr[1]],irislabel[min_j_arr[2]],irislabel[min_j_arr[3]],irislabel[min_j_arr[4]]]
        label= np.bincount(candidates).argmax() # decide by majority
    return label

def computeMinimumDistanceIndex(k,d,min_j_arr,min_d_arr,j):
    if k==1:
        if d < min_d_arr[0]:
            min_d_arr[0] = d
            min_j_arr[0] = j
    elif k==3:
        if d < min_d_arr[0]:
            min_d_arr[0] = d
            min_j_arr[0] = j
        elif d < min_d_arr[1]:
            min_d_arr[1] = d
            min_j_arr[1] = j
        elif d < min_d_arr[2]:
            min_d_arr[2] = d
            min_j_arr[2] = j
    elif k==5:
        if d < min_d_arr[0]:
            min_d_arr[0] = d
            min_j_arr[0] = j
        elif d < min_d_arr[1]:
            min_d_arr[1] = d
            min_j_arr[1] = j
        elif d < min_d_arr[2]:
            min_d_arr[2] = d
            min_j_arr[2] = j
        elif d < min_d_arr[3]:
            min_d_arr[3] = d
            min_j_arr[3] = j
        elif d < min_d_arr[4]:
            min_d_arr[4] = d
            min_j_arr[4] = j


def kNN(k,coords):
    trained = []
    xs = coords[:,0]
    ys = coords[:,2]
    for i in range(len(xs)):
        xp = xs[i]
        yp = ys[i]
        min_d_arr = []
        min_j_arr = []
        for i in range(k):
            min_d_arr.append(math.sqrt(xp**2 + yp**2)) # initialize array with arbitral large number
        for i in range(k):
            min_j_arr.append(0) # initialize array with zero 
        for j in range(len(xs)):
            if i==j:
                continue 
            xt = xs[j]
            yt = ys[j]
            d = calcDistance((xp,yp),(xt,yt))
            computeMinimumDistanceIndex(k,d,min_j_arr,min_d_arr,j) 

        label = computeLabel(k,min_j_arr)
        trained.append(label)
    return trained
    

def label_to_color(label_array):
    cs = []
    for l in label_array:
        if l==0:
            cs.append('r')
        elif l==1:
            cs.append('g')
        elif l==2:
            cs.append('b')
    return cs

def compute_score(label_train,label_trained):
    success = 0
    fail = 0
    for i in range(len(label_train)):
        if (label_train[i]==label_trained[i]):
            success = success + 1
        else:
            fail = fail + 1
    score = success / float(success+fail) * 100
    return score

trained1 = kNN(1,irisdata) 
trained3 = kNN(3,irisdata) 
trained5 = kNN(5,irisdata) 

score1 = compute_score(irislabel,trained1)
score3 = compute_score(irislabel,trained3)
score5 = compute_score(irislabel,trained5)

print "1-NN score: ",score1
print "3-NN score: ",score3
print "5-NN score: ",score5

t1= label_to_color(irislabel)
c1 = label_to_color(trained1)
c3 = label_to_color(trained3)
c5 = label_to_color(trained5)

xs = irisdata[:,0]
ys = irisdata[:,2]

subplot(221)
scatter(xs,ys,c=t1)
title('datasets')

subplot(222)
scatter(xs,ys,c=c1)
title('1-NN classifier')

subplot(223)
scatter(xs,ys,c=c3)
title('3-NN classifier')

subplot(224)
scatter(xs,ys,c=c5)
title('5-NN classifier')

show()

