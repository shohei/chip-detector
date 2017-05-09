import numpy as np
from pylab import *
import math

def beta(a,b):
    beta = math.gamma(a)*math.gamma(b)/math.gamma(a+b)
    return beta

ds = [(1.0,4.0),(6.0,28.0),(28.0,6.0)]
colors = ['b','g','r']
xs = np.linspace(0,4,50)
counter = 0
for d in ds:
    print counter
    fs = []
    c = colors[counter]
    for x in xs:
        d1 = d[0]
        d2 = d[1]
        f = 1/(beta(d1/2,d2/2))*(d1*x/(d1*x+d2))**(d1/2)*(1-(d1*x/(d1*x+d2)))**(d2/2)*(x**(-1))
        fs.append(f)
    plot(xs,fs,color=c)
    counter = counter+1

stringArray = []
for d in ds:
    stringArray.append('(d1,d2)={0}'.format(d))
legend(stringArray)
title('F distribution')
show()


