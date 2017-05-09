import numpy as np
from pylab import *
import math

ks = [1.0,2.0,3.0,4.0,5.0]
colors = ['b','g','r','c','m']
xs = np.linspace(0,8,50)
counter = 0
for k in ks:
    print counter
    fs = []
    c = colors[counter]
    for x in xs:
        f = ((0.5)**(k/2))/(math.gamma(k/2))*x**(k/2-1)*math.e**(-x/2)
        fs.append(f)
    plot(xs,fs,color=c)
    counter = counter+1

stringArray = []
for k in ks:
    stringArray.append('k={0}'.format(k))
legend(stringArray)
title('Chi-squared distribution')
show()


