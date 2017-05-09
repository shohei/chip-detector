import numpy as np
from pylab import *
import math

vs = [1.0,2.0,5.0,100.0]
colors = ['b','g','r','c']
xs = np.linspace(-5,5,100)
counter = 0
for v in vs:
    print counter
    fs = []
    c = colors[counter]
    for x in xs:
        fs.append(math.gamma((v+1)/2)/(math.sqrt(v*math.pi)*math.gamma(v/2))*((1+x**2/v)**(-(v+1)/2)))
    plot(xs,fs,color=c)
    counter = counter+1

stringArray = []
for v in vs:
    stringArray.append('v={0}'.format(v))
legend(stringArray)
title('Student\'s t distribution')
show()


