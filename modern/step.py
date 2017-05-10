import numpy as np
from pylab import *
from scipy.linalg import expm
from scipy import integrate

c1 = 1

ts = np.linspace(0,10,100)
A = np.matrix([[0,1],[-4,-2]])
B = np.matrix([0,1]).transpose()
u = 1

xs=[]
vs=[]
x0 = np.matrix([0,0]).transpose()
def particular_solution(t):
    return expm(A*t-t)*B*u
    
for t in ts:
    xt = (1-expm(A*t))*x0 + integrate.dblquad(particular_solution(t),0,t)
    x = xt[0].item()
    v = xt[1].item()
    xs.append(x)
    vs.append(v)

print len(ts),len(xs)
plot(ts,xs)
show()

