import numpy as np

def nabla_f(p):
    x = p[0].item()
    y = p[1].item()
    J = np.matrix([3*x**2-9*y,3*y**2-9*x]).T 
    return J

def Hesse(p):
    x = p[0].item()
    y = p[1].item() 
    H = np.matrix([[6*x,-9],[-9,6*y]])
    return H

x = np.matrix([3,2]).T
while True:
    df = nabla_f(x)
    H = Hesse(x)
    delta_x = np.linalg.solve(H,-df)
    if (np.linalg.norm(delta_x) < 0.001):
        break
    x = x + delta_x

print x

