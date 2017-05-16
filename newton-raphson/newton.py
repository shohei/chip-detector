# coding:utf-8

def f_dash(x):
    return float(3*x**2 -4*x + 1)

def f_2dash(x):
    return float(6*x - 4)

x = 100 
while True:
    print x
    x_nxt = x - f_dash(x)/f_2dash(x)
    delta = abs(x_nxt-x)
    if (delta<0.000001):
        break
    x = x_nxt

print x
