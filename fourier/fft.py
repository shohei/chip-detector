# coding:utf-8
import numpy as np
from pylab import *
import pdb

f = np.matrix([0,1,1,2,3,2,3,2]).transpose()
plot(f)
show()

w8 = e**(pi/4*1j) #1の原始8乗根
w4 = e**(pi/2*1j) #1の原始4乗根
w2 = e**(pi*1j) #1の原始2乗根

a0 = f[0]
a1 = f[1]
a2 = f[2]
a3 = f[3]
a4 = f[4]
a5 = f[5]
a6 = f[6]
a7 = f[7]

FFT2u_0_p = a0+a4
FFT2u_1_p = a0-a4
FFT2u_0_q = a2+a6
FFT2u_1_q = a2-a6

FFT2d_0_p = a1+a5
FFT2d_1_p = a1-a5
FFT2d_0_q = a3+a7
FFT2d_1_q = a3-a7

FFT4_0_p = FFT2u_0_p + (w2**0)*FFT2u_0_q
FFT4_1_p = FFT2u_1_p + (w2**1)*FFT2u_1_q
FFT4_2_p = FFT2u_0_p - (w2**2)*FFT2u_0_q
FFT4_3_p = FFT2u_1_p - (w2**3)*FFT2u_1_q
FFT4_0_q = FFT2d_0_p + (w2**0)*FFT2d_0_q
FFT4_1_q = FFT2d_1_p + (w2**1)*FFT2d_1_q
FFT4_2_q = FFT2d_0_p - (w2**2)*FFT2d_0_q
FFT4_3_q = FFT2d_1_p - (w2**3)*FFT2d_1_q

FFT8_0 = FFT4_0_p + (w4**0)*FFT4_0_q
FFT8_1 = FFT4_1_p + (w4**0)*FFT4_1_q
FFT8_2 = FFT4_2_p + (w4**0)*FFT4_2_q
FFT8_3 = FFT4_3_p + (w4**0)*FFT4_3_q
FFT8_4 = FFT4_0_p - (w4**0)*FFT4_0_q
FFT8_5 = FFT4_1_p - (w4**0)*FFT4_1_q
FFT8_6 = FFT4_2_p - (w4**0)*FFT4_2_q
FFT8_7 = FFT4_3_p - (w4**0)*FFT4_3_q

bk = np.array([FFT8_0[0,0],FFT8_1[0,0],FFT8_2[0,0],FFT8_3[0,0],FFT8_4[0,0],FFT8_5[0,0],FFT8_6[0,0],FFT8_7[0,0]])
Fk = np.conj(bk) / 8.0
print Fk

plot(real(Fk),imag(Fk),'bo')
show()

P = Fk*np.conj(Fk)
plot(P)
show()

