import cv2
import numpy as np
from pylab import *

image = cv2.imread('chip.png',cv2.CV_8UC1)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 0)
se = np.ones((7,7), dtype='uint8')
image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
cnt = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
mask = np.zeros(image.shape[:2], np.uint8)


cntr=0
for cs in cnt:
    print len(cs)
    xs=[]
    ys=[]
    for c in cs:
        if(cntr==4):
            x = c[:,0][0]
            y = c[:,1][0]
            xs.append(x)
            ys.append(y)
            print x,y
            plot(x,y,'ro')
    cntr = cntr+1

imshow(image,'gray')
show()


# cv2.drawContours(mask, cnt, -1, 255, -1)
# cv2.imshow("Keypoints", mask)
# cv2.waitKey(0)
