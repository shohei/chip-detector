from pylab import *
import imutils
import cv2

image = cv2.imread('chip.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
thresh = 250 
im_bw = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.bitwise_not(im_bw)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for c in cnts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("Image", image)

cv2.waitKey(0)
