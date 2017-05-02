from pylab import *
import cv2

im_gray = cv2.imread('chip.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
thresh = 250
im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

# imshow(im_bw,"gray")  # im_bw.shape (778, 854)
# show()

"""
以下の配列を、(x,y)の配列に変換する
array([[255, 255, 255, ..., 255, 255, 255],
       [255, 255, 255, ..., 255, 255, 255],
       [255, 255, 255, ..., 255, 255, 255],
       ...,
       [255, 255, 255, ..., 255, 255, 255],
       [255, 255, 255, ..., 255, 255, 255],
       [255, 255, 255, ..., 255, 255, 255]], dtype=uint8)


thresholdは

"""



