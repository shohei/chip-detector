#coding:utf-8
from pylab import *
import cv2
from sklearn.cluster import KMeans

NUM_CLUSTER=7

im_gray = cv2.imread('chip.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
thresh = 250
im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

# imshow(im_bw,"gray")  # im_bw.shape (778, 854) 778行 854列
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


0を抜き出してやればいい
"""

ymax = im_bw.shape[0]
xmax = im_bw.shape[1]

target=[]
xs=[]
ys=[]

for y in range(ymax):
    for x in range(xmax):
        if im_bw[y,x]==0:
            ys.append(y)
            xs.append(x)
            target.append((x,y))

# print target
# plot(xs,ys,'ro')
# show()

kmeans = KMeans(n_clusters=7).fit(target)
plot(xs,ys,'ro')
# imshow(im_bw,'gray')
print kmeans.cluster_centers_
cx = kmeans.cluster_centers_[:,0]
cy = kmeans.cluster_centers_[:,1]
plot(cx,cy,'bo')

show()






