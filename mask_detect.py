#coding:utf-8
from pylab import *
import cv2
from sklearn.cluster import KMeans

NUM_CLUSTER=7

# im_gray = cv2.imread('chip.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
# thresh = 250
# im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

image = cv2.imread('chip.png',cv2.CV_8UC1)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 0)
se = np.ones((7,7), dtype='uint8')
image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
cnt = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
mask = np.zeros(image.shape[:2], np.uint8)
cv2.drawContours(mask, cnt, -1, 255, -1)

ymax = mask.shape[0]
xmax = mask.shape[1]

target=[]
xs=[]
ys=[]

for y in range(ymax):
    for x in range(xmax):
        if mask[y,x]==0:
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






