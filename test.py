#이미지 파일 이용해서 차선 검출

import cv2
import numpy as np
import scipy.optimize as optimization
img=cv2.imread("./video/img_line.jpg")
#grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Canny edge
min, max=(50,200)
edge=cv2.Canny(img_gray,min,max,apertureSize = 3)
cv2.imshow('Canny Edge',edge)

#Hough Transform
lines=cv2.HoughLines(edge,1,np.pi/180,100)
L1=[] #[a b]
L2=[] #[c]
for line in lines:
    r, theta = line[0]
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*r
    y0=b*r
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*a)
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*a)
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
    L1.append([a,b])
    L2.append([r])
#Find Vanishing Point
U = np.linalg.pinv(L1) #pseudo inverse
x,y=np.dot(U,L2)
cv2.line(img,(x,y),(x,y),(0,0,255),5)

#ROI
point=[]
a=img.shape[0]
p_a=[]
b=img.shape[0]
p_b=[]
for i in range(len(edge)):
    for j in range(len(edge[0])):
        if edge[i][j]!=0:
            point.append((i,j))
            #cv2.line(img, (j,i),(j,i), (0, 0, 255), 5)
            if i>img.shape[0]*3/4:
                if x-j<0 and abs(x-j)<abs(a):
                    a=x-j
                    p_a=(j,i)
                elif x-j>0 and abs(x-j)<abs(b):
                    b=x-j
                    p_b=(j,i)
#cv2.line(img,p_a,p_a,(0,0,255),5)
#cv2.line(img,p_b,p_b,(0,0,255),5)
#vertex
if (img.shape[0]-1-y)*(x-p_a[0])/(y-p_a[1])+x>=img.shape[1]:
    v1=[img.shape[1]-1,img.shape[0]-1]
else:
    v1=[(img.shape[0]-1-y)*(x-p_a[0])/(y-p_a[1])+x,img.shape[0]-1]
if (img.shape[0]-1-y)*(x-p_b[0])/(y-p_b[1])+x<0:
    v2=[0,img.shape[0]-1]
else:
    v2=[(img.shape[0]-1-y)*(x-p_b[0])/(y-p_b[1])+x,img.shape[0]-1]
v3=[x,y]
mask = np.zeros(img.shape, dtype=np.uint8)
roi_corners = np.array([v1,v2,v3])
channel_count = img.shape[2]
ignore_mask_color = (255,)*channel_count
cv2.fillPoly(mask, np.int32([roi_corners]), ignore_mask_color)
# from Masterfool: use cv2.fillConvexPoly if you know it's convex
# apply the mask
masked_image = cv2.bitwise_and(img, mask)
cv2.imshow('res',masked_image)
cv2.imshow('img',img)


cv2.waitKey(0)
cv2.destroyAllWindows()