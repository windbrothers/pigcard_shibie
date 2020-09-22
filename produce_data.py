# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 16:00:35 2020

@author: zhyf
E-mail:zhyfwcy@gmail.com

说明：

"""
import cv2 as cv
import numpy as np
def shengchengshuju(ID,y0,y1,x0,x1,image):
    y0=y0+5
    y1=y1-5
    x0=x0+5
    x1=x1-5
    ptStart = (x0, y0)
    ptEnd = (x1, y1)
    ptStart1 = (x0, y1)
    ptEnd1 = (x1, y0)


    cropped = image[y0:y1,x0:x1]
    hsv1 = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hsv = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
    low_hsv = np.array([0,153,35])
    high_hsv = np.array([180,255,255])
##    红色
#    low_hsv = np.array([0,43,46])
#    high_hsv = np.array([10,255,255])

    Mask = cv.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
    dilation = cv.dilate(Mask,np.ones((5,5),np.uint8),iterations = 1)
    Mask = cv.erode(dilation,np.ones((3,3),np.uint8),iterations = 1)
#
    Mask1 = cv.inRange(hsv1,lowerb=low_hsv,upperb=high_hsv)
    dilation = cv.dilate(Mask1,np.ones((5,5),np.uint8),iterations = 1)
    Mask1 = cv.erode(dilation,np.ones((3,3),np.uint8),iterations = 1)

    cv.imwrite('Mask1.jpg',Mask1)


    pint_W=Mask.shape[0]-1
    pint_H=Mask.shape[1]-1
#    print(ID,'长度及宽度',pint_W,pint_H)
    area=(pint_W+1)*(pint_H+1)
    Wk=0#Wk 白色
    Bk=0#Bk 黑色
    for i in range(pint_W):
        for j in range(pint_H):
            if(Mask[i][j]==255):
#                print(Mask[pint_W][pint_H])
                Wk=Wk+1
            elif(Mask[i][j]==0):
#                print(Mask[pint_W][pint_H])
                Bk=Bk+1
            else:
                print('error!')
#                print(Mask[pint_W][pint_H])

#    if not os.path.exists('./ceshijubu/'):
#        os.makedirs('./ceshijubu/')
#    num=str(ID)
#    juquname='./ceshijubu/'+num+'.jpg'
#    cv.imwrite(juquname,Mask)
    R=Wk/area
    if(R>0.1):
        cv.line(image, ptStart, ptEnd, (0, 255, 0), 1, 4)
        cv.line(image, ptStart1, ptEnd1, (0, 255, 0), 1, 4)
        sign=1
    else:
        sign=0
    cv.waitKey()
    return sign
