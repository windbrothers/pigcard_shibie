# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:16:53 2020

@author: zhyf
E-mail:zhyfwcy@gmail.com

说明：

"""
from product_QRCode import decode_qr_code
from produce_data import shengchengshuju
import cv2 as cv

def bianji(image):
    ##    image = image.resize((1200, 900))
##    #外边框
##    cv.rectangle(image,(0,0),(1200,900),(255,0,0),2)
#    #1后备发情
#    cv.rectangle(image,(110,42),(730,168),(255,0,0),2)
#    #2二维码
#    cv.rectangle(image,(0,168),(730,438),(255,0,0),2)
#    #3胎次
#    cv.rectangle(image,(0,438),(730,675),(255,0,0),2)
#    #4免疫
#    cv.rectangle(image,(110,675),(750,900),(255,0,0),2)
#    #5背标
#    cv.rectangle(image,(780,675),(1090,850),(255,0,0),2)
#    #6顺产、助产
#    cv.rectangle(image,(910,600),(1195,675),(255,0,0),2)
#    #7返情、流产、空怀
#    cv.rectangle(image,(970,168),(1195,600),(255,0,0),2)
#    #8配种人
#    cv.rectangle(image,(730,42),(1075,168),(255,0,0),2)
#数据边框
#1.1
    ID=0
    cv.rectangle(image,(150,84),(730,168),(255,0,0),2)
    X=[]
    info=[]
    QR=[]
    w=int((730-150)/9)
#    print('1/***************************************')
    for i in range(9):
        x1=120+w*(i+1)
        X.append(x1)

    Y=105
    for i in range(2):
        y= Y+40*(i)
        for j in range(len(X)):
               y0=y-15
               y1=y+15
               x0=X[j]-25
               x1=X[j]+25
               cv.rectangle(image,(x0,y0),(x1,y1),(255,255,0),2)
               ID=ID+1
               info_array=shengchengshuju(ID,y0,y1,x0,x1,image)
               info.append(info_array)
#               print(info)
               x0=x1=y0=y1=0
#2.1
    cv.rectangle(image,(0,168),(365,438),(255,0,0),2)
    cv.rectangle(image,(365,168),(730,438),(255,0,0),2)
    X=[]
    w=int((730-0)/4)
#    print('2/***************************************')
    for i in range(2):
        x1=0+w*(2*(i+1)-1)
        X.append(x1)
    y=303
#    print(len(X))
    for j in range(len(X)):
       y0=y-120
       y1=y+120
       x0=X[j]-165
       x1=X[j]+165
       cv.rectangle(image,(x0,y0),(x1,y1),(255,255,0),2)
       ID=ID+1
       QR_array=decode_qr_code(ID,y0,y1,x0,x1,image)
       QR.append(QR_array)
       x0=x1=y0=y1=0
       

#2.3
    cv.rectangle(image,(0,478),(730,675),(255,0,0),2)
    X=[]
    w=int((730-0)/24)
#    print('3/***************************************')
    for i in range(12):
        x1=10+w*(2*(i+1)-1)
        X.append(x1)
    Y=498
    for i in range(5):
        y= Y+40*(i)
        for j in range(len(X)):
           y0=y-15
           y1=y+15
           x0=X[j]-25
           x1=X[j]+25
           cv.rectangle(image,(x0,y0),(x1,y1),(255,255,0),2)
           ID=ID+1
           info_array=shengchengshuju(ID,y0,y1,x0,x1,image)
           info.append(info_array)
#           print(info)
#           info=shengchengshuju(ID,y0,y1,x0,x1,image,info)
           x0=x1=y0=y1=0

#2.4
    cv.rectangle(image,(168,735),(750,900),(255,0,0),2)
    X=[]
    w=int((750-168)/10)
#    print('4/***************************************')
    for i in range(10):
        x1=143+w*(i+1)
        X.append(x1)
    Y=755
    for i in range(4):
        y= Y+40*(i)
        for j in range(len(X)):
           y0=y-15
           y1=y+15
           x0=X[j]-25
           x1=X[j]+25
           cv.rectangle(image,(x0,y0),(x1,y1),(255,255,0),2)
           ID=ID+1
           info_array=shengchengshuju(ID,y0,y1,x0,x1,image)
           info.append(info_array)
#           info=shengchengshuju(ID,y0,y1,x0,x1,image,info)
           x0=x1=y0=y1=0

#2.5
    cv.rectangle(image,(780,735),(1090,850),(255,0,0),2)
    X=[]
    w=int((1090-780)/5)
#    print('5/15***************************************')
    for i in range(5):

        x1=745+w*(i+1)
        X.append(x1)
#    print(X)
#    第一行
    Y=755

    for i in range(3):
        y= Y+40*(i)
        for j in range(len(X)):
           y0=y-15
           y1=y+15
           x0=X[j]-25
           x1=X[j]+25
           cv.rectangle(image,(x0,y0),(x1,y1),(255,255,0),2)
           ID=ID+1
           info_array=shengchengshuju(ID,y0,y1,x0,x1,image)
           info.append(info_array)
#           info=shengchengshuju(ID,y0,y1,x0,x1,image,info)
           x0=x1=y0=y1=0

#2.6
    cv.rectangle(image,(910,632),(1195,675),(255,0,0),2)
    X=[]
    w=int((1195-910)/2)
#    print('6/2***************************************')
    for i in range(2):
        x1=837+w*(i+1)
        X.append(x1)
    y=654

    for j in range(len(X)):

       y0=y-15
       y1=y+15
       x0=X[j]-65
       x1=X[j]+65
       cv.rectangle(image,(x0,y0),(x1,y1),(255,255,0),2)
       ID=ID+1
       info_array=shengchengshuju(ID,y0,y1,x0,x1,image)
       info.append(info_array)
#       info=shengchengshuju(ID,y0,y1,x0,x1,image,info)
       x0=x1=y0=y1=0


#2.7
    cv.rectangle(image,(970,210),(1195,600),(255,0,0),2)
    X=[]
    w=int((1195-970)/3)
#    print('7/30***************************************')
    for i in range(3):

        x1=935+w*(i+1)
        X.append(x1)
    Y=225

    for i in range(10):
        y= Y+40*(i)
        for j  in range(len(X)):
            y0=y-13
            y1=y+13
            x0=X[j]-34
            x1=X[j]+34
            cv.rectangle(image,(x0,y0),(x1,y1),(255,255,0),2)
            ID=ID+1
            info_array=shengchengshuju(ID,y0,y1,x0,x1,image)
            info.append(info_array)
#            info=shengchengshuju(ID,y0,y1,x0,x1,image,info)
            x0=x1=y0=y1=0

#2.8
    cv.rectangle(image,(730,84),(840,168),(255,0,0),2)
    X=[]
    w=int((840-730)/2)
#    print('8/13***************************************')
    for i in range(2):
        x1=705+w*(i+1)
        X.append(x1)
    Y=105
#    print(len(X))
    for i in range(2):
        y= Y+40*(i)
        for j  in range(len(X)):
            y0=y-13
            y1=y+13
            x0=X[j]-25
            x1=X[j]+25
            cv.rectangle(image,(x0,y0),(x1,y1),(255,255,0),2)
            ID=ID+1
            info_array=shengchengshuju(ID,y0,y1,x0,x1,image)
            info.append(info_array)
#            info=shengchengshuju(ID,y0,y1,x0,x1,image,info)
            x0=x1=y0=y1=0

    cv.rectangle(image,(870,42),(1080,168),(255,0,0),2)
    X=[]
    w=int((1080-870)/3)
    for i in range(3):
        x1=835+w*(i+1)
        X.append(x1)
    Y=65
#    print(len(X))
    for i in range(3):
        y= Y+40*(i)
        for j  in range(len(X)):
            y0=y-13
            y1=y+13
            x0=X[j]-25
            x1=X[j]+25
            cv.rectangle(image,(x0,y0),(x1,y1),(255,255,0),2)
            ID=ID+1
            info_array=shengchengshuju(ID,y0,y1,x0,x1,image)
            info.append(info_array)
#            info=shengchengshuju(ID,y0,y1,x0,x1,image,info)
            x0=x1=y0=y1=0

#    cv.imwrite(savefilepname,image)
    return image,info,QR