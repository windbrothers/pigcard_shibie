# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:34:08 2020

@author: zhyf
E-mail:zhyfwcy@gmail.com

说明：

"""
import cv2 as cv
import numpy as np
import math
from scipy import  ndimage
def jiaozheng(img,rotate_angle):
    if(rotate_angle==0):
               
               return img

    else:
            rotate_angle = math.degrees(math.atan(rotate_angle))
            if rotate_angle > 45:
                rotate_angle = -90 + rotate_angle
            elif rotate_angle < -45:
                rotate_angle = 90 + rotate_angle
            rotate_img = ndimage.rotate(img, rotate_angle)
#            print('success')
            return rotate_img

