# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:30:19 2020

@author: zhyf
E-mail:zhyfwcy@gmail.com

说明：

"""
import cv2 as cv
import numpy as np
import requests as req
from PIL import Image
from io import BytesIO
def load_img(image_path):
    try:
        response = req.get(image_path)
        image = Image.open(BytesIO(response.content))
        img = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
#        print(image)
        return 1,img
    except:
        img=[]
        return -1,img

