# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:42:43 2020

@author: zhyf
E-mail:zhyfwcy@gmail.com

说明：

"""
from PIL import Image
from pyzbar import pyzbar
def decode_LW_code(image):
    results= pyzbar.decode(Image.fromarray(image), symbols=[pyzbar.ZBarSymbol.QRCODE])
    if len(results):
        sign=results[0].data.decode("utf-8")
    else:
        sign='ERROR'
    return sign