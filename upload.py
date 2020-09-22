# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:10:44 2020

@author: zhyf
E-mail:zhyfwcy@gmail.com

说明：

"""

import paramiko
import os
def upload(filename):
#     name='result.jpg'
#     localsrc=os.path.join(os.getcwd(),r'Result/',name)
    
#     localsrc='./Result/'+filename
     localsrc='./Result/result.jpg'
     remotesrc=os.path.join("/data/showphoto/display/",filename)
     transport = paramiko.Transport(("139.9.114.227", 22))        # 获取Transport实例
     transport.connect(username="root", password="qw12qw12@1314")   # 建立连接
     sftp = paramiko.SFTPClient.from_transport(transport)
     sftp.put(localsrc,remotesrc)
     print('filename')
     transport.close()

