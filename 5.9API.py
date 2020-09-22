# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:16:37 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:11:19 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:29:28 2020

@author: zhyf
E-mail:zhyfwcy@gmail.com

说明：img 数组格式
     image RGB格式

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Thread
from jiaozheng_img import jiaozheng
from load_img_url import load_img
from PIL import Image
from flask import Flask,request
from flask import jsonify
import cv2 as cv
import os
#from LW import decode_LW_code
from yolo_Model import YOLO
#import shutil
import numpy as np
from biaoJi import bianji

#import app_login as login
import upload as up
#import file_upload as fup


app = Flask(__name__)#创建一个服务，赋值给APP
def createfile(savefilepath):
    if not os.path.exists(savefilepath):
        os.makedirs(savefilepath)
#    elif os.path.exists(savefilepath):
#        shutil.rmtree(savefilepath)
#        os.makedirs(savefilepath)

class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

def Recognition_process(url):
    print("HELLO")
    sign,Orig_img=load_img(url)
    if(sign!=1):
        
        full_json={
                "code": 1,
                "msg":"there is a problem with the URL link, the photo can't be downloaded",
                "success":False
                   }
        
#        print(full_json)
        return jsonify(full_json)
    else:
        Orig_img=Orig_img
        photoName=url.split("/")[-1]
        print(photoName)
        judge_Sign1,LW_img,cropped_img,rotate_angle= yolo.detect_image(photoName,Orig_img)
        cropped_img=jiaozheng(cropped_img,rotate_angle)
        if(judge_Sign1==0):
#judge_Sign2=1 成功数据
                judge_Sign2,R_img= yolo.detect_imageback(cropped_img)#此处存在问题，需要优化
                if(judge_Sign2==7):
                    print('未知错误')
                    full_json={ "code": 7,
                                 "msg":"Unknown error ",
                                 "success":False
                               }
                    print(full_json)
                    return full_json
                elif(judge_Sign2==6):
                    print('猪卡精度不够')
                    full_json={ "code": 6,
                                 "msg":"The precision of cut photos is not enough ",
                                 "success":False
                               }
                elif(judge_Sign2==5):
                    print('猪卡被遮挡')
                    full_json={ "code": 5,
                                 "msg":"Pig Card Blocked",
                                 "success":False
                               }
                    print(full_json)
                    return full_json
                elif(judge_Sign2==4):
                    print('多张猪卡')
                    full_json={ "code": 4,
                                 "msg":"Multiple Pig Cards ",
                                 "success":False
                               }
                elif(judge_Sign2==3):
                    print('猪卡被遮挡')
                    full_json={ "code": 3,
                                 "msg":" NO Pig Card ",
                                 "success":False
                               }
                    print(full_json)
                    return full_json
                elif(judge_Sign2==2):
                    print('猪卡被遮挡')
                    full_json={ "code": 2,
                                 "msg":"NO ＬＷ",
                                 "success":False
                               }
                    print(full_json)
                    return full_json
                elif(judge_Sign2==0):
                    image = Image.fromarray(cv.cvtColor(R_img,cv.COLOR_BGR2RGB))
#需不需要矫正到这么大的尺寸
                    
                    image=image.resize((1200, 900), Image.BILINEAR)
                    image =cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
                    
                    
                    B_J_img='./Result/'
                    createfile(B_J_img)
                    tbj=time.time()
                    biaoji_img,info_Card,QR=bianji(image)
                    time_url = time.time() - tbj
                    print('图像标记时间',time_url)
                    ResultName=B_J_img+'result.jpg'
                    cv.imwrite(ResultName,biaoji_img)

#                    cv.imwrite('./Result/result.jpg',biaoji_img)
                    tmp='./Result/result.jpg'
                    tmp='http://139.9.114.227/display/'+photoName
                    up.upload(photoName)
                card_key=[
                             'X','begin-ordinal_1','end-ordinal_1','X','begin-ordinal_2','end-ordinal_2','X','begin-ordinal_3','end-ordinal_3','X','begin-ordinal_4','end-ordinal_4','X','begin-ordinal_5','end-ordinal_5',
                              'X','begin-ordinal_6','end-ordinal_6','birth_number-1','quantity2-10','quantity3-10','quantity6-10','quantity4-10','quantity5-10','quantity7-10','birth_weight-10','birth_weight-4','wean_quantity-10',
                              'total_weight-10','total_weight-1','birth_number-2','quantity2-1','quantity3-1','quantity6-1','quantity4-1','quantity5-1','quantity7-1','birth_weight-20','birth_weight-5','wean_quantity-1','total_weight-20',
                              'total_weight-2','birth_number-3','quantity2-2','quantity3-2','quantity6-2','quantity4-2','quantity5-2','quantity7-2','birth_weight-30','birth_weight-6','wean_quantity-2','total_weight-30','total_weight-3',
                              'birth_number-4','quantity2-3','quantity3-3','quantity6-3','quantity4-3','quantity5-3','quantity7-3','birth_weight-1','birth_weight-7','wean_quantity-3','total_weight-40','total_weight-4','birth_number-5',
                              'quantity2-4','quantity3-4','quantity6-4','quantity4-4','quantity5-4','quantity7-4','birth_weight-2','birth_weight-8','wean_quantity-4','total_weight-50','total_weight-5','vaccines1-muzhu_1','vaccines2-muzhu_1',
                              'vaccines3-muzhu_1','vaccines4-muzhu_1','vaccines5-muzhu_1','vaccines6-muzhu_1','vaccines7-muzhu_1','vaccines8-muzhu_1','vaccines9-muzhu_1','vaccines10-muzhu_1','vaccines1-muzhu_2','vaccines2-muzhu_2',
                              'vaccines3-muzhu_2','vaccines4-muzhu_2','vaccines5-muzhu_2','vaccines6-muzhu_2','vaccines7-muzhu_2','vaccines8-muzhu_2','vaccines9-muzhu_2','vaccines10-muzhu_2','vaccines1-muzhu_3','vaccines2-muzhu_3',
                              'vaccines3-muzhu_3','vaccines4-muzhu_3','vaccines5-muzhu_3','vaccines6-muzhu_3','vaccines7-muzhu_3','vaccines8-muzhu_3','vaccines9-muzhu_3','vaccines10-muzhu_3','vaccines1-zizhu_1','vaccines2-zizhu_1',
                              'vaccines3-zizhu_1','vaccines4-zizhu_1','vaccines5-zizhu_1','vaccines6-zizhu_1','vaccines7-zizhu_1','vaccines8-zizhu_1','vaccines9-zizhu_1','vaccines10-zizhu_1','thickness-before_mating','hickness-gestation_30_day',
                              'thickness-gestation_75_days','thickness-gestation_90_days','thickness-before_obstetric','thickness-20','thickness-10','thickness-9','thickness-8','thickness-7','thickness-6','thickness-5','thickness-4',
                              'thickness-3','thickness-2','SC-birth_code','ZC-birth_code','Y4-ordinal_1','Y3-ordinal_1','Y5-ordinal_1','Y4-ordinal_2','Y3-ordinal_2','Y5-ordinal_2','Y4-ordinal_3','Y3-ordinal_3','Y5-ordinal_3','Y4-ordinal_4',
                              'Y3-ordinal_4','Y5-ordinal_4','Y4-ordinal_5','Y3-ordinal_5','Y5-ordinal_5','Y4-ordinal_6','Y3-ordinal_6','Y5-ordinal_6','Y4-ordinal_7','Y3-ordinal_7','Y5-ordinal_7','Y4-ordinal_8','Y3-ordinal_8','Y5-ordinal_8',
                              'Y4-ordinal_9','Y3-ordinal_9','Y5-ordinal_9','Y4-ordinal_10','Y3-ordinal_10','Y5-ordinal_10','operation_name-A','operation_name-B','operation_name-C','operation_name-D','score_1-breeding_1','score_1-breeding_2',
                              'score_1-breeding_3','score_2-breeding_1','score_2-breeding_2','score_2-breeding_3','score_3-breeding_1','score_3-breeding_2','score_3-breeding_3'
                              ]
                msg_val=[]
                Boar_Code=str(QR[0])
                if(Boar_Code=='ERROR'):
                    print('无法识别公猪二维码')
                else:
                    QRCode_Boar={
                             "confidence": 0.99,
                             "info": Boar_Code,
                             "tag": "QRCodeBoar"
                            }
                    msg_val.append(QRCode_Boar)
                Sows_Code=str(QR[1])
                if(Sows_Code=='ERROR'):
                    print('无法识别母猪二维码')
                else:
                    QRCode_Sows={
                             "confidence": 0.99,
                             "info": Sows_Code,
                             "tag": "QRCodeSows"
                            }
                    msg_val.append(QRCode_Sows)
                card_info={}
                for i in range(len(info_Card)):
                        if(info_Card[i]!=0):
                            Key=card_key[i]
                            Val=str(info_Card[i])
                            if(Key=='X'):
                                print('卡片校验失败')
                                full_json={
                                   "code": -1,
                                    "msg":msg_val,
                                    "result_path":tmp,
                                    "success":True
                                  }
                                return full_json
                            else:
                                card_info[Key]=Val
                msg_card={
                            "confidence": 0.99,
                             "info": card_info,
                             "tag": "ManagementCardA"
                            }
                msg_val.append(msg_card)
                full_json={
                           "code": 0,
                            "msg":msg_val,
                            "result_path":tmp,
                            "success":True
                              }
                return full_json
#            print("没有栏位，请调节卡片位置或调节摄像头")
                
        elif(judge_Sign1==2):
            full_json={
                    "code": 2,
                    "msg":"No LW",
                    "success":False
                       }
            return full_json
            print("照片里没有猪卡和栏位信息")
        elif(judge_Sign1==3):
            full_json={
                    "code": 3,
                    "msg":"No Pig Card",
                    "success":False
                       }
            return full_json
            print("照片里没有猪卡和栏位信息")

        elif(judge_Sign1==4):
            full_json={
                    "code": 4,
                    "msg":"Multiple Pig Cards ",
                    "success":False
                       }
            return full_json
            print("照片里有多张猪卡")
        elif(judge_Sign1==5):
            full_json={
                    "code": 5,
                    "msg":"Pig Card Blocked ",
                    "success":False
                       }
            return full_json
            print("猪卡被遮挡")   
        elif(judge_Sign1==6):
            full_json={
                    "code": 6,
                    "msg":"The precision of cut photos is not enough ",
                    "success":False
                       }
            return full_json
            print("切割后照片精度不够")            
        elif(judge_Sign1==7):
            full_json={
                    "code": 7,
                    "msg":"Unknown error ",
                    "success":False
                       }
            return full_json
            print("未知错误")    
#        elif(judge_Sign1==1):
##                    return " ok"
##                LW=decode_LW_code(LW_img)
##judge_Sign2=1 成功数据
#                judge_Sign2,R_img= yolo.detect_imageback(cropped_img)#此处存在问题，需要优化
#                if(judge_Sign2==-1):
#                    print('猪卡未按要求放置')
#                    full_json={ "code": 7,
#                                 "msg":"Pig Card Was Blocked",
#                                 "success":False
#                               }
#                    print(full_json)
#                    return full_json
#                elif(judge_Sign2==1):
#                    image = Image.fromarray(cv.cvtColor(R_img,cv.COLOR_BGR2RGB))
##需不需要矫正到这么大的尺寸
#                    image=image.resize((1200, 900), Image.BILINEAR)
#                    image =cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
#                    B_J_img='./Result/'
#                    createfile(B_J_img)
#                    tbj=time.time()
#                    biaoji_img,info_Card,QR=bianji(image)
#                    time_url = time.time() - tbj
#                    print('图像标记时间',time_url)
##                    ResultName=B_J_img+'result.jpg'
#                    ResultName=B_J_img+photoName
##                    print('完成照片存储')
#                    cv.imwrite(ResultName,biaoji_img)
#                    tmp='./Result/result.jpg'
#
#                    tmp='http://139.9.114.227/display/'+photoName
#
#                    up.upload(photoName)
##                    print('****************************')
##                    pwd=login.login()
# #                   print('pwd',pwd)
##                    namemMsg=fup.up(B_J_img+photoName,pwd)
##                    print('namemMsg',namemMsg)
##                    tmp='/group1/iot/'+namemMsg
##                    print('****************************')
#
#
#
#                card_key=[
#                             'X','begin-ordinal_1','end-ordinal_1','X','begin-ordinal_2','end-ordinal_2','X','begin-ordinal_3','end-ordinal_3','X','begin-ordinal_4','end-ordinal_4','X','begin-ordinal_5','end-ordinal_5',
#                              'X','begin-ordinal_6','end-ordinal_6','birth_number-1','quantity2-10','quantity3-10','quantity6-10','quantity4-10','quantity5-10','quantity7-10','birth_weight-10','birth_weight-4','wean_quantity-10',
#                              'total_weight-10','total_weight-1','birth_number-2','quantity2-1','quantity3-1','quantity6-1','quantity4-1','quantity5-1','quantity7-1','birth_weight-20','birth_weight-5','wean_quantity-1','total_weight-20',
#                              'total_weight-2','birth_number-3','quantity2-2','quantity3-2','quantity6-2','quantity4-2','quantity5-2','quantity7-2','birth_weight-30','birth_weight-6','wean_quantity-2','total_weight-30','total_weight-3',
#                              'birth_number-4','quantity2-3','quantity3-3','quantity6-3','quantity4-3','quantity5-3','quantity7-3','birth_weight-1','birth_weight-7','wean_quantity-3','total_weight-40','total_weight-4','birth_number-5',
#                              'quantity2-4','quantity3-4','quantity6-4','quantity4-4','quantity5-4','quantity7-4','birth_weight-2','birth_weight-8','wean_quantity-4','total_weight-50','total_weight-5','vaccines1-muzhu_1','vaccines2-muzhu_1',
#                              'vaccines3-muzhu_1','vaccines4-muzhu_1','vaccines5-muzhu_1','vaccines6-muzhu_1','vaccines7-muzhu_1','vaccines8-muzhu_1','vaccines9-muzhu_1','vaccines10-muzhu_1','vaccines1-muzhu_2','vaccines2-muzhu_2',
#                              'vaccines3-muzhu_2','vaccines4-muzhu_2','vaccines5-muzhu_2','vaccines6-muzhu_2','vaccines7-muzhu_2','vaccines8-muzhu_2','vaccines9-muzhu_2','vaccines10-muzhu_2','vaccines1-muzhu_3','vaccines2-muzhu_3',
#                              'vaccines3-muzhu_3','vaccines4-muzhu_3','vaccines5-muzhu_3','vaccines6-muzhu_3','vaccines7-muzhu_3','vaccines8-muzhu_3','vaccines9-muzhu_3','vaccines10-muzhu_3','vaccines1-zizhu_1','vaccines2-zizhu_1',
#                              'vaccines3-zizhu_1','vaccines4-zizhu_1','vaccines5-zizhu_1','vaccines6-zizhu_1','vaccines7-zizhu_1','vaccines8-zizhu_1','vaccines9-zizhu_1','vaccines10-zizhu_1','thickness-before_mating','hickness-gestation_30_day',
#                              'thickness-gestation_75_days','thickness-gestation_90_days','thickness-before_obstetric','thickness-20','thickness-10','thickness-9','thickness-8','thickness-7','thickness-6','thickness-5','thickness-4',
#                              'thickness-3','thickness-2','SC-birth_code','ZC-birth_code','Y4-ordinal_1','Y3-ordinal_1','Y5-ordinal_1','Y4-ordinal_2','Y3-ordinal_2','Y5-ordinal_2','Y4-ordinal_3','Y3-ordinal_3','Y5-ordinal_3','Y4-ordinal_4',
#                              'Y3-ordinal_4','Y5-ordinal_4','Y4-ordinal_5','Y3-ordinal_5','Y5-ordinal_5','Y4-ordinal_6','Y3-ordinal_6','Y5-ordinal_6','Y4-ordinal_7','Y3-ordinal_7','Y5-ordinal_7','Y4-ordinal_8','Y3-ordinal_8','Y5-ordinal_8',
#                              'Y4-ordinal_9','Y3-ordinal_9','Y5-ordinal_9','Y4-ordinal_10','Y3-ordinal_10','Y5-ordinal_10','operation_name-A','operation_name-B','operation_name-C','operation_name-D','score_1-breeding_1','score_1-breeding_2',
#                              'score_1-breeding_3','score_2-breeding_1','score_2-breeding_2','score_2-breeding_3','score_3-breeding_1','score_3-breeding_2','score_3-breeding_3'
#                              ]
#                msg_val=[]
##                LW_Code=str(LW)
##                if(LW_Code=='ERROR'):
##                    print('无法识别栏位号')
##                else:
##                    QR_LW={
##                            "confidence": 0.99,
##                             "info": LW_Code,
##                             "tag": "QRCodeColumn"
##                           }
###                    msg_val.append(QR_LW)
##                Boar_Code=str(QR[0])
#                if(Boar_Code=='ERROR'):
#                    print('无法识别公猪二维码')
#                else:
#                    QRCode_Boar={
#                             "confidence": 0.99,
#                             "info": Boar_Code,
#                             "tag": "QRCodeBoar"
#                            }
#                    msg_val.append(QRCode_Boar)
#
#                Sows_Code=str(QR[1])
#
#                if(Sows_Code=='ERROR'):
#                    print('无法识别母猪二维码')
#                else:
#                    QRCode_Sows={
#                             "confidence": 0.99,
#                             "info": Sows_Code,
#                             "tag": "QRCodeSows"
#                            }
#                    msg_val.append(QRCode_Sows)
#
#                card_info={}
#                for i in range(len(info_Card)):
#                        if(info_Card[i]!=0):
#
#                            Key=card_key[i]
#                            Val=str(info_Card[i])
#                            if(Key=='X'):
#                                print('校验失败')
#                            else:
#                                card_info[Key]=Val
#
#                msg_card={
#                            "confidence": 0.99,
#                             "info": card_info,
#                             "tag": "ManagementCardA"
#                            }
#                msg_val.append(msg_card)
#
#                full_json={
#                           "code": 0,
#                            "msg":msg_val,
#                            "result_path":tmp,
#                            "success":True
#                              }
#                return full_json
            
@app.route('/api',methods=['get'])#指定接口访问的路径，支持什么请求方式get，post
#请求后直接拼接入参方式
def API():
    url = request.args.get('url')#使用request.args.get方式获取拼接的入参数据
    executor = ThreadPoolExecutor(max_workers=1000)
    all_task = [executor.submit(Recognition_process,  url )]
    for future in as_completed(all_task):
        data = future.result()
        print("返回的数据是{}".format(data))
        return jsonify(data)
if __name__ == '__main__':
     yolo = YOLO()
     urls=[]
     app.run(
       host='0.0.0.0',
       port= 9988,
       )




