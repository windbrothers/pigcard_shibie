# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:29:28 2020

@author: zhyf
E-mail:zhyfwcy@gmail.com

说明：img 数组格式
     image RGB格式

"""

import time
#from timeit import default_timer as timer
from jiaozheng_img import jiaozheng   
from load_img_url import load_img
from PIL import Image
from flask import Flask,request
from flask import jsonify
import cv2 as cv
import os
from LW import decode_LW_code
from yolo_Model import YOLO
#import math
import numpy as np
from biaoJi import bianji
app = Flask(__name__)#创建一个服务，赋值给APP
def createfile(savefilepath):
    if not os.path.exists(savefilepath):
        os.makedirs(savefilepath)

#@app.route('/api',methods=['get'])#指定接口访问的路径，支持什么请求方式get，post
#请求后直接拼接入参方式
def API():
#step1 加载照片   
#    t=time.time()
#    url = request.args.get('url')#使用request.args.get方式获取拼接的入参数据
#    time_url = time.time() - t
#    print('url加载时间',time_url)
#    sign,Orig_img=load_img(url)
#    if(sign!=1):
#        full_json={
#                "code": 2,
#                "msg":"download pic failed",
#                "success":False
#                   }
#        return jsonify(full_json)
#    else:
#        Orig_img=Orig_img
#        photoName=url.split("/")[-1]
        
#sept1 测试本地照片       
#    filepath = './test/t1/'  
    filepath = './test/test/'      
#    filepath = './t/'     
    for filename in os.listdir(filepath):
        Orig_url = filepath+filename
        photoName=Orig_url.split("/")[-1]
        Orig_img = cv.imread(Orig_url)
     
#sept2 截取照片大致范围
        tdw=time.time()
#judge_Sign1=1  正常 
#judge_Sign1=0  没有栏位，现在阶段栏位在卡的上方
#judge_Sign1=-1  找不到卡 
#judge_Sign1=2  多张卡        
#此时策略：没有主卡就不找栏位   最好加入3个及4个点位的区分
        
        judge_Sign1,LW_img,cropped_img,rotate_angle= yolo.detect_image(photoName,Orig_img)
        cropped_img=jiaozheng(cropped_img,rotate_angle)
        if(judge_Sign1==0):
            print("没有栏位，请调节卡片位置或调节摄像头")
            Orig_img_save='./Result/Draw/org_'+photoName  
            cv.imwrite(Orig_img_save,Orig_img)     
            
        elif(judge_Sign1==-1):
            print("照片里没有猪卡和栏位信息")
        elif(judge_Sign1==2):
            print("照片里有多张猪卡")         
        elif(judge_Sign1==1):
            print("正常卡能切割卡")
            LW_save='./Result/LW/LW'+photoName
            cv.imwrite(LW_save,LW_img)    
            
            cropped_save='./Result/cropped/crop_'+photoName  
            cv.imwrite(cropped_save,cropped_img)    
        time_url = time.time() - tdw
        print('定位分割时间',time_url)

        if(judge_Sign1!=1):
                    full_json={
                            "code":0,
                             "msg":"数据存在问题",
                             "success":False
                              }
                    print(full_json)
        elif(judge_Sign1==1):
                LW=decode_LW_code(LW_img)
#judge_Sign2=1 成功数据                
                judge_Sign2,R_img= yolo.detect_imageback(cropped_img)#此处存在问题，需要优化
                if(judge_Sign2==-1):
                    print('猪卡未按要求放置')
                    full_json={ "code": 1,
                                 "msg":"猪卡未按要求放置",
                                 "success":False
                               }
                    print(full_json)
                    
                elif(judge_Sign2==1):
                    image = Image.fromarray(cv.cvtColor(R_img,cv.COLOR_BGR2RGB))
#需不需要矫正到这么大的尺寸
                    image=image.resize((1200, 900), Image.BILINEAR)
                    image =cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
                    B_J_img='./Result/biaoji/'
                    createfile(B_J_img)
                    tbj=time.time()
                    biaoji_img,info_Card,QR=bianji(image)
                    time_url = time.time() - tbj
                    print('图像标记时间',time_url)
                    ResultName=B_J_img+photoName
                    cv.imwrite(ResultName,biaoji_img)
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
                LW_Code=str(LW)
                if(LW_Code=='ERROR'):
                    print('无法识别栏位号')
                else:
                    QR_LW={
                            "confidence": 0.99999964237213135,
                             "info": LW_Code,
                             "tag": "LW"
                           }
                    msg_val.append(QR_LW)
                Boar_Code=str(QR[0])
                if(Boar_Code=='ERROR'):
                    print('无法识别公猪卡')
                else:
                    QRCode_Boar={
                             "confidence": 0.99999964237213135,
                             "info": Boar_Code,
                             "tag": "QRCode_Boar"
                            }  
                    msg_val.append(QRCode_Boar)
                    
                Sows_Code=str(QR[1])
                
                if(Sows_Code=='ERROR'):
                    print('无法识别母猪猪卡')
                else:
                    QRCode_Sows={
                             "confidence": 0.99999964237213135,
                             "info": Sows_Code,
                             "tag": "QRCode_Sows"
                            }  
                    msg_val.append(QRCode_Sows)                    

                card_info={}
                for i in range(len(info_Card)):
                        if(info_Card[i]!=0):
                            Key=card_key[i]
                            Val=str(info_Card[i])
                            card_info[Key]=Val
          
                msg_card={
                            "confidence": 0.99999964237213135,
                             "info": card_info,
                             "tag": "ManagementCardA"
                            }
                msg_val.append(msg_card)

                full_json={
                           "code": 0,
                            "msg":msg_val,
                            "result_path":'tmp',
                            "success":True
                              }

                print(full_json)
#                return jsonify(full_json)
##                    return 'ok'

if __name__ == '__main__':
     yolo = YOLO()
     API()
#     app.run(
#       host='0.0.0.0',
#       port= 9988,
##       debug=True
#       )

    


