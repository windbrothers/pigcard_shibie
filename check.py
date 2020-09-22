# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:46:28 2020

@author: zhyf
E-mail:zhyfwcy@gmail.com

说明：

"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:06:07 2020

@author: zhyf
E-mail:zhyfwcy@gmail.com

说明：

"""
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:12:23 2020

@author: zhyf
"""

import colorsys
import os
#from timeit import default_timer as timer
import time
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2 as cv
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
import shutil
import math
path = r'./C_S/Orig/'
 #待检测图片的位置
#path = './Orig/Orig/'  #待检测图片的位置
#path = './data/test/'  #待检测图片的位置
#path = './draw/'  #待检测图片的位置
result_path ='./check1/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
    os.makedirs(result_path+'/4/')
    os.makedirs(result_path+'/8/')
    os.makedirs(result_path+'/04/')
    os.makedirs(result_path+'/48/')
    os.makedirs(result_path+'/no/')
    os.makedirs(result_path+'/cut/')
#    txt_path =result_path + '/result.txt'
#    file = open(txt_path,'w')
elif os.path.exists(result_path):
     shutil.rmtree(result_path)
     os.makedirs(result_path)
     os.makedirs(result_path+'/4/')
     os.makedirs(result_path+'/8')
     os.makedirs(result_path+'/04/')
     os.makedirs(result_path+'/48/')
     os.makedirs(result_path+'/no/')
     os.makedirs(result_path+'/cut/')
#    txt_path =result_path + '/result.txt'
#    file = open(txt_path,'w')
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, name,image,img,cropped_save_path):

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes),name))
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                        size=np.floor(2e-2 * image.size[1] + 0.2).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500
        card_box=[]
        center_point=[]
        print('########################################')
        top, left, bottom, right = 0,1024,1024,0
        if(len(out_boxes)>0):

             for i, c in reversed(list(enumerate(out_classes))):
                     k=0
                     predicted_class = self.class_names[c]
#                     print('predicted_class',predicted_class)
                     score = out_scores[i]
                     print('old',i,predicted_class,score)
                     if(score>0.50):
                          box = out_boxes[i]
                          label = '{} {:.2f}'.format(predicted_class, score)
                          score = '{:.2f}'.format(score)
                          top, left, bottom, right = box
                          card_box.append(box)
                          top = max(0, np.floor(top + 0.5).astype('int32'))
                          left = max(0, np.floor(left + 0.5).astype('int32'))
                          bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                          right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
#画图
                          draw = ImageDraw.Draw(image)
                          label_size = draw.textsize(label, font)
                          if top - label_size[1] >= 0:
                              text_origin = np.array([left, top - label_size[1]])
                          else:
                              text_origin = np.array([left, top + 1])
                          # My kingdom for a good redistributable image drawing library.
                          for i in range(thickness):
                              draw.rectangle(
                                  [left + i, top + i, right - i, bottom - i],
                                  outline=self.colors[c])
                          draw.rectangle(
                              [tuple(text_origin), tuple(text_origin + label_size)],
                              fill=self.colors[c])
                          draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                          del draw
                          k=k+1
                          print('new',predicted_class,score)
#                          print('########################################')
             print(name,'精度在50%以上的标签有',len(card_box),'个')
#             if(len(out_boxes)==4):
             if(len(card_box)==4):
                       print('标准卡')
                       path=result_path+'./4/'+name
                       print('卡的边界',card_box)
                       w=abs(out_boxes[1][3]-out_boxes[1][1])
                       h=abs(out_boxes[1][2]-out_boxes[1][0])
                       HX=(int((w+h)/2))
                       WX=int((w+h)/2)
                       for n in range (len(card_box)):
                            y=int((card_box[n][0]+card_box[n][2])/2)
                            x=int((card_box[n][1]+card_box[n][3])/2)
                            point=(x,y)
                            center_point.append(point)
                            if(top<y):
                                 top=y
                            if(bottom>y):
                                 bottom=y
                            if(left>x):
                                 left=x
                            if(right<x):
                                 right=x
                       print('卡的中心点',center_point)
                       left=int(left)-WX
                       right=int(right)+WX
                       bottom=int(bottom)-HX
                       top=int(top)+HX
                       print('上下左右',top,bottom,left,right)
                       if(left<0):
                            left=0
                       if(right>1920):
                            right=1920
                       if(bottom<0):
                            bottom=0
                       if(top>1080):
                            top=1080
                       draw = ImageDraw.Draw(image)
                       print('new上下左右',top,bottom,left,right)
                       draw.rectangle(
                      [left , top , right, bottom ],
                        outline=self.colors[c]
                              )
                       cropped = img[bottom:top,left:right]

                       cropped = img[bottom:top,left:right]
                       print('bottom:top,left:right',bottom,top,left,right)
                       for i in range(left,right):#遍历所有长度的点
                            for j in range(bottom,top):#遍历所有宽度的点
                                 image.putpixel((i,j),(234,53,57,255))
                       maskImagepath=result_path+'./48/'+name
                       image.save(maskImagepath)
                       cropped_save_path=result_path+'./cut/'+name
                       cv.imwrite(cropped_save_path,cropped)
                       image.save(path)
             elif(len(card_box)>2 and len(card_box)<4):
                       print('疑似卡')
                       w=abs(out_boxes[1][3]-out_boxes[1][1])
                       h=abs(out_boxes[1][2]-out_boxes[1][0])
                       HX=(int((w+h)/2))
                       WX=int((w+h)/2)
                       for n in range (len(card_box)):
                            print('原来数据标签',card_box)
                            y=int((card_box[n][0]+card_box[n][2])/2)
                            x=int((card_box[n][1]+card_box[n][3])/2)
                            print('对应中心点',x,y)
                            point=(x,y)
                            center_point.append(point)

                       print('识别出来后的框选center_point',center_point)
                       D1=math.sqrt((abs((center_point[0][0]-center_point[1][0])))**2+
                               (abs((center_point[0][1]-center_point[1][1])))**2)
                       D2=math.sqrt((abs((center_point[0][0]-center_point[2][0])))**2+
                               (abs((center_point[0][1]-center_point[2][1])))**2)
                       D3=math.sqrt((abs((center_point[1][0]-center_point[2][0])))**2+
                               (abs((center_point[1][1]-center_point[2][1])))**2)
                       print('距离',D1,D2,D3)
                       D=[D1,D2,D3]
                       X=[center_point[0][0],center_point[1][0],center_point[2][0]]
                       Y=[center_point[0][1],center_point[1][1],center_point[2][1]]
                       Dmin=10000
                       Dmax=0
                       for n in range (len(D)):
                            if(Dmin>D[n]):
                                 Dmin=D[n]
                            if(Dmax<D[n]):
                                 print('N的值：',n)
                                 Dmax=D[n]
                                 n_end=n
                       print('Dmax的值：',Dmax)
                       if(n_end==0):
                            print('n_end的值：',n_end)
                            C_x=int((center_point[0][0]+center_point[1][0])/2)
                            C_y=int((center_point[0][1]+center_point[1][1])/2)
                            X4_cent=2*C_x-center_point[2][0]
                            Y4_cent=2*C_y-center_point[2][1]
                       elif(n_end==1):
                            print('n_end的值：',n_end)
                            C_x=int((center_point[0][0]+center_point[2][0])/2)
                            C_y=int((center_point[0][1]+center_point[2][1])/2)
                            X4_cent=2*C_x-center_point[1][0]
                            Y4_cent=2*C_y-center_point[1][1]
                       else:
                            print('n_end的值：',n_end)
                            C_x=int((center_point[1][0]+center_point[2][0])/2)
                            C_y=int((center_point[1][1]+center_point[2][1])/2)
                            X4_cent=2*C_x-center_point[0][0]
                            Y4_cent=2*C_y-center_point[0][1]
                       print('第四个点的坐标',X4_cent,Y4_cent)
                       print('中心点的坐标',C_x,C_x)
                       X.append(X4_cent)
                       Y.append(Y4_cent)
                       print('X的值：',X)
                       print('Y的值：',Y)
                       right=np.max(X)
                       left= np.min(X)
                       top=np.max(Y)
                       bottom=np.min(Y)
                       left=int(left)-WX
                       right=int(right)+WX
                       bottom=int(bottom)-HX
                       top=int(top)+HX
                       print('上下左右',top,bottom,left,right)
                       if(left<0):
                            left=0
                       if(right>1920):
                            right=1920
                       if(bottom<0):
                            bottom=0
                       if(top>1080):
                            top=1080
#                       print('上下左右',top,bottom,left,right)
                       draw = ImageDraw.Draw(image)
                       print('new上下左右',top,bottom,left,right)
                       draw.rectangle(
                      [left , top , right, bottom ],
                        outline=self.colors[c]
                              )
                       Ri=int((Dmin/Dmax)*100)
                       wucha=60-Ri
                       print('最大最小距离，Ri',Dmax,Dmin,Ri)
                       print('误差',wucha)
                       WC=wucha
                       WC=str('ZHKJ_error'+str(WC))
#                       draw.text((100, 300),  "Hello World", font = font, fill = (255, 255, 255))
                       draw = ImageDraw.Draw(image)
                       draw.text((int(C_x),int(C_y)), 'C', fill=(255, 48, 48), font=font)
                       draw.text((int(center_point[0][0]),int(center_point[0][1])), 'AA1', fill=(255, 48, 48), font=font)
                       draw.text((int(center_point[1][0]),int(center_point[1][1])), 'AA2', fill=(255, 48, 48), font=font)
                       draw.text((int(center_point[2][0]),int(center_point[2][1])), 'AA3', fill=(255, 48, 48), font=font)
                       draw.text((int(X4_cent),int(Y4_cent)), 'AA？', fill=(255, 48, 48), font=font)
                       draw.text((800, 600), WC, fill=(255, 48, 48), font=font)
                       if(wucha<10):
                             cropped = img[bottom:top,left:right]
                             print('bottom:top,left:right',bottom,top,left,right)
                             for i in range(left,right):#遍历所有长度的点
                                  for j in range(bottom,top):#遍历所有宽度的点
                                      image.putpixel((i,j),(234,53,57,255))
                             maskImagepath=result_path+'./48/'+name
                             image.save(maskImagepath)

                             cropped_save_path=result_path+'./cut/'+name
                             cv.imwrite(cropped_save_path,cropped)

                             path=result_path+'./04/'+name
                             image.save(path)


                       else:
                            path=result_path+'./no/'+name
                            print('没有存猪卡')
                            image.save(path)
#                       path=result_path+'./04/'+name
#                       image.save(path)
             elif(len(card_box)>4 and len(card_box)<8 ):
                       print('多张疑似卡')
                       path=result_path+'./48/'+name
                       image.save(path)
             else:
                       path=result_path+'./no/'+name
                       print('没有存猪卡')
                       image.save(path)
                       return image
        else:
             path=result_path+'./no/'+name
             image.save(path)
             return -1


    def close_session(self):
        self.sess.close()
if __name__ == '__main__':

    yolo = YOLO()
    zzz=1
    for filename in os.listdir(path):
        image_path = path+'/'+filename
        portion = os.path.split(image_path)
        image = Image.open(image_path)
        img = cv.imread(image_path)
        cropped_save_path =result_path
        name=portion[1]
#        print(name)

        print('处理照片张数是：',zzz)
        zzz=zzz+1
        r_image= yolo.detect_image(name,image,img,cropped_save_path)
        t1 = time.time()

        if(r_image==-1):
            print(portion[1]+'图中不存在卡片')
        else:
            print('ok')
            time_sum = time.time() - t1
            print('time sum:',time_sum)

#            image_save_path =result_path+'./draw/'+portion[1]
#            print('detect result save position:'+image_save_path)
#            r_image.save(image_save_path)

