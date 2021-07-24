import os
import random

trainval_percent = 0.2
train_percent = 0.8

myClass="pedestrain"
# xmlfilePath = "./aiLabel/VOC2012/"+myClass+"/Annotations"
# txtsavepath = "./aiLabel/VOC2012/"+myClass+"/Main"
xmlfilePath = "yolov5-master/VOCdevkit/Annotations"
txtsavepath = "yolov5-master/VOCdevkit/Main"
total_xml = os.listdir(xmlfilePath)

num = len(total_xml)
list = range(num)
tv = int(num* trainval_percent)#训练验证集数量
tr = int(tv * train_percent)#训练集数量
trainval = random.sample(list,tv)
train  = random.sample(trainval,tr)

# ftrainval = open('./aiLabel/VOC2012/'+myClass+'/ImageSets/Main/trainval.txt','w')
# ftest = open('./aiLabel/VOC2012/'+myClass+'/ImageSets/Main/test.txt','w')
# ftrain = open('./aiLabel/VOC2012/'+myClass+'/ImageSets/Main/train.txt','w')
# fval = open('./aiLabel/VOC2012/'+myClass+'/ImageSets/Main/val.txt','w')
ftrainval = open('yolov5-master/VOCdevkit/ImageSets/Main/trainval.txt','w')
ftest = open('yolov5-master/VOCdevkit/ImageSets/Main/test.txt','w')
ftrain = open('yolov5-master/VOCdevkit/ImageSets/Main/train.txt','w')
fval = open('yolov5-master/VOCdevkit/ImageSets/Main/val.txt','w')

for i in list:
    name = total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            fval.write(name)
        else:
            ftest.write(name)
    else:
        ftrain.write(name)



ftrainval.close()
ftrain.close()
fval.close()
ftest.close()


