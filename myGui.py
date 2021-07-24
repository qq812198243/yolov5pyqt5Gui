#-*- coding:utf-8 -*-

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

'''
Basic Layout
'''
__author__ = 'wqx'
import sys
from PyQt5.QtGui import  QPixmap
from PyQt5.QtWidgets import QTableWidgetItem,QTableWidget,QMessageBox,QWidget, QApplication, QGroupBox, QPushButton, QLabel, QHBoxLayout,  QVBoxLayout, QGridLayout, QFormLayout, QLineEdit, QTextEdit
from PyQt5.QtCore import pyqtSlot
import myDetectGui

#----------------

class myArgs:
    save_imgagnostic_nms=False
    augment=False
    classes=None
    conf_thres=0.25
    device=''
    exist_ok=False
    img_size=640
    iou_thres=0.45
    name='exp'
    nosave=False
    project='runs/detect'
    save_conf=False
    save_txt=False
    source='data/images'
    update=False
    view_img=False
    weights='yolov5wqx/weights/best.pt'
    agnostic_nms=True


#-----------------
myNamesDict={"person":0,"face_mask":0,"hat":0,"glasses":0,"head":0}

class APP(QWidget):
    # namesDict={"person":1,"face_mask":110,"hat":0,"glasses":0,"head":0}



    def __init__(self):
        super(APP,self).__init__()
        self.initUi()
        

    def initUi(self):
        self.createGridGroupBox()
        self.creatVboxGroupBox()
        self.creatFormGroupBox()
        mainLayout = QVBoxLayout()
        hboxLayout = QHBoxLayout()
        hboxLayout.addStretch()  
        hboxLayout.addWidget(self.gridGroupBox)
        hboxLayout.addWidget(self.vboxGroupBox)
        mainLayout.addLayout(hboxLayout)
        mainLayout.addWidget(self.formGroupBox)
        self.setLayout(mainLayout)

    def createGridGroupBox(self):
        self.gridGroupBox = QGroupBox("用户输入")
        layout = QGridLayout()

        nameLabel = QLabel("中文名称")
        self.nameLineEdit = QLineEdit(self)
        nameIdLabel = QLabel("工  号")
        self.nameIdLineEdit = QLineEdit(self)
        numLabel = QLabel("团队人数")
        self.numLineEdit = QLineEdit(self)

        mesgSubmit = QPushButton("提交用户数据")
        # timeLabel = QLabel("时间")
        # timeLineEdit = QLineEdit("9月15日")
        # imgeLabel = QLabel()
        # pixMap = QPixmap("tiangong.png")
        # imgeLabel.setPixmap(pixMap)
        layout.setSpacing(10) 
        layout.addWidget(nameLabel,1,0)
        layout.addWidget(self.nameLineEdit,1,1)
        layout.addWidget(nameIdLabel,2,0)
        layout.addWidget(self.nameIdLineEdit,2,1)
        layout.addWidget(numLabel,3,0)
        layout.addWidget(self.numLineEdit,3,1)

        layout.addWidget(mesgSubmit,4,1)
        # layout.addWidget(timeLabel,3,0)
        # layout.addWidget(timeLineEdit,3,1)
        # layout.addWidget(imgeLabel,0,2,4,1)
        layout.setColumnStretch(1, 10)
        self.gridGroupBox.setLayout(layout)
        self.setWindowTitle('工地安全提醒系统')
        mesgSubmit.clicked.connect(self.buttonClickedMesg)
        

    def creatVboxGroupBox(self):
        self.vboxGroupBox = QGroupBox("指示")
        layout = QVBoxLayout() 
        nameLabel = QLabel("报警提示")
        self.bigEditor = QTextEdit(self)
        self.bigEditor.setPlainText("一切正常")

        layout.addWidget(nameLabel)
        layout.addWidget(self.bigEditor)
        self.vboxGroupBox.setLayout(layout)

    def creatFormGroupBox(self):
        self.formGroupBox = QGroupBox("输出报告")
        layout = QFormLayout()
        performanceLabel = QLabel("性能特点：")
        performanceEditor = QLineEdit("为了施工安全进行信息提示")
        # planLabel = QLabel("检查信息发射：")
        # planEditor = QTextEdit()
        # planEditor.setPlainText("检查消息")
        
        tableLabel = QLabel("检查表格")
        self.mytableWidget = QTableWidget(self)
        # set row count
        self.mytableWidget.setRowCount(0)
        # set column count
        self.mytableWidget.setColumnCount(5)
        self.mytableWidget.setHorizontalHeaderLabels(["person","face_mask","hat","glasses","head"])
        
        # AddSubmit = QPushButton("启动")
        
        # AddSubmit.clicked.connect(self.addTabel)

        layout.addRow(performanceLabel,performanceEditor)
        layout.addRow(tableLabel,self.mytableWidget)
        # layout.addRow(AddSubmit)
        self.formGroupBox.setLayout(layout)
    
    def addTabel(self,newNamesDict=myNamesDict):
        try:
            
            row_cnt = self.mytableWidget.rowCount()     # 返回当前行数（尾部）
            self.mytableWidget.insertRow(row_cnt)       # 尾部插入一行新行表格
            column_cnt = self.mytableWidget.columnCount()   # 返回当前列数
            # for row_name in newNamesDict.keys():
            #     print("xxxxx")
            for column,row_name in zip(range(column_cnt),newNamesDict.keys()):
                
                item1 = QTableWidgetItem(str(newNamesDict[row_name]))
                self.mytableWidget.setItem(row_cnt, column, item1) #最后，将(行，列，内容)配置
            self.mytableWidget.resizeColumnsToContents()  # 设置列宽高按照内容自适应
            print("Slot Table Widget test setup successfully.")
        except Exception as error:
            print("Error for appending table list. error==",error)
    
    def updateWaring(self,myWarning="一切正常"):
        self.bigEditor.setPlainText(myWarning)

    #--------------------
    def detect(self,save_img=False,save_imgagnostic_nms=False,augment=False, classes=None, conf_thres=0.25, device='', exist_ok=False, img_size=640, iou_thres=0.45, name='exp', nosave=False, project='runs/detect', save_conf=False, save_txt=False, source='data/images', update=False, view_img=False, weights='yolov5wqx/weights/best.pt'):
        namesDict={"person":0,"face_mask":0,"hat":0,"glasses":0,"head":0}
        

        opt = myArgs()
        print("opt=",opt)
        
        # check_requirements(exclude=('pycocotools', 'thop'))

        # with torch.no_grad():
        #     if opt.update:  # update all models (to fix SourceChangeWarning)
        #         for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
        #             detect()
        #             strip_optimizer(opt.weights)
        #     else:
        #         detect()
        
        
        # 获取输出文件夹，输入源，权重，参数等参数
        # source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        imageDes={}
        view_img, save_txt, imgsz = opt.view_img, opt.save_txt, opt.img_size


        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        # 目录
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        # 获取设备
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        # 加载Float32模型，确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16  设置Float16

        # Second-stage classifier
        # 设置第二次分类，默认不使用
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        # 通过不同的输入源来设置不同的数据加载方式
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference 设置为True可以加快恒定图像大小的推断
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors 获取类别名字 设置画框的颜色
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference 进行一次前向推理,测试程序是否正常
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        """
        path 图片/视频路径
        img 进行resize+pad之后的图片
        img0 原size图片
        cap 当读取图片时为None，读取视频时为视频源
        """

        for path, img, im0s, vid_cap in dataset:

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32 
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # 没有batch_size的话则在最前面添加一个轴
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            """
            前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
            h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
            num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
            pred[..., 0:4]为预测框坐标
            预测框坐标为xywh(中心点+宽长)格式
            pred[..., 4]为objectness置信度
            pred[..., 5:-1]为分类结果
            """

            # Apply NMS
            """
            pred:前向传播的输出
            conf_thres:置信度阈值
            iou_thres:iou阈值
            classes:是否只保留特定的类别
            agnostic:进行nms是否也去除不同类别之间的框
            经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
            pred是一个列表list[torch.tensor]，长度为batch_size
            每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
            """
            # print("xxxxxxxxxxxxxx",opt.agnostic_nms)
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            # 添加二次分类，默认不使用
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            # 对每一张图片作处理
            
            for i, det in enumerate(pred):  # detections per image
                
                
                # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                # 设置保存图片/视频的路径
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                # 设置保存框坐标txt文件的路径
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                # 设置打印信息(图片长宽)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                
                if len(det):
                    
                    # Rescale boxes from img_size to im0 size
                    # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                    # 此时坐标格式为xyxy
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # 打印检测到的类别数量
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # print("name=",names[int(c)],"number=",int(n))#类别的数量
                        namesDict[names[int(c)]]=int(n)


                   
                    # print("imgName=",path,"s=",s)#图片地址和信息
                    
                    if source!="0":
                        imageDes[path.split('\\')[-1]]=s#给key赋值图片名
                        
                    # Write results
                    # 保存预测结果
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # 在原图上画框
                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                print("namesDict=",namesDict)
                
                self.addTabel(namesDict)
                print("xxxxxxxxx==",namesDict['hat'],namesDict['glasses'])
                if int(namesDict['hat'])>0 and int(namesDict['glasses'])>0:
                    
                    self.updateWaring("安全帽子与护目镜佩戴正常√")
                else:
                    self.updateWaring("请佩戴好安全帽子与护目镜×")
                
                namesDict={"person":0,"face_mask":0,"hat":0,"glasses":0,"head":0}
                # Print time (inference + NMS)
                # 打印前向传播+nms时间
                # print(f'{s}Done. ({t2 - t1:.3f}s)')#-----------------原

                # Stream results
                #如果设置展示，则show图片/视频
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                # 设置保存图片/视频
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            # 打开保存图片和txt的路径(好像只适用于MacOS系统)
            print(f"Results saved to {save_dir}{s}")
        # 打印总时间
        print(f'Done. ({time.time() - t0:.3f}s)')
    
    #---------------------
    @pyqtSlot()
    def buttonClickedMesg(self):
        textboxValueName = self.nameLineEdit.text()
        textboxValueNameID = self.nameIdLineEdit.text()
        textboxValueNumber = self.numLineEdit.text()
        buttonReply = QMessageBox.question(self, '提示信息', "是否提交数据", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            print('Yes clicked.'+textboxValueName+'  '+textboxValueNameID+'   '+textboxValueNumber)#提交到数据库
            self.detect(source="0",weights='yolov5wqx/weights/best.pt',save_img=False)
        else:
            print('No clicked.')

    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = APP()
    # newNamesDict={"person":0,"face_mask":0,"hat":0,"glasses":0,"head":0}

    # ex.addTabel(newNamesDict)
    ex.show()

    sys.exit(app.exec_())
