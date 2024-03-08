# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'virtual_background.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox, QLabel

from show.myqlabel import myLabel

from show.optic_flow_process import optic_flow_process
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F


from mmseg.apis import MMSegInferencer
from PIL import Image

bg = cv2.imread('./show/bg_img/bg.jpg')  # 读取背景图片
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
camera = False
output_sequence = []  # 存放输出的图片


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()
        self.height = int(self.screenheight * 0.7)
        self.width = int(self.screenwidth * 0.7)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(self.width, self.height)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(500, 300))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.verticalLayout_2.setContentsMargins(0, 0, -1, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setMinimumSize(QtCore.QSize(0, 0))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, -146, 213, 800))
        self.scrollAreaWidgetContents.setMinimumSize(QtCore.QSize(400, 1000))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.lbpic1 = myLabel(self.scrollAreaWidgetContents)
        self.lbpic1.setObjectName("lbpic1")
        self.lbpic1.setText("")
        self.verticalLayout_3.addWidget(self.lbpic1)
        self.lbpic2 = myLabel(self.scrollAreaWidgetContents)
        self.lbpic2.setObjectName("lbpic2")
        self.lbpic2.setText("")
        self.verticalLayout_3.addWidget(self.lbpic2)
        self.lbpic3 = myLabel(self.scrollAreaWidgetContents)
        self.lbpic3.setObjectName("lbpic3")
        self.lbpic3.setText("")
        self.verticalLayout_3.addWidget(self.lbpic3)
        self.lbpic4 = myLabel(self.scrollAreaWidgetContents)
        self.lbpic4.setObjectName("lbpic4")
        self.lbpic4.setText("")
        self.verticalLayout_3.addWidget(self.lbpic4)
        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 1)
        self.verticalLayout_3.setStretch(2, 1)
        self.verticalLayout_3.setStretch(3, 1)
        self.gridLayout_5.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_2.addWidget(self.scrollArea)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout_2.addItem(spacerItem2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        self.pushButton_3.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_3.setMaximumSize(QtCore.QSize(1000, 50))
        self.pushButton_3.setSizeIncrement(QtCore.QSize(0, 0))
        self.pushButton_3.setBaseSize(QtCore.QSize(0, 0))
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        self.textBrowser.setMinimumSize(QtCore.QSize(140, 50))
        self.textBrowser.setMaximumSize(QtCore.QSize(16777215, 50))
        self.textBrowser.setObjectName("textBrowser")
        self.horizontalLayout.addWidget(self.textBrowser)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_2.setStretch(0, 7)
        self.verticalLayout_2.setStretch(1, 2)
        self.verticalLayout_2.setStretch(2, 1)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 12)
        self.horizontalLayout_2.setStretch(2, 1)
        self.horizontalLayout_2.setStretch(3, 9)
        self.horizontalLayout_2.setStretch(4, 1)
        self.gridLayout_4.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 614, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")


        self.model1 = QtWidgets.QAction(MainWindow)
        self.model1.setObjectName("model1")
        self.model2 = QtWidgets.QAction(MainWindow)
        self.model2.setObjectName("model2")



        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")



        self.menu.addAction(self.model1)
        self.menu.addAction(self.model2)




        self.menu_2.addAction(self.actionOpen)
        self.menu_2.addAction(self.actionSave)
        self.menu_2.addSeparator()
        self.menu_2.addAction(self.actionExit)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.retranslateUi(MainWindow)

        self.actionExit.triggered.connect(self.exit)

        self.actionExit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.actionOpen.triggered.connect(self.open_image)
        self.actionSave.triggered.connect(self.save)
        self.pushButton_3.clicked.connect(self.open_image)
        self.pm1 = QPixmap("./show/bg_img/bg1.jpg")
        w = self.pm1.width()
        h = self.pm1.height()
        # 根据图像与label的比例，最大化图像在label中的显示
        ratio = max(w / self.lbpic1.width(), h / self.lbpic1.height())
        self.pm1.setDevicePixelRatio(ratio)
        # 图像在label中居中显示
        self.lbpic1.setAlignment(Qt.AlignCenter)
        self.lbpic1.setPixmap(self.pm1)
        self.pm2 = QPixmap("./show/bg_img/bg2.jpg")
        w = self.pm2.width()
        h = self.pm2.height()
        # 根据图像与label的比例，最大化图像在label中的显示
        ratio = max(w / self.lbpic2.width(), h / self.lbpic2.height())
        self.pm2.setDevicePixelRatio(ratio)
        # 图像在label中居中显示
        self.lbpic2.setAlignment(Qt.AlignCenter)
        self.lbpic2.setPixmap(self.pm2)
        self.pm3 = QPixmap("./show/bg_img/bg3.jpg")
        w = self.pm3.width()
        h = self.pm3.height()
        # 根据图像与label的比例，最大化图像在label中的显示
        ratio = max(w / self.lbpic3.width(), h / self.lbpic3.height())
        self.pm3.setDevicePixelRatio(ratio)
        # 图像在label中居中显示
        self.lbpic3.setAlignment(Qt.AlignCenter)
        self.lbpic3.setPixmap(self.pm3)
        self.pm4 = QPixmap("./show/bg_img/bg4.jpg")
        w = self.pm4.width()
        h = self.pm4.height()
        # 根据图像与label的比例，最大化图像在label中的显示
        ratio = max(w / self.lbpic4.width(), h / self.lbpic4.height())
        self.pm4.setDevicePixelRatio(ratio)
        # 图像在label中居中显示
        self.lbpic4.setAlignment(Qt.AlignCenter)
        self.lbpic4.setPixmap(self.pm4)
        self.lbpic1.connect_customized_slot(lambda:self.show_image(1))
        self.lbpic2.connect_customized_slot(lambda:self.show_image(2))
        self.lbpic3.connect_customized_slot(lambda:self.show_image(3))
        self.lbpic4.connect_customized_slot(lambda:self.show_image(4))
        self.isCamera = True



        self.model1.triggered.connect(lambda:self.Open(1)) #这里是改变模型的按钮所对应的函数
        self.model2.triggered.connect(lambda:self.Open(2))
        #可以添加一个按钮，用来选择模型


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Virtual Background"))
        self.pushButton_3.setText(_translate("MainWindow", "Open File"))
        self.menu.setTitle(_translate("MainWindow", "Models"))
        self.menu_2.setTitle(_translate("MainWindow", "Edit"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.model1.setText(_translate("MainWindow", "Supervise-Portrait"))  #这里是改变模型的按钮
        #可以添加一个按钮，用来选择模型
        self.model2.setText(_translate("MainWindow", "PP-HumanSeg14K"))

        self.actionExit.setText(_translate("MainWindow", "Exit"))



    def open_image(self):
        global bg
        self.image = None
        # 获取图像的路径
        self.img_path = QFileDialog.getOpenFileName()[0]
        # 将路径存储到对话框中
        self.textBrowser.setText(self.img_path)
        # 可选的图像格式
        img_type = [".bmp", ".jpg", ".png", ".gif"]
        box = QMessageBox()
        box.setWindowTitle("Warning")
        box.setStandardButtons(QMessageBox.Ok)
        for ig in img_type:
            if ig not in self.img_path:
                continue
            else:
                self.image = True
                # 如果是图像文件名的话，读取图像
                bg = cv2.imread(self.img_path)
                # 将图像转换成RGB格式
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                
        if self.image is None:
            box.setText("请选择图片文件")
            box.exec_()


    def exit(self):
        global camera
        if camera == True:
            self.cap.release()
            self.stopEvent = threading.Event()
            self.stopEvent.clear()


    def show_image(self, num):
        global bg
        if num == 1:
            bg = cv2.imread('show/bg_img/bg1.jpg')
        elif num == 2:
            bg = cv2.imread('show/bg_img/bg2.jpg')
        elif num == 3:
            bg = cv2.imread('show/bg_img/bg3.jpg')
        elif num == 4:
            bg = cv2.imread('show/bg_img/bg4.jpg')

                # 获取图像的宽和高

        # 将图像转换成RGB格式
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)



    def Open(self, num):
        global camera
        if camera == True:
            self.cap.release()
            self.stopEvent = threading.Event()
            self.stopEvent.clear()
        if not self.isCamera:
            self.fileName, self.fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4')
            self.cap = cv2.VideoCapture(self.fileName)
            self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            # 下面两种rtsp格式都是支持的
            # cap = cv2.VideoCapture("rtsp://admin:Supcon1304@172.20.1.126/main/Channels/1")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        # 创建视频显示线程
        if num == 1:
            th = threading.Thread(target=lambda:self.Display(1))
        else:
            th = threading.Thread(target=lambda:self.Display(2))
        th.start()

    def save(self):
        global output_sequence

        output_folder = 'show/output'  # 保存输出图像的文件夹路径

        img = cv2.cvtColor(output_sequence[-1], cv2.COLOR_RGB2BGR)

        n = len(os.listdir(output_folder)) + 1
        filepath = os.path.join(output_folder, 'output' + str(n) + '.jpg')
        cv2.imwrite(filepath, output_sequence[-1])






    def Display(self, num):
        global camera
        camera = True
        global bg
        count = 0  # 初始化计数器


        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:  # 如果没有读取到帧，则结束循环
                break

            if count % 1 == 0:
                # 处理每十帧抽取一帧的代码
                # 这里只是简单地将帧添加到列表中，你可以在这里加入自己的处理算法

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
                w = img.width()
                h = img.height()
                ratio = max(w / self.label.width(), h / self.label.height())
                img.setDevicePixelRatio(ratio)
                # 图像在label中居中显示
                self.label_2.setAlignment(Qt.AlignCenter)
                self.label_2.setPixmap(QPixmap.fromImage(img))

                if num == 1:
                    config_path = 'final_config.py'  # 模型文件路径
                    checkpoint_path = 'checkpoints/Supervise-Portrait.pth'  # 权重路径
                else:
                    config_path = 'baidu.py'  # 模型文件路径
                    checkpoint_path = 'checkpoints/PP-HumanSeg14K.pth'  # 权重路径
                img_path = frame  # 图片路径，这里既可以是一张图片，也可以是一个文件夹
                # 这一行会加载模型，会花一些时间，放在初始化里，后边只要用inferencer就行
                inferencer = MMSegInferencer(model=config_path, weights=checkpoint_path)
                # 推理给定图像
                result = inferencer(img_path, show=False)
                score_map_temprary = []  # 暂时存放score_map
                global output_sequence
                post_or_not = True
                frame_sequence = [result["predictions"]]
                img_sequence = [frame]
                origin_img = img_sequence[0]
                h, w, _ = origin_img.shape
                # 读入原始数据
                score_map = frame_sequence[0]
                if post_or_not:
                    for img in frame_sequence:
                        score_map = img.copy()
                        # 进行post process
                        mask_original = score_map.copy()
                        mask_original = (mask_original).astype("uint8")
                        _, mask_thr = cv2.threshold(mask_original, 240, 1, cv2.THRESH_BINARY)
                        kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
                        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25))
                        mask_erode = cv2.erode(mask_thr, kernel_erode)
                        mask_dilate = cv2.dilate(mask_erode, kernel_dilate)
                        score_map_temprary.append(mask_dilate * 255)
                for score_maps in score_map_temprary:
                    score_maps = torch.from_numpy(score_map)

                    score_maps = score_maps.unsqueeze(0).unsqueeze(0)

                # 进行逆变换操作
                    target_shape = (h, w)
                    target_h, target_w = target_shape
                    mode = 'bilinear'

                    # 执行反向操作，例如反向缩放
                    score_maps = F.interpolate(score_maps.float(), size=(target_h, target_w), mode=mode)


                    alpha = score_maps.squeeze(1).permute(1, 2, 0).numpy()
                # print(alpha)

                    bg = cv2.resize(bg, (w, h))

                    if bg.ndim == 2:
                        bg = bg[..., np.newaxis]

                    out = (alpha * origin_img + (1 - alpha) * bg).astype(np.uint8)

                    output_sequence.append(out)



                img = QImage(output_sequence[-1].data, output_sequence[-1].shape[1], output_sequence[-1].shape[0],
                             output_sequence[-1].shape[1] * 3, QImage.Format_RGB888)
                w = img.width()
                h = img.height()
                ratio = max(w / self.label.width(), h / self.label.height())
                img.setDevicePixelRatio(ratio)
                # 图像在label中居中显示
                self.label.setAlignment(Qt.AlignCenter)
                self.label.setPixmap(QPixmap.fromImage(img))

            #cv2.waitKey(1)
            count += 1

        self.cap.release()





