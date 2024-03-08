from PyQt5 import QtCore
from PyQt5.QtCore import  pyqtSignal
from PyQt5.QtWidgets import  QLabel

class myLabel(QLabel):
    button_clicked_signal = QtCore.pyqtSignal()

    def mouseReleaseEvent(self, QMouseEvent):
        self.button_clicked_signal.emit()

    def connect_customized_slot(self, func):  ##重载一下鼠标点击事件
        self.button_clicked_signal.connect(func)