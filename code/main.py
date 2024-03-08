from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import virtual_background1  # 导入图像界面设计文件



if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = virtual_background1.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
