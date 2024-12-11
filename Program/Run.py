from PyQt5.QtWidgets import QApplication, QMainWindow
from qt_material import apply_stylesheet
from Program.View import Ui_MainWindow
import qdarkstyle
import simpy
import sys



def main():
    env = simpy.Environment()
    try:
        app = QApplication(sys.argv)
        # setup stylesheet
        # apply_stylesheet(app, theme='dark_cyan.xml')
        #加载qdarkstyle样式
        dark_stylesheet = qdarkstyle.load_stylesheet_pyqt5()
        app.setStyleSheet(dark_stylesheet)
        MainWindow = QMainWindow()

        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"应用启动失败: {e}")  # 错误处理，输出启动失败原因


if __name__ == '__main__':
    main()
