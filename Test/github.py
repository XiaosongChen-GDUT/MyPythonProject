import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as f
# 这个包很关键
import numpy as np
class a(QWidget):
    def __init__(self):
        super().__init__()
        self.label = None
        self.initui()

    def initui(self):
        self.setWindowTitle('尝试')
        self.resize(1000,700)
        h=QVBoxLayout()
        figcanvas=self.get_picture()
        figcanvas.mpl_connect('key_press_event', self.onclick)
        h.addWidget(figcanvas)
        self.setLayout(h)

    def get_picture(self):
        fig, ax = plt.subplots()
        figcanvas=f(fig)
        # f 就是FigureCanvasQTAgg
        ax.scatter(np.random.randint(0,100,size=20), np.random.randint(0,100,size=20))
        # 散点图
        return figcanvas

    def onclick(self,event):
        print(event.key)
# 点击一下，就会打印一个1，简单
# 而且范围整个窗口，不仅仅在图的坐标里

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    a = a()
    a.show()
    sys.exit(app.exec_())