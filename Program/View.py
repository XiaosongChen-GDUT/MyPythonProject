# -*- coding: utf-8 -*-
# from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
# from PyQt5.QtWidgets import QWidget
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as  NavigationToolbar
from PyQt5 import QtCore, QtGui, QtWidgets
from Program.DataModel import Model
from Algorithm.PathPlanning import Path_Planning
from Program.graph_Canvas import graph_FigureCanvas,ResultsDialog
from datetime import datetime
import matplotlib

matplotlib.use("Qt5Agg")  # 声明使用QT5
class Ui_MainWindow(object):
    def __init__(self):
        #读取数据
        self.model = Model();
        self.maps = [self.model.floor1, self.model.floor2, self.model.floor3,self.model.combined_graph]
        #创建画布
        self.first_floor_canvas = graph_FigureCanvas(floor=self.maps[0], title="First Floor")
        self.second_floor_canvas = graph_FigureCanvas(floor=self.maps[1], title="Second Floor")
        self.third_floor_canvas = graph_FigureCanvas(floor=self.maps[2], title="Third Floor")
        self.combined_floor_canvas = graph_FigureCanvas(floor=self.maps[3], title="Combined_graph")
        #添加属性
        self.maps[0].graph["canvas"] = self.first_floor_canvas
        #画布集合
        self.floor_Canvas_list = [self.first_floor_canvas, self.second_floor_canvas, self.third_floor_canvas, self.combined_floor_canvas]
        #寻路算法
        self.path_planing = Path_Planning(self.floor_Canvas_list)

        # MainWindow.setMinimumHeight(900)
        # MainWindow.setMinimumWidth(1500)
        # MainWindow.showMaximized()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setMinimumHeight(900)
        MainWindow.setMinimumWidth(1500)
        MainWindow.showMaximized()

        self.centralwidget = QtWidgets.QWidget(MainWindow)

        font = QtGui.QFont()
        font.setPointSize(12)
        self.centralwidget.setFont(font)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setContentsMargins(0, 3, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setAutoFillBackground(True)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(True)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.first_floor = QtWidgets.QWidget()
        font = QtGui.QFont()
        font.setPointSize(11)
        self.first_floor.setFont(font)
        self.first_floor.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.first_floor.setObjectName("first_floor")
        self.tabWidget.addTab(self.first_floor, "")
        self.second_floor = QtWidgets.QWidget()
        self.second_floor.setObjectName("second_floor")
        self.tabWidget.addTab(self.second_floor, "")
        self.third_floor = QtWidgets.QWidget()
        self.third_floor.setAccessibleName("")
        self.third_floor.setObjectName("third_floor")
        self.tabWidget.addTab(self.third_floor, "")
        self.allMaps = QtWidgets.QWidget()
        self.allMaps.setObjectName("allMaps")
        self.tabWidget.addTab(self.allMaps, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        self.Right_frame = QtWidgets.QFrame(self.centralwidget)
        self.Right_frame.setObjectName("Right_frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.Right_frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.AGV_groupBox = QtWidgets.QGroupBox(self.Right_frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.AGV_groupBox.setFont(font)
        self.AGV_groupBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.AGV_groupBox.setAutoFillBackground(False)
        self.AGV_groupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.AGV_groupBox.setObjectName("AGV_groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.AGV_groupBox)
        self.verticalLayout_2.setContentsMargins(0, 0, -1, 0)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.AddAGV_gridLayout = QtWidgets.QGridLayout()
        self.AddAGV_gridLayout.setContentsMargins(-1, 9, -1, -1)
        self.AddAGV_gridLayout.setObjectName("AddAGV_gridLayout")
        self.AGV_ID = QtWidgets.QLabel(self.AGV_groupBox)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.AGV_ID.setFont(font)
        self.AGV_ID.setAlignment(QtCore.Qt.AlignCenter)
        self.AGV_ID.setObjectName("AGV_ID")
        self.AddAGV_gridLayout.addWidget(self.AGV_ID, 1, 0, 1, 1)
        self.random_addAGV = QtWidgets.QPushButton(self.AGV_groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.random_addAGV.setFont(font)
        self.random_addAGV.setObjectName("random_addAGV")
        self.AddAGV_gridLayout.addWidget(self.random_addAGV, 0, 0, 1, 2)
        self.AGV_location = QtWidgets.QLabel(self.AGV_groupBox)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.AGV_location.setFont(font)
        self.AGV_location.setAlignment(QtCore.Qt.AlignCenter)
        self.AGV_location.setObjectName("AGV_location")
        self.AddAGV_gridLayout.addWidget(self.AGV_location, 2, 0, 1, 1)
        self.AGV_ID_comboBox = QtWidgets.QSpinBox(self.AGV_groupBox)
        self.AGV_ID_comboBox.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.AGV_ID_comboBox.setFont(font)
        self.AGV_ID_comboBox.setAlignment(QtCore.Qt.AlignCenter)
        self.AGV_ID_comboBox.setMaximum(10000)
        self.AGV_ID_comboBox.setObjectName("AGV_ID_comboBox")
        self.AddAGV_gridLayout.addWidget(self.AGV_ID_comboBox, 1, 1, 1, 1)
        self.AGV_location_comboBox = QtWidgets.QSpinBox(self.AGV_groupBox)
        self.AGV_location_comboBox.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.AGV_location_comboBox.setFont(font)
        self.AGV_location_comboBox.setAlignment(QtCore.Qt.AlignCenter)
        self.AGV_location_comboBox.setMaximum(10000)
        self.AGV_location_comboBox.setObjectName("AGV_location_comboBox")
        self.AddAGV_gridLayout.addWidget(self.AGV_location_comboBox, 2, 1, 1, 1)
        self.addAGV = QtWidgets.QPushButton(self.AGV_groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.addAGV.setFont(font)
        self.addAGV.setObjectName("addAGV")
        self.AddAGV_gridLayout.addWidget(self.addAGV, 3, 0, 1, 2)
        self.verticalLayout_2.addLayout(self.AddAGV_gridLayout)
        self.verticalLayout.addWidget(self.AGV_groupBox)
        self.Task_groupBox = QtWidgets.QGroupBox(self.Right_frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Task_groupBox.setFont(font)
        self.Task_groupBox.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.Task_groupBox.setObjectName("Task_groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.Task_groupBox)
        self.verticalLayout_3.setContentsMargins(0, 0, 9, 0)
        self.verticalLayout_3.setSpacing(10)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.Task_gridLayout = QtWidgets.QGridLayout()
        self.Task_gridLayout.setObjectName("Task_gridLayout")
        self.start_spinBox = QtWidgets.QSpinBox(self.Task_groupBox)
        self.start_spinBox.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.start_spinBox.setFont(font)
        self.start_spinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.start_spinBox.setMaximum(10000)
        self.start_spinBox.setObjectName("start_spinBox")
        self.Task_gridLayout.addWidget(self.start_spinBox, 2, 1, 1, 1)
        self.end_spinBox = QtWidgets.QSpinBox(self.Task_groupBox)
        self.end_spinBox.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.end_spinBox.setFont(font)
        self.end_spinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.end_spinBox.setMaximum(10000)
        self.end_spinBox.setObjectName("end_spinBox")
        self.Task_gridLayout.addWidget(self.end_spinBox, 3, 1, 1, 1)
        self.end_label = QtWidgets.QLabel(self.Task_groupBox)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.end_label.setFont(font)
        self.end_label.setAlignment(QtCore.Qt.AlignCenter)
        self.end_label.setObjectName("end_label")
        self.Task_gridLayout.addWidget(self.end_label, 3, 0, 1, 1)
        self.random_AddTask = QtWidgets.QPushButton(self.Task_groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.random_AddTask.setFont(font)
        self.random_AddTask.setObjectName("random_AddTask")
        self.Task_gridLayout.addWidget(self.random_AddTask, 1, 0, 1, 2)
        self.addTask = QtWidgets.QPushButton(self.Task_groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.addTask.setFont(font)
        self.addTask.setObjectName("addTask")
        self.Task_gridLayout.addWidget(self.addTask, 5, 0, 1, 2)
        self.start_label = QtWidgets.QLabel(self.Task_groupBox)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.start_label.setFont(font)
        self.start_label.setAlignment(QtCore.Qt.AlignCenter)
        self.start_label.setObjectName("start_label")
        self.Task_gridLayout.addWidget(self.start_label, 2, 0, 1, 1)
        self.point_AGV_comboBox = QtWidgets.QComboBox(self.Task_groupBox)
        self.point_AGV_comboBox.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.point_AGV_comboBox.setFont(font)
        self.point_AGV_comboBox.setAutoFillBackground(True)
        self.point_AGV_comboBox.setEditable(True)
        self.point_AGV_comboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.point_AGV_comboBox.setObjectName("point_AGV_comboBox")
        self.point_AGV_comboBox.addItem("")
        self.point_AGV_comboBox.addItem("")
        self.Task_gridLayout.addWidget(self.point_AGV_comboBox, 4, 1, 1, 1)
        self.point_AGV = QtWidgets.QLabel(self.Task_groupBox)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.point_AGV.setFont(font)
        self.point_AGV.setAlignment(QtCore.Qt.AlignCenter)
        self.point_AGV.setObjectName("point_AGV")
        self.Task_gridLayout.addWidget(self.point_AGV, 4, 0, 1, 1)
        self.verticalLayout_3.addLayout(self.Task_gridLayout)
        self.verticalLayout.addWidget(self.Task_groupBox)
        self.test_path_planing = QtWidgets.QGroupBox(self.Right_frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.test_path_planing.setFont(font)
        self.test_path_planing.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.test_path_planing.setObjectName("test_path_planing")
        self.gridLayout = QtWidgets.QGridLayout(self.test_path_planing)
        self.gridLayout.setObjectName("gridLayout")
        self.reset_Button = QtWidgets.QPushButton(self.test_path_planing)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.reset_Button.setFont(font)
        self.reset_Button.setObjectName("reset_Button")
        self.gridLayout.addWidget(self.reset_Button, 0, 0, 1, 2)
        self.random_Add_Cases = QtWidgets.QPushButton(self.test_path_planing)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.random_Add_Cases.setFont(font)
        self.random_Add_Cases.setObjectName("random_Add_Cases")
        self.gridLayout.addWidget(self.random_Add_Cases, 1, 0, 1, 2)
        self.Source_Button = QtWidgets.QPushButton(self.test_path_planing)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Source_Button.setFont(font)
        self.Source_Button.setObjectName("Source_Button")
        self.gridLayout.addWidget(self.Source_Button, 2, 0, 1, 1)
        self.source_spinBox = QtWidgets.QSpinBox(self.test_path_planing)
        self.source_spinBox.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.source_spinBox.setFont(font)
        self.source_spinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.source_spinBox.setMaximum(10000)
        self.source_spinBox.setObjectName("source_spinBox")
        self.gridLayout.addWidget(self.source_spinBox, 2, 1, 1, 1)
        self.Target_Button = QtWidgets.QPushButton(self.test_path_planing)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Target_Button.setFont(font)
        self.Target_Button.setObjectName("Target_Button")
        self.gridLayout.addWidget(self.Target_Button, 3, 0, 1, 1)
        self.target_spinBox = QtWidgets.QSpinBox(self.test_path_planing)
        self.target_spinBox.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.target_spinBox.setFont(font)
        self.target_spinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.target_spinBox.setMaximum(10000)
        self.target_spinBox.setObjectName("target_spinBox")
        self.gridLayout.addWidget(self.target_spinBox, 3, 1, 1, 1)
        self.point_algorithm_Qlable = QtWidgets.QLabel(self.test_path_planing)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.point_algorithm_Qlable.setFont(font)
        self.point_algorithm_Qlable.setAlignment(QtCore.Qt.AlignCenter)
        self.point_algorithm_Qlable.setObjectName("point_algorithm_Qlable")
        self.gridLayout.addWidget(self.point_algorithm_Qlable, 4, 0, 1, 1)
        self.point_algorithm_comboBox = QtWidgets.QComboBox(self.test_path_planing)
        self.point_algorithm_comboBox.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.point_algorithm_comboBox.setFont(font)
        self.point_algorithm_comboBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.point_algorithm_comboBox.setAutoFillBackground(True)
        self.point_algorithm_comboBox.setEditable(True)
        self.point_algorithm_comboBox.setMaxVisibleItems(30)
        self.point_algorithm_comboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.point_algorithm_comboBox.setIconSize(QtCore.QSize(16, 20))
        self.point_algorithm_comboBox.setObjectName("point_algorithm_comboBox")
        for i in range(0, 5):    # 添加四种算法 会生成序列 [0, 1, 2, 3]
            self.point_algorithm_comboBox.addItem("")
        # self.point_algorithm_comboBox.addItem("")
        # self.point_algorithm_comboBox.addItem("")
        # self.point_algorithm_comboBox.addItem("")
        # self.point_algorithm_comboBox.addItem("")
        # self.point_algorithm_comboBox.addItem("")
        # self.point_algorithm_comboBox.addItem("")
        self.gridLayout.addWidget(self.point_algorithm_comboBox, 4, 1, 1, 1)
        self.heuristic_Qlable = QtWidgets.QLabel(self.test_path_planing)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.heuristic_Qlable.setFont(font)
        self.heuristic_Qlable.setAlignment(QtCore.Qt.AlignCenter)
        self.heuristic_Qlable.setObjectName("heuristic_Qlable")
        self.gridLayout.addWidget(self.heuristic_Qlable, 5, 0, 1, 1)
        self.heuristic_ComboBox = QtWidgets.QComboBox(self.test_path_planing)
        self.heuristic_ComboBox.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.heuristic_ComboBox.setFont(font)
        self.heuristic_ComboBox.setAutoFillBackground(True)
        self.heuristic_ComboBox.setEditable(True)
        self.heuristic_ComboBox.setMaxVisibleItems(50)
        self.heuristic_ComboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.heuristic_ComboBox.setIconSize(QtCore.QSize(16, 20))
        self.heuristic_ComboBox.setObjectName("heuristic_ComboBox")
        self.heuristic_ComboBox.addItem("")
        self.heuristic_ComboBox.addItem("")
        self.heuristic_ComboBox.addItem("")
        self.heuristic_ComboBox.addItem("")
        self.gridLayout.addWidget(self.heuristic_ComboBox, 5, 1, 1, 1)
        self.add_case = QtWidgets.QPushButton(self.test_path_planing)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.add_case.setFont(font)
        self.add_case.setObjectName("add_case")
        self.gridLayout.addWidget(self.add_case, 6, 0, 1, 2)
        self.verticalLayout.addWidget(self.test_path_planing)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.horizontalLayout.addWidget(self.Right_frame)
        self.horizontalLayout.setStretch(0, 8)
        self.horizontalLayout.setStretch(1, 1)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1149, 101))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.textBrowser = QtWidgets.QTextBrowser(self.scrollAreaWidgetContents)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_4.addWidget(self.textBrowser)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_5.addWidget(self.scrollArea)
        self.verticalLayout_5.setStretch(0, 6)
        self.verticalLayout_5.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1151, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        # self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setBaseSize(QtCore.QSize(0, 5))
        self.statusbar.setAutoFillBackground(True)
        self.statusbar.setInputMethodHints(QtCore.Qt.ImhTime)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.open = QtWidgets.QAction(MainWindow)
        self.open.setCheckable(False)
        self.open.setPriority(QtWidgets.QAction.HighPriority)
        self.open.setObjectName("open")
        self.Save = QtWidgets.QAction(MainWindow)
        self.Save.setObjectName("Save")
        self.menu.addAction(self.open)
        self.menu.addAction(self.Save)
        self.menubar.addAction(self.menu.menuAction())

        self.init_Canvas()
        self.init_Connect()

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)   #默认第一层地图
        self.heuristic_ComboBox.setCurrentIndex(1)  #默认选择曼哈顿距离
        # 设置当前选项卡并使其获得焦点
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    #self.init_Canvas()
    #self.init_Connect()
    #初始化连接
    def init_Connect(self):
        self.statusbar.showMessage("ready")  # 状态栏显示信息

        self.Source_Button.clicked.connect(self.handle_Source_Button)
        self.Target_Button.clicked.connect(self.handle_Target_Button)
        self.random_Add_Cases.clicked.connect(self.handle_random_Add_Cases_button)
        self.add_case.clicked.connect(self.handle_add_case_button)
        self.random_AddTask.clicked.connect(self.handle_random_AddTask_button)
        self.addTask.clicked.connect(self.handle_addTask_button)
        self.random_addAGV.clicked.connect(self.handle_random_addAGV_button)
        self.addAGV.clicked.connect(self.handle_addAGV_button)
        self.point_AGV_comboBox.currentIndexChanged.connect(self.handle_point_AGV_comboBox)
        self.point_algorithm_comboBox.currentIndexChanged.connect(self.handle_point_algorithm_comboBox)
        self.reset_Button.clicked.connect(self.handle_reset_Button)

        # self.first_floor_canvas.mpl_connect('axes_enter_event', self.on_enter_event)
        # self.second_floor_canvas.mpl_connect('axes_enter_event', self.on_enter_event)
        # self.third_floor_canvas.mpl_connect('axes_enter_event', self.on_enter_event)
        # self.combined_floor_canvas.mpl_connect('axes_enter_event', self.on_enter_event)

        self.first_floor_canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.second_floor_canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.third_floor_canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.combined_floor_canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.first_floor.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.second_floor.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.third_floor.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.allMaps.setFocusPolicy(QtCore.Qt.ClickFocus)

    #初始化地图画布
    def init_Canvas(self):
        # self.first_floor.setFocusPolicy(QtCore.Qt.ClickFocus)   # 设置第一层地图获得焦点
        # self.second_floor.setFocusPolicy(QtCore.Qt.ClickFocus)  # 设置第二层地图获得焦点
        # self.third_floor.setFocusPolicy(QtCore.Qt.ClickFocus)   # 设置第三层地图获得焦点
        # self.allMaps.setFocusPolicy(QtCore.Qt.ClickFocus)      # 设置全景地图获得焦点

        self.first_floor_layout = QtWidgets.QVBoxLayout(self.first_floor)
        first_floor_toolbar = NavigationToolbar(self.first_floor_canvas, self.first_floor)
        self.first_floor_layout.addWidget(first_floor_toolbar)
        self.first_floor_layout.addWidget(self.first_floor_canvas)
        self.first_floor_layout.setContentsMargins(0, 0, 0, 0)
        self.first_floor_layout.setObjectName("first_floor_layout")
        self.first_floor_canvas.draw_floor()#画出第一层地图

        self.second_floor_layout = QtWidgets.QVBoxLayout(self.second_floor)
        second_floor_toolbar = NavigationToolbar(self.second_floor_canvas, self.second_floor)
        self.second_floor_layout.addWidget(second_floor_toolbar)
        self.second_floor_layout.addWidget(self.second_floor_canvas)
        self.second_floor_layout.setContentsMargins(0, 0, 0, 0)
        self.second_floor_layout.setObjectName("second_floor_layout")
        self.second_floor_canvas.draw_floor()#画出第二层地图

        self.third_floor_layout = QtWidgets.QVBoxLayout(self.third_floor)
        third_floor_toolbar = NavigationToolbar(self.third_floor_canvas, self.third_floor)
        self.third_floor_layout.addWidget(third_floor_toolbar)
        self.third_floor_layout.addWidget(self.third_floor_canvas)
        self.third_floor_layout.setContentsMargins(0, 0, 0, 0)
        self.third_floor_layout.setObjectName("third_floor_layout")
        self.third_floor_canvas.draw_floor()#画出第三层地图

        self.combined_graph_layout = QtWidgets.QVBoxLayout(self.allMaps)
        combined_floor_toolbar = NavigationToolbar(self.combined_floor_canvas, self.combined_floor_canvas)
        self.combined_graph_layout.addWidget(combined_floor_toolbar)
        self.combined_graph_layout.addWidget(self.combined_floor_canvas)
        self.combined_graph_layout.setContentsMargins(0, 0, 0, 0)
        self.combined_graph_layout.setObjectName("combined_graph_layout")
        # self.combined_floor_canvas.draw_floor()#画出全景地图

    def on_enter_event(self, event):
        widget = self.tabWidget.currentWidget()
        # widget.setFocusPolicy(QtCore.Qt.ClickFocus)
        print("on_enter_event",widget.objectName())

    def keyPressEvent(self, event):
        print("keyPressEvent",event.key)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "四向穿梭车货位优化与集成调度管理系统"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.first_floor), _translate("MainWindow", "第一层货位地图"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.second_floor), _translate("MainWindow", "第二层货位地图"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.third_floor), _translate("MainWindow", "第三层货位地图"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.allMaps), _translate("MainWindow", "全景地图"))
        self.AGV_groupBox.setTitle(_translate("MainWindow", "车辆管理"))
        self.AGV_ID.setText(_translate("MainWindow", "AGV_ID"))
        self.random_addAGV.setText(_translate("MainWindow", "随机添加AGV"))
        self.AGV_location.setText(_translate("MainWindow", "起始位置"))
        self.addAGV.setText(_translate("MainWindow", "添加AGV"))
        self.Task_groupBox.setTitle(_translate("MainWindow", "任务管理"))
        self.end_label.setText(_translate("MainWindow", "终点"))
        self.random_AddTask.setText(_translate("MainWindow", "随机添加任务"))
        self.addTask.setText(_translate("MainWindow", "添加任务"))
        self.start_label.setText(_translate("MainWindow", "起点"))
        self.point_AGV_comboBox.setItemText(0, _translate("MainWindow", "AGV1"))
        self.point_AGV_comboBox.setItemText(1, _translate("MainWindow", "AGV2"))
        self.point_AGV.setText(_translate("MainWindow", "指定AGV"))
        self.test_path_planing.setTitle(_translate("MainWindow", "测试路径算法"))
        self.reset_Button.setText(_translate("MainWindow", "重置地图"))
        self.random_Add_Cases.setText(_translate("MainWindow", "随机添加案例"))
        self.Source_Button.setText(_translate("MainWindow", "Source"))
        self.Target_Button.setText(_translate("MainWindow", "Target"))
        self.point_algorithm_Qlable.setText(_translate("MainWindow", "指定算法"))
        self.point_algorithm_comboBox.setCurrentText(_translate("MainWindow", "Dijkstra"))
        self.point_algorithm_comboBox.setItemText(0, _translate("MainWindow", "Dijkstra"))
        self.point_algorithm_comboBox.setItemText(1, _translate("MainWindow", "Astar"))
        self.point_algorithm_comboBox.setItemText(2, _translate("MainWindow", "ATL_star"))
        self.point_algorithm_comboBox.setItemText(3, _translate("MainWindow", "self_Astar"))
        self.point_algorithm_comboBox.setItemText(4, _translate("MainWindow", "weight_ATL_star"))
        self.heuristic_Qlable.setText(_translate("MainWindow", "启发函数"))
        self.heuristic_ComboBox.setCurrentText(_translate("MainWindow", "欧几里得"))
        self.heuristic_ComboBox.setItemText(0, _translate("MainWindow", "欧几里得"))
        self.heuristic_ComboBox.setItemText(1, _translate("MainWindow", "曼哈顿"))
        self.heuristic_ComboBox.setItemText(2, _translate("MainWindow", "切比雪夫"))
        self.heuristic_ComboBox.setItemText(3, _translate("MainWindow", "平方欧几里得"))
        self.add_case.setText(_translate("MainWindow", "测试案例"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.open.setText(_translate("MainWindow", "open"))
        self.Save.setText(_translate("MainWindow", "Save"))

    #对比路径算法
    def handle_random_Add_Cases_button(self):
        try:
            floor = self.maps[self.tabWidget.currentIndex()]         # 获取当前地图
            canvas = self.floor_Canvas_list[self.tabWidget.currentIndex()]  # 获取当前地图的canvas
            source = self.source_spinBox.value()
            target = self.target_spinBox.value()
            # targets = [26,50,323,1399,1727,1748]  # 测试目标点
            heristic_name = self.heuristic_ComboBox.currentText()  # 获取启发函数名称
            heuristic_index = self.heuristic_ComboBox.currentIndex()  # 获取启发函数索引
            # algorithm_name = self.point_algorithm_comboBox.currentText()  # 获取算法名称
            # algorithm_index = self.point_algorithm_comboBox.currentIndex()  # 获取算法索引
            # algorithm_indices = [0, 1, 2, 3]                              # 算法索引
            algorithm_results = {}  # 存储每个算法的结果
            # 获取当前日期和时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # algorithm_results = self.path_planing.test_ATL_star(floor, source, target, heuristic_index)
            algorithm_results = self.path_planing.test_All_Nodes(floor, source, target, heuristic_index)
            # algorithm_count = self.point_algorithm_comboBox.count()         # 获取算法数量
            # for i in range(algorithm_count):                                # 遍历算法
            #     algorithm_name = self.point_algorithm_comboBox.itemText(i)
            #     take_time, path, explored, cost, turn_count = self.path_planing.Analyze_Path(floor, source, target, i, heuristic_index)
            #     canvas.show_visited_process(floor, explored)    # 显示探索过程
            #     # canvas.show_visited_process_slowly(floor, explored)  # 显示探索过程
            #     canvas.show_path(floor, path, i)  # 显示路径
            #     # canvas.save_image(source, target, algorithm_name, heristic_name)  # 保存图片
            #     # 存储结果
            #     algorithm_results[algorithm_name] = {
            #         'take_time': take_time,
            #         'explored': len(explored),
            #         'cost': cost,
            #         'turn_count': turn_count
            #     }
            #     log_message = (f"{current_time} - 路径算法测试案例：起点{source}, "
            #                    f"终点{target}, 算法：{algorithm_name},启发函数：{heristic_name}, "
            #                    # f"路径：{path}，"
            #                    f"最短距离：{cost}, 耗时：{take_time}"
                # # 将日志信息输出到文本框
                # self.textBrowser.append(log_message)
                # canvas.reset_canvas()  # 重置地图
            self.dialog = ResultsDialog(algorithm_results)
            self.textBrowser.append(f"{current_time} - 对比测试案例：起点{source}, 终点{target}。\n")
            for algorithm_name, result in algorithm_results.items():
                self.textBrowser.append(f'{algorithm_name} - 最短距离：{result["cost"]}, 耗时：{result["take_time"]}毫秒， 转向次数：{result["turn_count"]},探索节点数：{result["explored"]}。\n')
        except Exception as e:
            self.textBrowser.append(f"{current_time} - 对比测试案例失败: {e}\n")  # 错误处理，输出失败原因

    #测试路径算法的触发函数
    def handle_add_case_button(self):
        try:
            floor = self.maps[self.tabWidget.currentIndex()]         # 获取当前地图
            canvas = self.floor_Canvas_list[self.tabWidget.currentIndex()]  # 获取当前地图的canvas
            source = self.source_spinBox.value()
            target = self.target_spinBox.value()
            heuristic_index = self.heuristic_ComboBox.currentIndex()  # 获取启发函数索引
            heristic_name = self.heuristic_ComboBox.currentText()  # 获取启发函数名称
            algorithm_name = self.point_algorithm_comboBox.currentText()  # 获取算法名称
            algorithm_index = self.point_algorithm_comboBox.currentIndex()  # 获取算法索引
            # 获取当前日期和时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            take_time, path, explored, cost, turn_count = self.path_planing.Analyze_Path(floor, source, target, algorithm_index, heuristic_index)  # 运行路径算法
            canvas.show_visited_process(floor,explored)
            # canvas.show_visited_process_slowly(floor, explored)  # 显示探索过程
            canvas.show_path(floor,path,algorithm_index)    # 显示路径
            log_message = (f"{current_time} - 路径算法测试案例：起点{source}, "
                           f"终点{target}, 算法：{algorithm_name},启发函数：{heristic_name}, "
                           # f"路径：{path}，"
                           f"最短距离：{cost}, 耗时：{take_time}毫秒， 转向次数：{turn_count},探索节点数：{len(explored)}。\n")
                #将日志信息输出到文本框
            # canvas.save_image(source, target, algorithm_name,heristic_name)  # 保存图片
            self.textBrowser.append(log_message)
        except Exception as e:
            self.textBrowser.append(f"{current_time} - 路径算法测试案例失败: {e}")  # 错误处理，输出失败原因

    def handle_random_AddTask_button(self):
        print("随机添加任务")

    def handle_addTask_button(self):
        print("添加任务")

    def handle_random_addAGV_button(self):
        print("随机添加AGV")

    def handle_addAGV_button(self):
        print("添加AGV")

    def handle_point_AGV_comboBox(self):
        print("指定AGV")

    def handle_point_algorithm_comboBox(self):
        # print("指定算法")
        pass
    def handle_Source_Button(self):
        # floor = self.maps[self.tabWidget.currentIndex()]         # 获取当前地图
        canvas = self.floor_Canvas_list[self.tabWidget.currentIndex()]  # 获取当前地图的canvas
        current_node = canvas.highlighted_node
        if current_node is not None:
            print(f"current_node :{current_node}")
            self.source_spinBox.setValue(current_node)
        else:
            self.source_spinBox.clear() # 清除原有值

    def handle_Target_Button(self):
        canvas = self.floor_Canvas_list[self.tabWidget.currentIndex()]  # 获取当前地图的canvas
        current_node = canvas.highlighted_node
        if current_node is not None:
            print(f"current_node :{current_node}")
            self.target_spinBox.setValue(current_node)
        else:
            self.target_spinBox.clear() # 清除原有值
    #重置地图
    def handle_reset_Button(self):
        canvas = self.floor_Canvas_list[self.tabWidget.currentIndex()]  # 获取当前地图的canvas
        canvas.reset_canvas()  # 重置地图

    def handle_open_button(self):
        print("open")

    def handle_Save_button(self):
        print("Save")
        pass








