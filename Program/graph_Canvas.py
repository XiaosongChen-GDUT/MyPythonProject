import sys

from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import  QtGui, QtWidgets
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import os
#继承自FigureCanvas的类  嵌入PYQT5窗口中的地图的画布
class graph_FigureCanvas(FigureCanvas):
    def __init__(self,floor = None,title = None, parent=None, width=15, height=5, dpi=100):
        self.floor = floor
        self.title = title
        #第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)# 创建图形对象并配置其参数

        #第二步：在父类中激活Figure窗口
        super(graph_FigureCanvas, self).__init__(self.fig)# 初始化父类
        # self.fig.patch.set_facecolor('#01386a')  # 设置绘图区域颜色
        if 'members' in self.floor.graph:
            floors = self.floor.graph['members']
            # self.fig, self.axs = plt.subplots(1, len(floors), figsize=(15, 5))
            self.axs = self.fig.subplots(1, len(floors))
            self.ax_list =  self.axs.tolist()  # 将 numpy 数组转换为列表
        else:
            #     #第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
            self.ax = self.fig.add_subplot(111)# 添加子图到图形中
            # self.ax.spines['bottom'].set_position(('data', 0))  # 设置x轴线再Y轴0位置
            # self.ax.spines['left'].set_position(('data',0))  # 设置y轴在x轴0位置
            # self.ax.spines['top'].set_visible(False)  # 去掉上边框
            # self.ax.spines['right'].set_visible(False)  # 去掉右边框
            # self.ax.spines['bottom'].set_visible(False)  # 去掉下边框
            # self.ax.patch.set_facecolor("#01386a")  # 设置ax区域背景颜色
        self.setParent(parent) # 设置父窗口
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)# 设置大小策略为可扩展
        self.updateGeometry()# 更新几何形状
        self.fig.tight_layout()# 调整子图的布局

        # 定义退出标志
        self.is_running = True
        self.connect_event()  # 连接事件
        self.highlighted_node = None  # 高亮的节点

    def connect_event(self):
        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)  # 鼠标左键按下
        # self.mpl_connect('button_release_event', self.on_mouse_release)  # 鼠标左键释放
        # self.mpl_connect('motion_notify_event', self.on_mouse_move)  # 鼠标移动
        # self.mpl_connect("scroll_event", self.on_mouse_wheel)	#鼠标滚动事件
        self.cid2 =self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)  # 键盘按下

    #重写keyPressEvent()
    def keyPressEvent(self, event):
        key = event.key()
        print(f"keyPressEvent: {key}")
        if key == Qt.Key_Enter:
            self.keyPressEvent.emit('Enter')
        elif key == Qt.Key_Escape:
            self.keyPressEvent.emit('Escape')
        else:
            self.keyPressEvent.emit('')

    # 键盘按下事件
    def on_key_press(self, event):
        print(f"on_key_press: {event.key}")
        # if event.key == 'escape':
        #     self.is_running = False
        # elif event.key == 'enter':
        #     self.is_running = True

    # 鼠标左键按下高亮显示节点
    def on_mouse_press(self, event):
        if event.button == 1:  # 鼠标左键
            click_x, click_y = event.xdata, event.ydata  # 获取鼠标点击的坐标
            if click_x is None or click_y is None:
                return
            # 设置一个阈值，用于确定点击坐标与节点坐标之间的“接近度”
            threshold = 0.5
            clicked_node = None
            # 获取节点位置
            location = nx.get_node_attributes(self.floor, 'location')
            colors = nx.get_node_attributes(self.floor, 'node_colors')
            # 遍历节点位置字典
            for node, (x, y) in location.items():
                delta_x = abs(x - click_x)
                delta_y = abs(y - click_y)
                if delta_x < threshold and delta_y < threshold:
                    clicked_node = node
                    break
            if clicked_node == self.highlighted_node:  # 如果点击的节点和之前高亮的节点相同，则取消高亮
                print("选中节点与上一次相同！")
                return
            # 如果之前有高亮的节点，复原
            if self.highlighted_node is not None:
                print(f"1.复位前一次高亮的节点: {self.highlighted_node}")
                x_old, y_old = location[self.highlighted_node]
                self.ax.scatter(x_old, y_old, color=colors[self.highlighted_node], marker='o',edgecolors='black',linewidths=0.7,zorder=2)  # 恢复之前节点的颜色和大小
            self.draw()  ## 更新绘图
            # 高亮当前点击的节点
            if clicked_node is not None:
                self.highlighted_node = clicked_node
                x, y = location[clicked_node]
                print(f"2.本次点击的节点: {clicked_node}, 坐标: ({x}, {y})")
                self.ax.scatter(x, y, c='red', marker='o', edgecolors='black', linewidths=0.7, zorder=2)
                self.draw()  ## 更新绘图
            else:# 如果没有点击到节点，重置高亮节点
                self.highlighted_node = None
                self.draw()  ## 更新绘图

    #重置地图画布
    def reset_canvas(self):
        self.ax.cla()           # 清理绘图区
        # self.figure.clf()       # 清理画布，清理画布后必须重新添加绘图区
        self.draw_floor()
        self.fig.canvas.draw()  # 画布重绘
        self.fig.canvas.flush_events()  # 刷新事件

    '''显示搜索过程'''
    def show_visited_process(self,graph,explored,delay=0.01):
        # colors = nx.get_node_attributes(graph, 'node_colors')
        locations = [graph.nodes[node]['location'] for node in explored]  # 获取所有位置
        # original_colors = self.scatter_collection.get_facecolors()
        # node_list = list(graph.nodes())
        # gray_value = mcolors.to_rgba('gray')    # 灰色的 RGBA 值
        # for node in explored:
        #     # start_time = time.time()
        #     if not self.is_running:
        #         print("*** Escape key pressed,show_visited_process --> stopping animation")
        #         break  # 按下 Escape 键后终止
        #     node_index = node_list.index(node)  # 获取 node 在节点列表中的索引
        #     original_colors[node_index] = gray_value  # 更新颜色
        #     self.scatter_collection.set_facecolor(original_colors)
        #     # self.fig.canvas.draw()  # 重绘图形
        #     self.fig.canvas.draw_idle()  # 画布渲染
        #     self.fig.canvas.flush_events()  # 刷新事件
        # print(f"已走过{node}，耗时{time.time()-start_time}秒") #0.25秒左右浮动，越来越慢
        explored_x = [loc[0] for loc in locations]  # 提取 x 坐标
        explored_y = [loc[1] for loc in locations]  # 提取 y 坐标

        self.ax.scatter(explored_x, explored_y, color='cyan',marker='o',edgecolors='black',  linewidth=0.7, zorder=2)  # 只画出路径点

    #显示搜索过程，慢慢显示
    def show_visited_process_slowly(self,graph, explored):
        try:
            explored = list(explored)  # 转换为列表
            # 获取原始颜色并保存
            # original_colors = self.scatter_collection.get_facecolors()
            # node_list = list(graph.nodes())
            # 记录当前已访问的节点索引
            current_index = 0
            gray_value = mcolors.to_rgba('gray')  # 灰色的 RGBA 值
            # 定义更新函数
            def update(frame):
                nonlocal current_index          # 使用外部变量
                if current_index < len(explored):
                    node = explored[current_index]  # 当前要显示的节点
                    self.scatter_collection.get_facecolor()[node] = gray_value  # 更新为访问色
                    current_index += 1          # 更新索引
                # 完成后停止动画
                if current_index >= len(explored):
                    ani.event_source.stop()     # 停止动画
                # 返回当前更新的散点对象
                return [self.scatter_collection]

            # 创建 FuncAnimation  Bltitting 允许您仅重绘已更改的艺术对象，而不是整个图形，从而提高绘制效率。
            # 1.确保您的 FuncAnimation 使用 blit=True：这将启用 Blitting。
            # 2.确保您的 update 函数返回要更新的艺术对象：在 update 函数中，确保返回更新的艺术对象。
            ani = FuncAnimation(self.fig, update, frames=len(explored), interval=10, repeat=False, blit=True)
        except Exception as e:
            print(f"show_visited_process_slowly error: {e}")

    #高亮显示路径
    def show_path(self,graph, path,algorithm_index):
        try:
            locations = [graph.nodes[node]['location'] for node in path]  # 获取所有位置
            path_x = [loc[0] for loc in locations]  # 提取 x 坐标
            path_y = [loc[1] for loc in locations]  # 提取 y 坐标
            color = 'gray'
            if algorithm_index == 0:#dijkstra算法
                color = 'blue'
            elif algorithm_index == 1:#A*算法
                color = 'green'
            # elif algorithm_index == 2:# 双向Dijkstra
            #     color = 'purple'
            elif algorithm_index == 2:#ATL算法
                color = 'red'
            elif algorithm_index == 3:#改进的A*算法
                color = 'orange'
                # color = 'pink'
                # color = 'brown'
                # color = 'cyan'
            # self.ax.scatter(path_x, path_y, color=color,marker='o',edgecolors='black',  linewidth=0.7, zorder=2)  # 只画出路径点
            self.ax.plot(path_x, path_y, color=color, marker='o', linewidth=0.7, zorder=2)  # 画出路径的边和点
            self.draw()  # 更新绘图
        except Exception as e:
            print(f"show_path error: {e}")

    def show_path_with_color(self,graph, path,color='gray',name=''):
        try:
            locations = [graph.nodes[node]['location'] for node in path]  # 获取所有位置
            path_x = [loc[0] for loc in locations]  # 提取 x 坐标
            path_y = [loc[1] for loc in locations]  # 提取 y 坐标
            color = 'gray'
            color = 'orange'
            # color = 'pink'
            # color = 'brown'
            # color = 'cyan'
            # self.ax.scatter(path_x, path_y, color=color,marker='o',edgecolors='black',  linewidth=0.7, zorder=2)  # 只画出路径点
            self.ax.plot(path_x, path_y, color=color, marker='o', linewidth=0.7, zorder=2)  # 画出路径的边和点
            self.ax.set_title(name)
            self.draw()  # 更新绘图
        except Exception as e:
            print(f"show_path error: {e}")

        # 鼠标左键释放
    # def on_mouse_release(self, event):
    #     pass
    #     # if event.button == 1:  # 鼠标左键
    #     #     self.lef_mouse_pressed = False
    #     #     print(f"on_mouse_release鼠标位置: ({event.x}, {event.y})")
    # # 鼠标移动
    # def on_mouse_move(self, event):
    #     pass
    # # 鼠标滚动事件
    # def on_mouse_wheel(self, event):
    #     pass
    #     # # 鼠标滚动事件
    #     # if event.button == 'up':
    #     #     print(f"on_mouse_wheel鼠标滚动: 放大")
    #     #     # self.ax.set_xlim(self.ax.get_xlim() * 1.1)
    #     #     # self.ax.set_ylim(self.ax.get_ylim() * 1.1)
    #     # elif event.button == 'down':
    #     #     print(f"on_mouse_wheel鼠标滚动: 缩小")
    #     #     # self.ax.set_xlim(self.ax.get_xlim() * 0.9)
    #     #     # self.ax.set_ylim(self.ax.get_ylim() * 0.9)
    #     # self.draw()  # 重绘图形

    #绘制地图
    def draw_floor(self):
        start_time = time.time()
        if 'members' in self.floor.graph:
            members = self.floor.graph['members']
            # self.draw_floors(members, [f"Floor {i}" for i in range(1, len(members)+1)])
            # start_time = time.time()
            # 绘制每个子图
            for ax, graph, title in zip(self.ax_list, members, [f"Floor {i}" for i in range(1, len(members)+1)]):
                # 获取节点位置和颜色
                colors = nx.get_node_attributes(graph, 'node_colors')
                location = nx.get_node_attributes(graph, 'location')
                pos = nx.get_node_attributes(graph, 'pos')
                # 提取 X, Y 画布坐标
                x = [loc[0] for loc in location.values()]
                y = [loc[1] for loc in location.values()]
                ax.set_title(title)
                # 绘制边
                for edge in graph.edges():
                    x_edges = [location[edge[0]][0], location[edge[1]][0]]
                    y_edges = [location[edge[0]][1], location[edge[1]][1]]
                    ax.plot(x_edges, y_edges, c='gray',zorder=1)
                ax.scatter(x, y, c=[colors[node] for node in graph.nodes()], marker='o',edgecolors='black',linewidths=0.7,zorder=1)
                ax.set_xlabel('排 坐标')
                ax.set_ylabel('列 坐标')

            self.fig.tight_layout()#调整子图间距
            self.draw()#更新绘图内容
            end_time = time.time()
            print(f"绘制全景地图耗时：{end_time-start_time}秒")
        else:
            # 获取节点位置和颜色
            pos = nx.get_node_attributes(self.floor, 'pos')
            colors = nx.get_node_attributes(self.floor, 'node_colors')
            location = nx.get_node_attributes(self.floor, 'location')
            # 提取 X, Y 画布坐标
            x, y = zip(*location.values())
            # x = [loc[0] for loc in location.values()]
            # y = [loc[1] for loc in location.values()]
            self.ax.set_title(self.title)
            # # 绘制边
            for edge in self.floor.edges():
                x_edges = [location[edge[0]][0], location[edge[1]][0]]
                y_edges = [location[edge[0]][1], location[edge[1]][1]]
                self.ax.plot(x_edges, y_edges, c='gray',zorder=1)
                # 获取边的权重值（假设可以通过 edge_weight 方法获取）
                # edge_weight = self.floor.edges[edge]['weight']  # 直接获取权重值
                # self.ax.text((x_edges[0] + x_edges[1]) / 2, (y_edges[0] + y_edges[1]) / 2, edge_weight, ha='center',
                #              va='center',
                #              color='black',
                #              zorder=2)
            # 绘制散点
            #self.scatter_collection包含了散点图的所有信息，包括颜色、标记、边缘颜色等。
            self.scatter_collection = self.ax.scatter(x, y, c=[colors[node] for node in self.floor.nodes()],
                                                      marker='o',edgecolors='black',linewidths=0.7,zorder=1)
            # self.scatter_collection.set_facecolor(colors.values())      #重置地图时需要set_facecolor
            # 创建图例元素
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='black',linewidth=1,  markerfacecolor='yellow', markersize=8, label='巷道节点'),
                plt.Line2D([0], [0], marker='s', color='black', linewidth=1, markerfacecolor='yellow',  markersize=8, label='入库节点'),
                plt.Line2D([0], [0], marker='o', color='black', linewidth=1, markerfacecolor='lightblue',  markersize=8, label='货位节点')
            ]
            #添加图例
            self.ax.legend(handles=legend_elements, loc='upper left',ncol=3, bbox_to_anchor=(0.01, 1.01), borderaxespad=0)
            self.ax.set_xlabel('排 坐标')
            self.ax.set_ylabel('列 坐标')
            plt.tight_layout()#调整子图间距
            self.draw()#更新绘图内容
            end_time = time.time()
            print(f"绘制{self.title}地图耗时：{end_time-start_time}秒")

    #保存图片
    def save_image(self, start, end, algorithm,heristic_name,folder="pics"):
        # 获取当前文件的目录
        current_directory = os.path.dirname(os.path.abspath(__file__))
        # print(f"当前文件目录：{current_directory}")                #D:\Application_Data\MyPythonProject\Program
        # 获取当前文件的父目录
        parent_directory = os.path.dirname(current_directory)     #D:\Application_Data\MyPythonProject
        # 拼接出完整的文件夹路径
        folder_path = os.path.join(parent_directory, folder)
        # 检查pics文件夹是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            print(f"保存图像{start}-{end}-{algorithm}-{heristic_name}.png，文件夹{folder_path}不存在，创建文件夹！")
            os.makedirs(folder_path)
        filename = f"{start}-{end}-{algorithm}-{heristic_name}.png"
        file_path = os.path.join(folder_path, filename)
        # 生成文件名
        # file_path = "pics/" + filename
        self.ax.set_title(f"{start} -> {end}:Algorithm:{algorithm} ")
        self.fig.savefig(file_path, dpi=300)
        print(f"保存图片成功！{file_path}")

    # def draw_floors(self, floors, titles):#绘制多个地图
    #     start_time = time.time()
    #     # 创建多个子图
    #     # self.fig, self.axs = self.fig.subplots(1, len(floors), figsize=(15, 5))
    #     # self.axs = self.fig.subplots(1, len(floors))
    #     # 绘制每个子图
    #     for ax, graph, title in zip(self.axs, floors, titles):
    #         # 获取节点位置和颜色
    #         pos = nx.get_node_attributes(graph, 'pos')
    #         colors = nx.get_node_attributes(graph, 'node_colors')
    #         location = nx.get_node_attributes(graph, 'location')
    #         # 提取 X, Y 画布坐标
    #         x = [loc[0] for loc in location.values()]
    #         y = [loc[1] for loc in location.values()]
    #         ax.set_title(title)
    #         # 绘制边
    #         for edge in graph.edges():
    #             x_edges = [location[edge[0]][0], location[edge[1]][0]]
    #             y_edges = [location[edge[0]][1], location[edge[1]][1]]
    #             ax.plot(x_edges, y_edges, c='gray',zorder=1)
    #         ax.scatter(x, y, c=[colors[node] for node in graph.nodes()], marker='o',edgecolors='black',linewidths=0.7,zorder=1)
    #         ax.set_xlabel('排 坐标')
    #         ax.set_ylabel('列 坐标')
    #
    #     self.fig.tight_layout()#调整子图间距
    #     self.draw()#更新绘图内容
    #     end_time = time.time()
    #     print(f"绘制全景地图耗时：{end_time-start_time}秒")

#结果展示对话框
class ResultsDialog(QWidget):#QDialog
    def __init__(self, algorithm_results):
        # super(ResultsDialog, self).__init__()
        super().__init__()
        self.setWindowTitle("算法结果对比展示")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QtWidgets.QVBoxLayout()
        self.results = algorithm_results
        # 创建 matplotlib 图形
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

        self.plot_results(algorithm_results)  # 绘制结果
        self.show()  # 显示对话框

    def plot_results(self, algorithm_results):
        # 获取数据
        algorithm_names = list(algorithm_results.keys())    #横坐标
        take_times = [result['take_time'] for result in algorithm_results.values()]
        costs = [result['cost'] for result in algorithm_results.values()]
        turn_counts = [result['turn_count'] for result in algorithm_results.values()]
        explored_counts = [result['explored'] for result in algorithm_results.values()]
        # for name, result in algorithm_results.items():

        # 创建子图
        ax1 = self.figure.add_subplot(221)
        # ax1.bar(algorithm_names, take_times, color='b') # 绘制柱状图
        ax1.plot(algorithm_names, take_times, color='b', marker='o') # 绘制柱状图
        # 为ax1的每个点添加注释
        for i, txt in enumerate(take_times):
            ax1.annotate(txt, (algorithm_names[i], take_times[i]), textcoords="offset points", xytext=(0,10), ha='center')
        ax1.set_title('算法耗时')
        ax1.set_ylabel('时间（秒）')
        # ax1.set_xlabel('算法种类')

        ax2 = self.figure.add_subplot(222)
        # ax2.bar(algorithm_names, costs, color='g')
        ax2.plot(algorithm_names, costs, color='g', marker='o')
        # 为ax2的每个点添加注释
        for i, txt in enumerate(costs):
            ax2.annotate(txt, (algorithm_names[i], costs[i]), textcoords="offset points", xytext=(0,10), ha='center')
        ax2.set_title('最短距离')
        ax2.set_ylabel('距离')
        # ax2.set_xlabel('算法种类')

        ax3 = self.figure.add_subplot(223)
        # ax3.bar(algorithm_names, turn_counts, color='r')
        ax3.plot(algorithm_names, turn_counts, color='r', marker='o')
        for i, txt in enumerate(turn_counts):
            ax3.annotate(txt, (algorithm_names[i], turn_counts[i]), textcoords="offset points", xytext=(0,10), ha='center')
        ax3.set_title('转向次数')
        ax3.set_ylabel('次数')
        # ax3.set_xlabel('算法种类')

        ax4 = self.figure.add_subplot(224)
        # ax4.bar(algorithm_names, explored_counts, color='y')
        ax4.plot(algorithm_names, explored_counts, color='y', marker='o')
        for i, txt in enumerate(explored_counts):
            ax4.annotate(txt, (algorithm_names[i], explored_counts[i]), textcoords="offset points", xytext=(0,10), ha='center')
        ax4.set_title('探索节点数')
        ax4.set_ylabel('节点数')
        # ax4.set_xlabel('算法种类')

        self.canvas.draw()
