import networkx as nx
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import time


class MapView(tk.Tk,object):

    # 创建一个空的有向图
    DG = nx.Graph()
    nodes_data = []   # 用于存储节点数据
    edges_data = []   # 用于存储边数据
    pos = {}          # 用于存储节点位置
    node_colors = []  # 用于存储节点颜色
    window_width = 100
    window_height = 50
    def __init__(self,DG,Env,Controller):
        super(MapView, self).__init__()  # 初始化父类

        self.geometry("1080x800+100+100")   # 设置窗口大小和在屏幕上的位置
        self.pos = DG.graph['pos']
        self.node_colors = DG.graph['node_colors']
        self.DG = DG
        self.Env = Env
        self.Controller = Controller

        start_time = time.time()  # 记录绘图时间
        # 创建matplotlib图形
        self.figure, self.ax = plt.subplots(facecolor='gray')  # 设置图形大小
        plt.ion()  # 开启交互模式
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)# 中间件映射
        # self.ax.set_background_color('lightgreen')  # 设置背景颜色
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)# 获取到中间件上的图，并且pack到UI组件root上。

        #matplotlib的导航工具栏显示上来(默认是不会显示它的)
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.config(background='lightblue')
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=0)

        # 添加输入框和标签
        tk.Label(self, text="起点：", font=("Arial", 12), width=6, height=1, bg='lightblue').place(x=100, y=700)
        self.start_entry = tk.Entry(self, font=("Arial", 12), width=6, bg='white')
        self.start_entry.place(x=170, y=700)
        tk.Label(self, text="终点：", font=("Arial", 12), width=6, height=1, bg='lightblue').place(x=100, y=730)
        self.end_entry = tk.Entry(self, font=("Arial", 12), width=6, bg='white')
        self.end_entry.place(x=170, y=730)
        #添加“起点"标签
        tk.Label(self, text="AGV起点：", font=("Arial", 12), width=10, height=1, bg='lightblue').place(x=250, y=700)
        self.agv_start_entry = tk.Entry(self, font=("Arial", 12), width=6, bg='white')
        self.agv_start_entry.place(x=350, y=700)
        #添加“node"标签
        tk.Label(self, text="AGV_ID：", font=("Arial", 12), width=10, height=1, bg='lightblue').place(x=250, y=730)
        self.agv_id_entry = tk.Entry(self, font=("Arial", 12), width=6, bg='white')
        self.agv_id_entry.place(x=350, y=730)
        # 添加“add_agv”按钮
        self.add_agv_button = tk.Button(self, text="添加AGV",background="lightyellow", command=self.on_add_agv).place(x=350, y=760)
        # 添加“start”按钮
        self.start_button = tk.Button(self, text="开始",background="lightyellow", command=self.on_start).place(x=100, y=760)
        # 添加“clear”按钮
        self.clear_button = tk.Button(self, text="清除",background="lightgreen", command=self.on_clear).place(x=170, y=760)

        self.draw_graph()  # 绘制图形

        print("---绘图耗时： %s  ---" % (time.time() - start_time))

    # 定义add_agv按钮事件处理函数
    def on_add_agv(self):
        if(self.agv_start_entry.get()=="" or self.agv_start_entry.get().isdigit() == False):
            print("请输入正确的AGV起点！")
            return
        if(self.agv_id_entry.get()=="" or self.agv_id_entry.get().isdigit() == False):
            print("请输入正确的AGV_ID！")
            return
        # 获取AGV起点数据
        agv_start_point = int(self.agv_start_entry.get())
        agv_id = int(self.agv_id_entry.get())
        if(not self.DG.has_node(agv_start_point)):
            print("AGV起点不存在！")
            return
        else:
            if(self.Controller.Add_AGV(agv_id,agv_start_point) == True):
                print(f"添加AGV成功！AGV起点: {agv_start_point},AGV_ID: {agv_id}")
                # 重新绘制图形
                # nx.draw_networkx_nodes(self.DG, self.pos, [agv_start_point], node_size=80, alpha=0.8, node_color='red')


    # 定义start按钮事件处理函数
    def on_start(self):
        # 获取起点和终点数据
        # 如果不是None，则获取其内容
        if self.start_entry.get().isdigit() and self.end_entry.get().isdigit():
            start_point = self.start_entry.get()
            end_point = self.end_entry.get()
            print(f"起点: {start_point}, 终点: {end_point}")
        else:
            print("输入框未正确初始化！")
            start_point = None
            end_point = None
            return
        start_point = int(self.start_entry.get())  # 尝试将起点转换为整数
        end_point = int(self.end_entry.get())      # 尝试将终点转换为整数

        if(not self.DG.has_node(start_point)):
            print("起点不存在！")
        elif(not self.DG.has_node(end_point)):
            print("终点不存在！")
        elif(self.DG.has_node(start_point) and self.DG.has_node(end_point)):
            path = nx.shortest_path(self.DG, source=start_point, target=end_point)
            print(f"最短路径: {path}")
            for node in path:
                # 获取节点的坐标
                # x, y = self.pos[node]
                nx.draw_networkx_nodes(self.DG, self.pos, nodelist=[node], node_size=80, alpha=0.8, node_color='lightgreen')
                # plt.pause(0.5)  # 延迟，以便图形更新

    def on_clear(self):
        # 清除输入框内容
        self.start_entry.delete(0, tk.END)
        self.end_entry.delete(0, tk.END)
        self.agv_start_entry.delete(0, tk.END)
        # 重新绘制图
        self.ax.clear()  # 清除之前的绘图
        # nx.draw_networkx_nodes(self.DG, self.pos, node_size=80, alpha=0.8, node_color=self.node_colors)
        self.draw_graph()  # 重新绘制图形

    #打开指定的图片文件，缩放至指定尺寸
    def get_image(filename,width,height):
        im = tk.Image.open(filename).resize((width, height))
        return tk.ImageTk.PhotoImage(im)

    def update_graph(self):
        # 更新图形
        self.ax.clear()  # 清除之前的绘图
        nx._clear_cache()
        self.draw_graph()
        self.canvas.draw()  # 重新绘制图形
        # 每1000毫秒后再次调用此函数
        # self.after(1000, self.update_graph)

    def draw_graph(self):
        # 画图
        nx.draw(self.DG,self.pos,with_labels=True, node_size=80, alpha=0.8, node_color=self.node_colors,
                edge_color='lightgray', font_size=6)#

        # 使用更高效的函数绘制图
        # nx.draw_networkx_nodes(self.DG, self.pos, node_size=80, alpha=0.8, node_color=self.node_colors)
        # nx.draw_networkx_edges(self.DG, self.pos, edge_color='lightgray', alpha=0.5)
        # nx.draw_networkx_labels(self.DG, self.pos, font_size=6, font_color='black')
        # plt.title('Topology Map')
        #在默认情况下，Matplotlib 的绘图操作是阻塞的，
        # 即在调用 plt.show()之前，代码会一直阻塞在绘图函数处，直到手动关闭图形窗口后才会继续执行后续代码。
        # 这种模式适用于静态图形展示，但不适合需要实时更新图形的交互式应用场景。
        # plt.ion()

        # plt.show()#False交互式显示， 不阻塞程序继续运行

    def draw_Paths(self,path):
        # 画出最短路径
        nx.draw_networkx_nodes(self.DG, self.pos, nodelist=path,  node_size=80, alpha=0.8,node_color='red')
        # plt.draw()  # 更新图形
        plt.plot()

