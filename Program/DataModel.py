import time
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import matplotlib
from matplotlib import image as mpimg
from matplotlib.animation import FuncAnimation

matplotlib.rcParams['font.sans-serif'] = ['SimHei'] #font.sans-serif参数来指定"SimHei"字体
matplotlib.rcParams['axes.unicode_minus'] = False	#axes.unicode_minus参数用于显示负号

row_space = 1.54  #排间距
to_second_Floor = 3.15                      #1楼换层到2楼的高度
to_third_Floor = 2.55                       #2楼换层到3楼的高度
enter_point = [51,348,636,925,1110,1620]    #入口点
out_point = [445,820,971,1156]  #出口点
fist_connect_point = [642,1116,674,1148]        #1楼提升机接驳点
second_connect_point = [2374,2844,2406,2876]    #2楼接驳点
third_connect_point = [3899,4135]               #3楼接驳点
connect_points = [642,1116,674,1148,2374,2844,2406,2876,3899,4135]#所有接驳点
#1501、1540定位空托盘出口点  暂时不考虑空托盘入口点，定位货位点！
pallet_out = [1501,1540]
# file_path = '../Resource/WMSGraph_11_6.xlsx'
file_path = '../Resource/WMSGraph_12_9.xlsx'
class Model:
    def __init__(self):
        self.combined_graph = nx.Graph()
        #读取数据并创建地图
        self.combined_graph, self.floor1, self.floor2, self.floor3=self.create_Combined_Graph()
        #添加地图属性
        self.combined_graph.graph['members']=[self.floor1, self.floor2, self.floor3]

    def findPath(self, start, end):
        path = nx.dijkstra_path(self.combined_graph, start, end)
        return path

    def create_Combined_Graph(self):#创建合并的地图
        #读取数据并创建地图
        data = pd.read_excel(file_path)
        floor1 = self.read_map(data, 1)
        floor2 = self.read_map(data, 2)
        floor3 = self.read_map(data, 3)
        self.combined_graph = nx.compose_all([floor1, floor2, floor3])

        #1楼提升机接驳点
        first_node_A =  self.get_Node_BY_Attribute(floor1, 'pos',(14,10,1))#642
        first_node_B =  self.get_Node_BY_Attribute(floor1, 'pos',(24,10,1))#1116
        first_node_C =  self.get_Node_BY_Attribute(floor1, 'pos',(14,42,1))#674
        first_node_D =  self.get_Node_BY_Attribute(floor1, 'pos',(24,42,1))#1148
        #添加地图属性
        floor1.graph['connect_point']= [first_node_A, first_node_B, first_node_C, first_node_D]
        floor1.graph['enter_point'] = enter_point
        floor1.graph['out_point'] = out_point

        # floor1.graph['pallet_out'] = pallet_out
        #2楼接驳点
        second_node_A =  self.get_Node_BY_Attribute(floor2, 'pos',(14,10,2))#2374
        second_node_B =  self.get_Node_BY_Attribute(floor2, 'pos',(24,10,2))#2844
        second_node_C =  self.get_Node_BY_Attribute(floor2, 'pos',(14,42,2))#2406
        second_node_D =  self.get_Node_BY_Attribute(floor2, 'pos',(24,42,2))#2876
        floor2.graph['connect_point'] = [second_node_A, second_node_B, second_node_C, second_node_D]
        #3楼接驳点
        third_node_A =  self.get_Node_BY_Attribute(floor3, 'pos',(14,10,3))#3899
        third_node_B =  self.get_Node_BY_Attribute(floor3, 'pos',(24,10,3))#4135
        floor3.graph['connect_point'] = [third_node_A, third_node_B]
        # 连接1-2楼接驳点
        self.combined_graph.add_edge(first_node_A, second_node_A,weight=to_second_Floor)
        self.combined_graph.add_edge(first_node_B, second_node_B,weight=to_second_Floor)
        self.combined_graph.add_edge(first_node_C, second_node_C,weight=to_second_Floor)
        self.combined_graph.add_edge(first_node_D, second_node_D,weight=to_second_Floor)
        # 连接2-3楼接驳点
        self.combined_graph.add_edge(second_node_A, third_node_A,weight=to_third_Floor)
        self.combined_graph.add_edge(second_node_B, third_node_B,weight=to_third_Floor)

        return self.combined_graph, floor1, floor2, floor3

    def draw_3D_map(self, combined_graph, title='三维地图'):#绘制三维地图
        # 合并地图属性
        # self.combined_graph.graph['pos'] = {**floor1.graph['pos'], **floor2.graph['pos'], **floor3.graph['pos']}#解包操作符 **。这个操作符使得能够将多个字典中的键值对合并为一个新的字典。
        # self.combined_graph.graph['node_colors']={**floor1.graph['node_colors'], **floor2.graph['node_colors'], **floor3.graph['node_colors']}
        # self.combined_graph.graph['location']={**floor1.graph['location'], **floor2.graph['location'], **floor3.graph['location']}
        # self.combined_graph.graph['id']={**floor1.graph['id'], **floor2.graph['id'], **floor3.graph['id']}
        # self.combined_graph.graph['status']={**floor1.graph['status'], **floor2.graph['status'], **floor3.graph['status']}
        # self.combined_graph.graph['dimension']={**floor1.graph['dimension'], **floor2.graph['dimension'], **floor3.graph['dimension']}
        #连接换层点_graph,title='三维地图'):
        pos = nx.get_node_attributes(combined_graph, 'pos')  # 假设节点位置包含在节点属性 'pos' 中
        colors = nx.get_node_attributes(combined_graph, 'node_colors')  # 假设节点颜色包含在节点属性 'node_colors' 中
        # print(colors)
        colors_list = [colors[node] for node in combined_graph.nodes()]  # 节点颜色列表
        # 提取 X, Y, Z 坐标
        x = [pos[node][0] for node in combined_graph.nodes()]
        y = [pos[node][1] for node in combined_graph.nodes()]
        z = [pos[node][2] for node in combined_graph.nodes()]
        # 创建 3D 图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 绘制节点
        ax.scatter(x, y, z, c=colors_list, marker='o')
        # 绘制边
        for edge in combined_graph.edges():
            x_edges = [pos[edge[0]][0], pos[edge[1]][0]]
            y_edges = [pos[edge[0]][1], pos[edge[1]][1]]
            z_edges = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x_edges, y_edges, z_edges, c='b')
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_zlabel('Z 坐标')
        ax.set_title(title)
        plt.show()

    def draw_floors(self, floors, titles):#绘制多个地图
        start_time = time.time()
        # 创建多个子图
        fig, axs = plt.subplots(1, len(floors), figsize=(15, 5))

        # 绘制每个子图
        for ax, graph, title in zip(axs, floors, titles):
            # 获取节点位置和颜色
            pos = nx.get_node_attributes(graph, 'pos')
            colors = nx.get_node_attributes(graph, 'node_colors')
            location = nx.get_node_attributes(graph, 'location')
            # 提取 X, Y 画布坐标
            x = [loc[0] for loc in location.values()]
            y = [loc[1] for loc in location.values()]
            ax.set_title(title)
            # 绘制边
            for edge in graph.edges():
                x_edges = [location[edge[0]][0], location[edge[1]][0]]
                y_edges = [location[edge[0]][1], location[edge[1]][1]]
                ax.plot(x_edges, y_edges, c='gray',zorder=1)
            ax.scatter(x, y, c=[colors[node] for node in graph.nodes()], marker='o',edgecolors='black',linewidths=0.7,zorder=2)
            ax.set_xlabel('排 坐标')
            ax.set_ylabel('列 坐标')
        plt.tight_layout()#调整子图间距
        end_time = time.time()
        print(f"绘制{len(floors)}层地图耗时：{end_time-start_time}秒")
        plt.show()

    # 定义一个函数，用于获取指定属性值的节点
    def get_Node_BY_Attribute(self,graph,attr_name, expected_value):
        # print(f"attr_name:{attr_name}, expected_value:{expected_value}")
        # print(f"graph 类型: {type(graph)}")#graph 类型: <class 'tuple'>
        for node, attributes in graph.nodes(True):
            # 检查指定的属性名并与值进行匹配
            if attr_name in attributes and attributes[attr_name] == expected_value:
                # print(f"node:{node}, attributes:{attributes}")
                return node
        return None

    def draw_floor(self,graph, title):#绘制单层地图
        # 获取节点位置和颜色
        pos = nx.get_node_attributes(graph, 'pos')
        colors = nx.get_node_attributes(graph, 'node_colors')
        location = nx.get_node_attributes(graph, 'location')#{1: (0, 0), 2: (0, 1.5), 3: (0, 3.0)}
        # 提取 X, Y 坐标
        x=[location[node][0] for node in graph.nodes()]
        y=[location[node][1] for node in graph.nodes()]
        # 创建图形
        plt.figure()
        plt.title(title)

        # 绘制边
        for edge in graph.edges():
            x_edges = [location[edge[0]][0], location[edge[1]][0]]
            y_edges = [location[edge[0]][1], location[edge[1]][1]]
            plt.plot(x_edges, y_edges, c='gray',zorder=1)# # 绘制边，设置 zorder 较小使其在下面
            # 绘制散点图
        plt.scatter(x, y, c=[colors[node] for node in graph.nodes()], marker='o',edgecolors='black',linewidths=0.7,zorder=2)

        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.grid()
        plt.show()

    def read_map(self,Data,layer):#读取单层地图数据
        floor = nx.Graph()#创建无向图
        node_colors = {}  # 用于存储节点颜色
        node_markers = {}  # 用于存储节点标记
        pos = {}  # 用于存储节点位置
        nodes_data = []  # 用于存储节点数据
        roadway_count = 0  # 巷道计数
        freight_count = 0  # 货位计数
        location = {}  # 存储节点在画布中的位置
        start_time = time.time()
        if layer == 1:
            start_colum = 0
        elif layer == 2:
            start_colum = 8
        elif layer == 3:
            start_colum = 16
        else:
            print("输入层数错误！")
            return None
        try:
            num=Data.iloc[:, start_colum].dropna().astype(int)  # 读取第一列数据，删除空值，转为整数
            id = Data.iloc[:, start_colum+1].dropna()  # 读取第二列数据，删除空值
            status = Data.iloc[:, start_colum+2].dropna().astype(int)  # 读取第三列数据，删除空值
            dimension = Data.iloc[:len(num), start_colum+3]  # 读取第四列数据
            # 读取起点
            start_layer1=Data.iloc[:, start_colum+4].dropna().astype(int)  # 读取第五列数据，删除空值
            # 读取终点
            end_layer1=Data.iloc[:, start_colum+5].dropna().astype(int)  # 读取第六列数据，删除空值
            # 读取距离
            distance=Data.iloc[:, start_colum+6].dropna().astype(float).round(2)  # 读取第七列数据，删除空值,将距离转为浮点数并四舍五入
            # 读取地图数据
            if len(num) == len(id) == len(status) == len(dimension):
                for n,i,s,d in zip(num,id,status,dimension):
                    i=str(i)    #将ID转为字符串
                    parts = i.split('-')
                    if parts[0] != 'm':
                        row, col, layer = map(int, parts)#将parts中的元素分别转为整型并赋值给row,col,layer
                    else:
                        row, col, layer = map(int, parts[1:])
                    pos[n] = (row, col,layer)  #使用排、列、层作为位置
                    # 创建节点属性字典
                    node_attr = {
                        'id': i,
                        'status': s,    # -1表示通道，0表示货位
                        'dimension': d  # 节点货位的尺寸类型
                    }
                    # 根据状态设置节点颜色
                    if s == -1:
                        if n in enter_point:#入口点
                            node_colors[n] = 'lightgreen'
                            node_markers[n] = 's' # 正方形
                        elif n in out_point:#出口点
                            node_colors[n] = 'darkorange'
                            node_markers[n] = 's'
                        elif n in connect_points:#提升机接驳点
                            node_colors[n] = 'darkorange'
                            node_markers[n] = 'D' # 菱形
                        else:
                            node_colors[n] = '#FFFF00' #'#BDAF00' #'yellow'  # 通道用黄色显示
                            node_markers[n] = 'o' # 圆形
                            roadway_count += 1
                    else:
                        node_colors[n] = 'lightblue'  # 货位用蓝色、圆形显示
                        node_markers[n] = 'o' # 圆形
                        freight_count += 1
                    nodes_data.append((n, node_attr))    # 节点属性字典加入列表
            else:
                print(f"第{layer}层地图数据len(num) ={len(num)} "
                      f" len(id) ={len(id)} ,len(status) = {len(status)}, len(dimension) = {len(dimension)}长度不一致！")
                return None
            #添加节点
            floor.add_nodes_from(nodes_data)
            # 添加边，设置权重
            for start, end, weight in zip(start_layer1, end_layer1, distance):
                floor.add_edge(start, end, weight=weight)
            # 添加节点在画布中的位置
            start_node = min(floor.nodes())#获取起点
            location[start_node] = (0, 0)  # 画布起点位置，默认坐标设为 (0, 0)
            # location[start_node] = (0, 100)  # 画布起点位置，默认坐标设为 (0, 0)
            queue = deque([start_node])#将起点加入队列
            while queue:
                current_node = queue.popleft()#弹出队首元素
                for neighbor in floor.neighbors(current_node):#遍历当前节点的邻居
                    if neighbor not in location:#邻居不在location中
                        queue.append(neighbor)#将邻居加入队列
                        weight_value = round(floor[current_node][neighbor]['weight'], 2)#获取边的权重
                        current_node_row =  round(location[current_node][0], 2) # 当前节点的 排 坐标
                        current_node_column =  round(location[current_node][1], 2) # 当前节点的 列 坐标

                        # if pos[neighbor][0] > pos[current_node][0]:#判断X轴方向
                        #     location[neighbor] = (round(location[current_node][0] + weight_value, 2), location[current_node][1])
                        # elif pos[neighbor][0] < pos[current_node][0]:#判断X轴方向
                        #     location[neighbor] = (round(location[current_node][0] - weight_value,2), location[current_node][1])
                        # elif pos[neighbor][1] > pos[current_node][1]:#判断Y轴方向
                        #     location[neighbor] = (location[current_node][0],round(location[current_node][1] + weight_value,2))
                        # elif pos[neighbor][1] < pos[current_node][1]:#判断Y轴方向
                        #     location[neighbor] = (location[current_node][0], round(location[current_node][1] - weight_value,2))
                        #之前在dts中定义的排-列-层
                        if pos[neighbor][0] > pos[current_node][0]:#判断排，向下
                            location[neighbor] = (current_node_row , current_node_column + weight_value)
                        elif pos[neighbor][0] < pos[current_node][0]:#判断排，向上
                            location[neighbor] = (current_node_row , current_node_column - weight_value)
                        elif pos[neighbor][1] > pos[current_node][1]:#判断列，向右
                            location[neighbor] = (current_node_row  + weight_value, current_node_column)
                        elif pos[neighbor][1] < pos[current_node][1]:#判断列，向左
                            location[neighbor] = (current_node_row - weight_value, current_node_column)
                        else:
                            print(f"节点{neighbor}位置错误！pos：{pos[neighbor]}")
                            continue
            nx.set_node_attributes(floor, location, 'location')#将location加入图的属性中
            nx.set_node_attributes(floor, pos,'pos')
            nx.set_node_attributes(floor, node_colors,'node_colors')
            nx.set_node_attributes(floor, node_markers,'node_markers')
            end_time = time.time()
            print(f"读取第{layer}层地图数据耗时：{end_time-start_time}秒, 巷道数：{roadway_count}, 货位数：{freight_count}")
            return floor
        except FileNotFoundError:
                print(f"文件 {self.file_path} 未找到。")
        except Exception as e:
            print(f"读取第{layer}层地图数据时发生错误!: {e}")
            return None

    def read_firstFloor(self):
        file_path = '../Resource/map_data2.xlsx'
        try:
            start_time = time.time()
            usecols = [0, 1, 2, 3]  # 读取Excel文件中第一部分序号、ID、状态、类型四列数据
            data = pd.read_excel(file_path)#,usecols=usecols
            # print(data)
            # 显示列名
            # print("\n序号:")
            num = data.iloc[:, 0] #通过行和列的整数索引来访问数据
            num = num.dropna().astype(int)  # 删除包含空值的行,将序号转为整数

            # print("\nID:")
            id = data.iloc[:, 1].dropna()  # 删除包含空值的行

            # print("\n状态:")
            status = data.iloc[:, 2].dropna()  # 删除包含空值的行

            # print("\n类型:")
            dimension = data.iloc[:len(num), 3]

            #起点
            start_layer1=data.iloc[:, 4]
            start_layer1=start_layer1.dropna()  # 删除包含空值的行
            start_layer1=start_layer1.astype(int)  # 将起点转为整数
            #终点
            end_layer1=data.iloc[:, 5]
            end_layer1=end_layer1.dropna()  # 删除包含空值的行
            end_layer1=end_layer1.astype(int)  # 将终点转为整数
            #距离
            distance=data.iloc[:, 6]
            distance=distance.dropna()  # 删除包含空值的行
            distance=distance.astype(float)  # 将距离转为浮点数


        except FileNotFoundError:
            print(f"文件 {self.file_path} 未找到。")
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
        if len(num) == len(id) == len(status) == len(dimension):
            for n,i,s,d in zip(num,id,status,dimension):
                i=str(i)    #将ID转为字符串
                parts = i.split('-')
                if parts[0] != 'm':
                    row, col, layer = map(int, parts)#将parts中的元素分别转为整型并赋值给row,col,layer
                else:
                    row, col, layer = map(int, parts[1:])
                self.pos[n] = (row, col)  #使用排和列作为位置
                # 创建节点属性字典
                node_attr = {
                    'index': n,
                    'row': row,
                    'col': col,
                    'layer': layer,
                    'status': s,
                    'id': i,
                    'dimension': d
                }
                # 根据状态设置节点颜色
                if s == -1:
                    self.node_colors.append('yellow')  # 通道用黄色显示
                else:
                    self.node_colors.append('lightblue')  # 其他状态用蓝色显示
                self.nodes_data.append((n, node_attr))    # 节点属性字典加入列表
            #添加权重
            for start, end, weight in zip(start_layer1, end_layer1, distance):
                self.edges_data.append((start, end, {'weight': weight}))
            # 添加节点
            self.DG.add_nodes_from(self.nodes_data)
            # 添加边
            self.DG.add_edges_from(self.edges_data)
            self.DG.graph['pos']=self.pos
            self.DG.graph['node_colors']=self.node_colors
            print(f"读取文件 {self.file_path} 耗时 {time.time() - start_time:.2f} 秒。")
            return self.DG
            # return (self.DG,self.pos,self.node_colors)
        else:
            print("数据格式错误，请检查文件格式。")
            return None

    def update_colors(self, graph, path, node_color='red', edge_color='orange'):
        pos = nx.get_node_attributes(graph, 'pos')
        colors = nx.get_node_attributes(graph, 'node_colors')

        # 更新路径上节点的颜色
        for node in path:
            colors[node] = node_color

        # 更新路径上的边颜色
        for start, end in zip(path[:-1], path[1:]):
            edge = (start, end)
            # 绘制边
            plt.plot([pos[edge[0]][0], pos[edge[1]][0]], [pos[edge[0]][1], pos[edge[1]][1]], c=edge_color)

        # 更新节点颜色
        plt.scatter([pos[node][0] for node in graph.nodes()], [pos[node][1] for node in graph.nodes()], c=[colors[node] for node in graph.nodes()], marker='o')
        plt.pause(0.5)  # 暂停0.5秒，以便视觉上能看到更新。

def main():
    # plt.style.use('_mpl-gallery')
    dm = Model()
    combined_graph, floor1, floor2, floor3=dm.combined_graph, dm.floor1, dm.floor2, dm.floor3
    # 绘制所有楼层地图
    dm.draw_floors([floor1, floor2, floor3], ["Floor 1", "Floor 2", "Floor 3"])

    # 假设给定的路径
    # path = [i for i in range(1, 25)]  # 示例路径节点ID，实际根据需求调整
    # # 动态更新节点与边的颜色
    # for node in path:
    #     dm.update_colors(floor1, [node])  # 每次只更新当前节点的颜色
    #绘制3维地图
    # dm.draw_3D_map(combined_graph)
    # # 绘制每一层地图
    # dm.draw_floor(floor1, "Floor 1")
    # dm.draw_floor(floor2, "Floor 2")
    # dm.draw_floor(floor3, "Floor 3")

if __name__ == '__main__':
    main()


Status = {0: "空闲", 1: "忙碌", 2: "故障"}
class Vehicle:
    def __init__(self,View,Env, ID, initial_position=0):
        self.max_speed = 5  # AGV最大速度
        self.acceleration = 2  # AGV加速度
        self.deceleration = 2  # AGV减速度
        self.current_speed = 0  # AGV当前速度
        self.status = 0  # AGV当前状态 0-空闲 1-忙碌 2-故障

        self.view = View    # 图
        self.env = Env    # 环境
        self.ID = ID    # AGVID
        self.current_position = initial_position     # AGV初始位置
        self.create_show( self.current_position)  # 创建AGV图片
        self.image = mpimg.imread("../Resource/AGV.png")  # 图片

    def create_show(self, node):
        """
        更新AGV图片的位置，显示在指定的节点上。
        :param node: 指定的节点
        """
        # 计算节点的位置
        # self.DG.nodes[node]['pos'] = (node[0], node[1])
        # x, y = self.DG.nodes[node]['pos']
        self.DG = self.view.DG
        # print(self.DG)
        x = self.DG.nodes[node]['row']
        y = self.DG.nodes[node]['col']
        print(f"AGV的位置为: ({x}, {y})")
        # 更新AGV图片的位置

        # nx.draw_networkx_nodes(self.DG, (x, y), [node], node_size=80, alpha=0.8, node_color='red')
        # plt.draw()
        # plt.pause(0.001)
        # self.agv_image = self.ax.imshow(self.Image, aspect='auto', extent=(x-0.1, x+0.1, y-0.1, y+0.1), zorder=2)

    def move(self):
        for node in self.path:
            yield self.env.timeout(1)  # 模拟每一步的时间
            self.current_position = node
            print(f"AGV 当前节点: {node}, 时间: {self.env.now}")

    def moveing_AGV(self,path):
        # 移动AGV
        current_speed = 0
        current_node_index = 0
        speed = 1  # 默认速度
        while current_node_index < len(path) - 1:
            start_node = path[current_node_index]
            end_node = path[current_node_index + 1]
            # 计算移动距离（假设图中有边的权重表示距离）
            distance = self.DG.edges[start_node, end_node]['weight']  # 获取两节点之间的距离
            # 计算到达下一个节点所需的时间
            time_to_move = distance / speed
            # 移动过程
            # for _ in range(int(time_to_move)):
            time.sleep(time_to_move)  # 每秒钟更新一次
            # 更新当前位置
            self.current_position = end_node
            x, y = self.DG.nodes[self.current_position]  # 获取目标节点的位置
            self.agv_image = self.ax.imshow(self.Image, aspect='auto', extent=(x-0.1, x+0.1, y-0.1, y+0.1), zorder=2)

            current_node_index += 1
            # print(f"到达节点: {end_node}")
        print("AGV已到达终点。")


class Elevator:
    def __init__(self, ID, elavator_position ,initial_floor, Env):
        self.ID = ID  # 电梯ID
        self.elavator_position = elavator_position  # 电梯位置
        self.current_floor = initial_floor  # 电梯当前楼层
        self.env = Env  # 环境
        self.current_status = 0  # 电梯当前状态 0-空闲 1-忙碌 2-故障
        self.max_speed = 1      # 电梯最大速度
        self.acceleration = 1   # 电梯加速度
        self.current_speed = 0  # 电梯当前速度
        self.wait_time = 1      # 电梯等待时间

        self.load_status = 0  # 电梯载重状态 0-空 1-满
        self.load_direction = 0  # 电梯载重方向 0-上 1-下
        self.Task_list = []  # 电梯载重列表






