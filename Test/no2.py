import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
# 假设你的Python脚本和Resource文件夹在同一个目录下
# file_path = '../Resource/data.xlsx'
file_path = '../Resource/map_data2.xlsx'
# 记录开始时间
start_time = time.time()
# 读取Excel文件
try:
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
    distance=distance.astype(float)  # 将距离转为整数
except FileNotFoundError:
    print(f"文件 {file_path} 未找到。")
except Exception as e:
    print(f"读取文件时发生错误: {e}")
print(f"第一层：start_layer1 {len(start_layer1)} end_layer1 {len(end_layer1)} distance {len(distance)}")
print(f"第一层：num {len(num)} id {len(id)} status {len(status)} dimension {len(dimension)}")
# 创建一个空的有向图
DG = nx.DiGraph()
# 解析ID并添加节点
nodes_data = []   # 用于存储节点数据
edges_data = []   # 用于存储边数据
pos = {}          # 用于存储节点位置
node_colors = []  # 用于存储节点颜色
if len(num) == len(id) == len(status) == len(dimension):
    #zip(num, id ,status) 将 num 和 id ,status两个列表合并成一个迭代器,同时遍历 num 和 id ,status
    for n,i,s,d in zip(num,id,status,dimension):
        i=str(i)    #将ID转为字符串
        parts = i.split('-')
        if parts[0] != 'm':
            row, col, layer = map(int, parts)#将parts中的元素分别转为整型并赋值给row,col,layer
        else:
            row, col, layer = map(int, parts[1:])
        pos[n] = (row, col)  # 使用排和列作为位置
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
            node_colors.append('yellow')  # 通道用黄色显示
        else:
            node_colors.append('lightblue')  # 其他状态用蓝色显示
        nodes_data.append((n, node_attr))    # 节点属性字典加入列表

# 确保 start_layer1, end_layer1, distance 的长度相同
if len(start_layer1) == len(end_layer1) == len(distance):
    # 使用 zip 函数将三个列表合并成一个迭代器
    for start, end, weight in zip(start_layer1, end_layer1, distance):
        edges_data.append((start, end, {'weight': weight}))

else:
    print("起点、终点和权重列表的长度不一致，无法添加边。")
# 添加一些示例边
# edges = [(1, 2), (2, 3),  (3, 4)]
# # 将节点数据和边数据合并
# edges_data = [(u, v, {'weight': 1}) for u, v in edges]
# 添加节点
DG.add_nodes_from(nodes_data)
# 添加边
DG.add_edges_from(edges_data)

# 记录结束时间
end_time = time.time()
# 计算并输出执行时间
execution_time = end_time - start_time
print(f"读取数据构建地图执行时间: {execution_time} 秒")
print(DG)#第一层： 1550 nodes and 3153 edges
start_time = time.time()
nx.draw(DG,pos,with_labels=True, node_size=80, alpha=0.8, node_color=node_colors,
        edge_color='black', font_size=6)#font_weight='bold',序号加粗  width=1,
plt.title('Topology Map')
# 记录结束时间
end_time = time.time()
# 计算并输出执行时间
execution_time = end_time - start_time
print(f"绘制地图执行时间: {execution_time} 秒")
plt.show()

