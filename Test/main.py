import networkx as nx
import matplotlib.pyplot as plt
import random

def create_warehouse_layout(num_locations):
    # 创建一个无向图
    G = nx.Graph()

    # 添加节点，每个节点代表一个货位
    for i in range(num_locations):
        G.add_node(f'A{i+1}')

    # 随机添加边，连接货位
    for i in range(num_locations):
        for j in range(i+1, num_locations):
            if random.random() > 0.5:  # 50%的概率添加一条边
                G.add_edge(f'A{i+1}', f'A{j+1}')
    return G

def draw_warehouse_layout(G):
    # 绘制图形
    pos = nx.spring_layout(G)  # 为图形创建布局
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title('Warehouse Layout')
    plt.show()

# 创建一个包含100个货位的拓扑地图
warehouse_layout = create_warehouse_layout(100)

# 绘制拓扑地图
draw_warehouse_layout(warehouse_layout)