import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
from collections import defaultdict

class AllocationVisualizer:
    def __init__(self, floor, title_prefix="Cargo Allocation Visualization"):
        """
        初始化可视化器。

        :param floor: 拓扑图（networkx 图）
        :param title_prefix: 图表标题前缀
        """
        self.floor = floor
        self.title_prefix = title_prefix
        self.pos = nx.get_node_attributes(self.floor, 'pos')  # 获取节点位置 (x, y, z)

    def adjust_color_brightness(self, color, factor):
        """
        调整颜色的亮度。

        :param color: 原始颜色（RGB 格式）
        :param factor: 亮度因子（0 到 1，1 为最亮，0 为最暗）
        :return: 调整后的颜色
        """
        rgb = mcolors.to_rgb(color)
        hsv = mcolors.rgb_to_hsv(rgb)
        hsv[2] = hsv[2] * factor
        return mcolors.hsv_to_rgb(hsv)

    def get_nodes_by_layer(self):
        """
        按楼层分组节点。

        :return: 字典 {layer: [node_ids], ...}
        """
        nodes_by_layer = defaultdict(list)
        for node in self.floor.nodes():
            # 假设 pos 是一个 (x, y, z) 元组，z 表示楼层（1, 2, 3）
            layer = int(self.pos[node][2])  # z 坐标表示楼层
            nodes_by_layer[layer].append(node)
        return nodes_by_layer

    def visualize_allocation(self, allocated_nodes, aisles_dict):
        """
        按楼层可视化货位分配方案，绘制三张地图。

        :param allocated_nodes: 分配的节点列表 [(node_id, sku, order), ...]
        :param aisles_dict: 货道字典，用于确定节点所属的巷道
        """
        # 按楼层分组节点
        nodes_by_layer = self.get_nodes_by_layer()

        # 为每个 SKU 分配基础颜色（全局一致）
        sku_list = list(set(node_info[1] for node_info in allocated_nodes))
        base_colors = plt.cm.get_cmap('tab10', len(sku_list))
        sku_color_map = {sku: base_colors(i) for i, sku in enumerate(sku_list)}

        # 按巷道分组节点
        node_to_aisle = {}
        for aisle_id, aisle_info in aisles_dict.items():
            for node in aisle_info['nodes']:
                node_to_aisle[node] = aisle_id

        # 按楼层绘制地图
        for layer in sorted(nodes_by_layer.keys()):  # 按楼层顺序绘制（1, 2, 3）
            print(f"Drawing map for Layer {layer}...")

            # 创建该楼层的子图
            layer_nodes = nodes_by_layer[layer]
            layer_subgraph = self.floor.subgraph(layer_nodes)

            # 过滤该楼层的 allocated_nodes
            layer_allocated_nodes = [
                (node, sku, order) for node, sku, order in allocated_nodes
                if node in layer_nodes
            ]

            # 按巷道统计节点分配顺序
            aisle_orders = defaultdict(list)
            for node_info in layer_allocated_nodes:
                node, sku, order = node_info
                aisle_id = node_to_aisle.get(node)
                if aisle_id is not None:
                    aisle_orders[aisle_id].append(node_info)

            # 创建新图表
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_title(f"{self.title_prefix} - Layer {layer}")

            # 获取节点属性
            location = nx.get_node_attributes(layer_subgraph, 'location')
            node_markers = nx.get_node_attributes(layer_subgraph, 'node_markers')
            # 获取节点的独有颜色（假设属性名为 'node_color'）
            node_colors_attr = nx.get_node_attributes(layer_subgraph, 'node_colors')  # 可能需要根据实际属性名调整

            # 提取 X, Y 坐标
            x, y = zip(*location.values()) if location else ([], [])

            # 绘制边
            for edge in layer_subgraph.edges():
                x_edges = [location[edge[0]][0], location[edge[1]][0]]
                y_edges = [location[edge[0]][1], location[edge[1]][1]]
                ax.plot(x_edges, y_edges, c='gray', zorder=1)

            # 为每个节点设置颜色
            node_colors = {}
            for node in layer_subgraph.nodes():
                # 优先使用节点的独有颜色（如果存在）
                if node in node_colors_attr:
                    node_colors[node] = node_colors_attr[node]
                else:
                    # 如果没有独有颜色，设置为默认颜色（未分配的货位节点）
                    node_colors[node] = 'lightblue'

            for aisle_id, nodes_in_aisle in aisle_orders.items():
                # 按分配顺序排序
                nodes_in_aisle.sort(key=lambda x: x[2])
                num_nodes = len(nodes_in_aisle)
                for i, (node, sku, order) in enumerate(nodes_in_aisle):
                    # 根据分配顺序调整颜色亮度（从深到浅）
                    # brightness = 1.0 - (i / max(num_nodes - 1, 1)) * 0.5
                    # 根据分配顺序调整颜色亮度（从深到浅）
                    brightness = (i / max(num_nodes - 1, 1)) * 0.5 + 0.5  # 从 0.5（深）到 1.0（浅）
                    base_color = sku_color_map[sku]
                    node_colors[node] = self.adjust_color_brightness(base_color, brightness)

            # 绘制节点
            node_colors_list = [node_colors[node] for node in layer_subgraph.nodes()]
            node_markers_list = [node_markers[node] for node in layer_subgraph.nodes()]

            for marker in set(node_markers_list):
                indices = [i for i, m in enumerate(node_markers_list) if m == marker]
                ax.scatter(
                    [x[i] for i in indices],
                    [y[i] for i in indices],
                    c=[node_colors_list[i] for i in indices],
                    marker=marker,
                    edgecolors='black',
                    linewidths=0.7,
                    zorder=1
                )

            # 创建图例
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='black', linewidth=1, markerfacecolor='yellow', markersize=8, label='巷道节点'),
                plt.Line2D([0], [0], marker='o', color='black', linewidth=1, markerfacecolor='lightblue', markersize=8, label='货位节点'),
                plt.Line2D([0], [0], marker='s', color='black', linewidth=1, markerfacecolor='lightgreen', markersize=8, label='入库节点'),
                plt.Line2D([0], [0], marker='s', color='black', linewidth=1, markerfacecolor='darkorange', markersize=8, label='出库节点'),
                plt.Line2D([0], [0], marker='D', color='black', linewidth=1, markerfacecolor='darkorange', markersize=8, label='换层节点'),
            ]
            # 添加 SKU 颜色图例
            for sku, color in sku_color_map.items():
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='black', linewidth=1, markerfacecolor=color, markersize=8, label=f'SKU {sku}')
                )

            ax.legend(handles=legend_elements, loc='center left', ncol=1, bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
            ax.set_xlabel('列 坐标')
            ax.set_ylabel('排 坐标')
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            plt.tight_layout()
            plt.show()