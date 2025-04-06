import warnings
from collections import defaultdict

from NSGA2.FS.FS_Individual import Individual
import random
import networkx as nx
enter_node = [51, 348, 636, 925, 1110, 1620]# 入口点
"""预计算节点中心高度"""
heights = [3.15, 2.55, 1.5]  # 各层高度
cumulative_heights = [0, 3.15, 5.7]  # 各层累计高度
class Problem:
    def __init__(self, num_of_variables, variables_range, model, aisles_dict, pending_Loc):
        """
        :param objectives: 目标函数列表 [F1_Weight, F2_Balanced, F3_Efficiency]
        :param num_of_variables: 变量数量（货道数量）
        :param variables_range: 变量取值范围（货道索引范围）
        :param model: 数据模型对象，包含 TopoGraph 和其他信息
        :param aisles_dict: 货道字典 {aisle_id: {type, nodes, ...}}
        """
        objectives = [self.f1_weight, self.f2_balanced, self.f3_efficiency]
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.objectives = objectives
        self.variables_range = variables_range
        self.model = model
        self.TopoGraph = model.combined_graph  # 只读图结构
        self.aisles_dict = aisles_dict
        self.node_sku_map = {}  # {node_id: sku_id}，每次解码时重置
        self.pos = nx.get_node_attributes(self.TopoGraph, 'pos')
        self.locations = nx.get_node_attributes(self.TopoGraph, 'location')
        self.pending_Loc = pending_Loc  # 存储待入库信息
        self.node_centers = self._precompute_node_centers()  # 预计算所有节点的层中心高度

    def generate_individual(self):
        """随机生成一个个体"""
        individual = Individual()
        individual.features = [random.randint(0, self.variables_range-1)
                               for _ in range(self.num_of_variables)]
        return individual

    def decode_individual(self, individual, pending_Loc):
        """将货道索引解码为货位分配，不修改 TopoGraph"""
        individual.allocated_nodes = []
        self.node_sku_map.clear()  # 重置映射表
        used_nodes = set()  # 临时记录已分配的节点，避免重复分配
        # 初始化临时容量字典，复制原始容量
        remaining_capacity = {aisle_id: aisle['capacity'] for aisle_id, aisle in self.aisles_dict.items()}
        # 遍历待入库货物
        for sku, sku_data in pending_Loc.items():
            num_needed = sku_data['num']
            # 遍历货道索引(个体方案)
            for aisle_idx in individual.features:
                if num_needed <= 0:
                    break
                aisle_id = list(self.aisles_dict.keys())[aisle_idx]
                selected_nodes = self.select_location(sku, aisle_id, num_needed, used_nodes,remaining_capacity)
                if selected_nodes:
                    individual.allocated_nodes.extend(selected_nodes)
                    num_needed -= len(selected_nodes)
            # print(f"sku:{sku} 节点分配：{individual.allocated_nodes}, 剩余容量：{num_needed}")

        individual.is_decoded = True
        return individual.allocated_nodes

    """根据货道ID和货物类型,选择分配的货位节点列表（深度优先存储）"""
    def select_location(self, sku, aisle_id, num_needed, used_nodes,remaining_capacity):
        """从货道中选择货位，不修改 TopoGraph"""
        aisle = self.aisles_dict.get(aisle_id)
        if not aisle or remaining_capacity[aisle_id] <= 0:  # 使用临时容量检查
            return []
        type = aisle['type']  # -1: 左侧最深，0: 双向队列，1: 右侧最深
        # 获取货道所有节点
        aisle_nodes = aisle['nodes']
        # 使用 used_nodes 检查可用节点，而不是依赖 TopoGraph 的 status
        available_nodes = [node for node in aisle_nodes if node not in used_nodes]
        num_available = min(len(available_nodes), num_needed, remaining_capacity[aisle_id])
        if num_available == 0:
            return []
        # 选择节点
        if type == -1:  # 向左存储，从左侧（小 y）开始
            selected = sorted(available_nodes, key=lambda node: self.pos[node][1])[:num_available]
        else :  # 向右存储，从右侧（大 y）开始
            selected = sorted(available_nodes, key=lambda node: self.pos[node][1], reverse=True)[:num_available]
        # else:  # type == 0，双向队列，从左侧开始（可调整）
        #     selected = sorted(available_nodes, key=lambda node: self.pos[node][1])[:num_available]

        for node in selected:
            self.node_sku_map[node] = sku
            used_nodes.add(node)  # 记录为已使用
        remaining_capacity[aisle_id] -= len(selected)  # 更新临时容量
        return selected

    """预计算所有节点的层中心高度"""
    def _precompute_node_centers(self):
        node_centers = {}
        for node in self.pos:
            (x, y, z) = self.pos[node]
            cumulative_height = cumulative_heights[z - 1]  # 前z-1层总高度
            current_layer_center = cumulative_height + heights[z - 1] / 2  # 前z-1层总高度+当前层高度/2
            node_centers[node] = current_layer_center  # 节点的层中心。在后续处理中，只需要通过节点编号直接获取层中心值，而不需要每次重新计算。
        return node_centers
    '''-----------------------------以下为计算目标函数值----------'''
    def f1_weight(self, allocated_nodes):
        if not allocated_nodes:
            warnings.warn("Gravity-allocated_nodes is empty, return default centroid 0.0")
            return 0.0
        total_moment = 0
        total_mass = 0
        for node in allocated_nodes:
            sku = self.node_sku_map[node]
            quality = self.pending_Loc[sku]['quality']
            layer_center = self.node_centers[node]
            total_moment += quality * layer_center
            total_mass += quality
        if total_mass == 0:
            warnings.warn("Total mass is zero, returning default centroid 0.0")
            return 0.0
        return round(total_moment / total_mass, 3)

    def f2_balanced(self, allocated_nodes):
        if not allocated_nodes:
            return 0.0
        def safe_variance(values):
            n = len(values)
            if n < 2:
                warnings.warn(f"安全计算方差（处理样本数不足的情况）!,return 0.0 {values}", UserWarning)
                return 0.0
            mean = sum(values) / n
            return sum((x - mean)**2 for x in values) / (n - 1)
        current_x = defaultdict(float)
        current_y = defaultdict(float)
        current_z = defaultdict(float)
        for node in allocated_nodes:
            sku_id = self.node_sku_map[node]
            rate = self.pending_Loc[sku_id]['rate']
            _, _, z = self.pos[node]
            # current_x[y] += rate
            # current_y[x] += rate
            # current_z[z] += rate
            x,y = self.locations[node]
            current_x[x] += rate
            current_y[y] += rate
            current_z[z] += rate
        var_x = safe_variance(list(current_x.values()))
        var_y = safe_variance(list(current_y.values()))
        var_z = safe_variance(list(current_z.values()))
        return var_x + var_y + var_z

    def f3_efficiency(self, allocated_nodes):
        efficient = 0.0
        for node in allocated_nodes:
            sku_id = self.node_sku_map[node]
            sku_data = self.pending_Loc[sku_id]
            enter_node = sku_data["enter_node"]
            rate = sku_data["rate"]
            if enter_node == node:
                warnings.warn("计算效率时，出入库点相同，跳过！")
                return 0.0
            try:
                path_length = nx.shortest_path_length(self.TopoGraph, source=enter_node, target=node)
            except nx.NetworkXException:
                path_length = 1e6
            efficient += path_length * rate
        return efficient

    def calculate_objectives(self, individual):
        """计算个体的目标函数值"""
        if not individual.is_decoded:
            # 解码个体，更新 allocated_nodes
            self.decode_individual(individual, self.pending_Loc)
        # 计算目标函数值
        individual.objectives = [f(individual.allocated_nodes) for f in self.objectives]
        return individual.objectives

