import warnings
from collections import defaultdict
from Algorithm.PathPlanning import Path_Planning
import numpy as np

from NSGA2.FS4.FS4_Individual import Individual
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
        self.order_counter = 0  # 添加全局计数器，用于记录分配顺序
        self.path_planning = Path_Planning(model)  # 路径规划器对象
        # 使用列表推导式和内置函数来获取最大列、最大排和最高层
        rows, cols, layers = zip(*self.pos.values())
        self.max_col = max(cols)        # 最大列
        self.max_row = max(rows)        # 最大排
        self.max_layer = max(layers)    # 最高层


    def generate_individual(self):
        """随机生成一个个体"""
        individual = Individual()
        # 获取并且打乱 aisles_dict 的键
        # shuffled_aisle_ids = list(self.aisles_dict.keys())
        # random.shuffle(shuffled_aisle_ids)
        # individual.features = []  # 初始化为空的染色体
        # individual.allocated_nodes = None  # 初始化为空的分配结果
        shuffled_aisle_ids = [random.randint(0, self.variables_range-1)
                               for _ in range(self.num_of_variables)]
        random.shuffle(shuffled_aisle_ids)
        individual.features = shuffled_aisle_ids
        return individual

    def decode_individual(self, individual, pending_Loc):
        """
        解码个体，生成新的 allocated_nodes 结构：[(node_id, sku, order), ...]。
        :return: 分配的节点列表
        将货道索引解码为货位分配，不修改 TopoGraph"""
        individual.allocated_nodes = []
        self.node_sku_map.clear()  # 重置映射表
        used_nodes = set()  # 临时记录已分配的节点，避免重复分配
        # 初始化临时容量字典，复制原始容量
        remaining_capacity = {aisle_id: aisle['capacity'] for aisle_id, aisle in self.aisles_dict.items()}
        # print(individual.features)
        # 遍历待入库货物
        for sku, sku_data in pending_Loc.items():
            num_needed = sku_data['num']
            # 遍历货道索引(个体方案)
            for aisle_idx in individual.features:
                if num_needed <= 0:
                    break
                aisle_id = list(self.aisles_dict.keys())[aisle_idx]  # 货道ID
                selected_nodes = self.select_location(sku, aisle_id, num_needed, used_nodes,remaining_capacity)
                if selected_nodes:
                    individual.allocated_nodes.extend(selected_nodes)
                    num_needed -= len(selected_nodes)
            # print(f"sku:{sku} 节点分配：{individual.allocated_nodes}, 剩余容量：{num_needed}")

        individual.is_decoded = True
        return individual.allocated_nodes

    """根据货道ID和货物类型,选择分配的货位节点列表（深度优先存储）"""
    def select_location(self, sku, aisle_id, num_needed, used_nodes,remaining_capacity):
        """
        从货道中选择货位，返回 [(node_id, sku, order), ...] 格式的节点列表。
        :param sku: 当前 SKU 的 ID
        :param aisle_id: 货道 ID
        :param num_needed: 需要分配的货位数量
        :param used_nodes: 已使用的节点集合
        :param remaining_capacity: 货道的剩余容量字典
        :return: 选中的节点列表
        """
        aisle = self.aisles_dict.get(aisle_id)
        if not aisle or remaining_capacity[aisle_id] <= 0:  # 使用临时容量检查
            return []
        type = aisle['type']  # -1: 左侧最深，0: 双向队列，1: 右侧最深
        # 获取货道所有节点
        aisle_nodes = aisle['nodes']
        # 获取 SKU 的 dimension
        sku_dimension = self.pending_Loc[sku]['dimension']
        # 获取所有节点的 dimension（假设存储在 TopoGraph 的节点属性中）
        node_dimensions = nx.get_node_attributes(self.TopoGraph, 'dimension')
        # 使用 used_nodes 检查可用节点，而不是依赖 TopoGraph 的 status
        # available_nodes = [node for node in aisle_nodes if node not in used_nodes]
        # 筛选可用节点：未使用且 dimension 匹配
        available_nodes = [
            node for node in aisle_nodes
            if node not in used_nodes and node_dimensions.get(node) == sku_dimension
        ]
        num_available = min(len(available_nodes), num_needed, remaining_capacity[aisle_id])
        if num_available == 0:
            return []
        # 选择节点
        if type == -1:  # 向左存储，从左侧（小 y）开始
            selected_nodes = sorted(available_nodes, key=lambda node: self.pos[node][1])[:num_available]
        else :  # 向右存储，从右侧（大 y）开始
            selected_nodes = sorted(available_nodes, key=lambda node: self.pos[node][1], reverse=True)[:num_available]
        # else:  # type == 0，双向队列，从左侧开始（可调整）
        #     selected_nodes = sorted(available_nodes, key=lambda node: self.pos[node][1])[:num_available]
        self.order_counter = 0  # 重置分配顺序计数器,标志入货道的前后顺序
        # 生成新的节点列表格式：[(node_id, sku, order), ...]
        selected = []
        for node in selected_nodes:
            selected.append((node, sku, self.order_counter))
            self.node_sku_map[node] = sku
            used_nodes.add(node)
            self.order_counter += 1  # 递增分配顺序计数器
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
        """
        适配新的 allocated_nodes 结构：[(node_id, sku, order), ...]。
        """
        if not allocated_nodes:
            warnings.warn("Gravity-allocated_nodes is empty, return default centroid 0.0")
            return 0.0
        total_moment = 0
        total_mass = 0
        pos = nx.get_node_attributes(self.TopoGraph, 'pos')
        for node, sku, _ in allocated_nodes:
            quality = self.pending_Loc[sku]['quality']
            # layer_center = self.node_centers[node]
            # total_moment += quality * layer_center
            x, y, z = pos[node]
            total_moment += quality * (z-0.5)
            total_mass += quality
        if total_mass == 0:
            warnings.warn("Total mass is zero, returning default centroid 0.0")
            return 0.0
        return round(total_moment / total_mass, 3)

    def f2_balanced(self, allocated_nodes):
        """
        适配新的 allocated_nodes 结构：[(node_id, sku, order), ...]。
        """
        if not allocated_nodes:
            warnings.warn("f2_balanced-allocated_nodes is empty, return default  0.0")
            return 0.0
        def safe_standard_deviation(values,max_count):#
            n = len(values)
            if n < 2:
                warnings.warn(f"安全计算标准差（处理样本数不足的情况）!,return 0.0 {values}", UserWarning)
                return 0.0
            mean = sum(values) / max_count   # 计算均值
            variance = sum((x - mean)**2 for x in values) / (max_count - 1)
            return np.sqrt(variance)
        current_x = defaultdict(float)
        current_y = defaultdict(float)
        current_z = defaultdict(float)
        for node, sku, _ in allocated_nodes:
            rate = self.pending_Loc[sku]['rate']
            row, col, layer = self.pos[node]
            current_x[row] += rate    # 按行累计周转率
            current_y[col] += rate    # 按列累计周转率
            current_z[layer] += rate     # 按层累计周转率

        x_rates = [current_x[i] for i in range(self.max_row)]  # 所有排的周转率
        y_rates = [current_y[i] for i in range(self.max_col)]  # 所有列的周转率
        z_rates = [current_z[i] for i in range(self.max_layer)]  # 所有层的周转率
        # print(" X轴样本数：",len(x_rates)," 行理论样本数：",self.max_row)
        # print(" Y轴样本数：",len(y_rates)," 列理论样本数：",self.max_col)
        # print(" Z轴样本数：",len(z_rates)," 层理论样本数：",self.max_layer)
        var_x = safe_standard_deviation(x_rates,self.max_row)
        var_y = safe_standard_deviation(y_rates,self.max_col)
        var_z = safe_standard_deviation(z_rates,self.max_layer)
        return var_x + var_y + var_z

    def f3_efficiency(self, allocated_nodes):
        """
        适配新的 allocated_nodes 结构：[(node_id, sku, order), ...]。
        """
        if not allocated_nodes:
            warnings.warn("f3_efficiency-allocated_nodes is empty, return default 0.0")
            return 0.0
        efficient = 0.0
        time_cost = 0
        for node, sku, _ in allocated_nodes:
            sku_id = self.node_sku_map[node]
            sku_data = self.pending_Loc[sku_id]
            enter_node = sku_data["enter_node"]
            rate = sku_data["rate"]
            if enter_node == node:
                warnings.warn("计算效率时，出入库点相同，跳过！")
                return 0.0
            try:
                path_length = nx.shortest_path_length(self.TopoGraph, source=enter_node, target=node)
                # path,length,explored = self.path_planning.improve_A_star(self.TopoGraph, source=enter_node, target=node)
                # time_cost = self.path_planning.cal_path_time(self.TopoGraph, path)
            except nx.NetworkXException:
                print("路径规划失败，效率值为1e6")
                time_cost = 1e6
            # efficient += time_cost * rate
            efficient += path_length * rate
        return efficient

    def evaluate_individual(self, individual):
        """
        评估单个个体的目标值。

        :param individual: 个体对象
        :return: 目标值列表 [f1, f2, f3]
        """
        if not individual.is_decoded:
            self.decode_individual(individual, self.pending_Loc)
        objectives = [f(individual.allocated_nodes) for f in self.objectives]
        return objectives

    # def calculate_objectives(self, population):
    #     """
    #     并行计算种群中所有个体的目标值。
    #
    #     :param population: 种群对象
    #     :return: None（直接修改个体的 objectives 属性）
    #     """
    #     # 准备并行任务
    #     tasks = [ind for ind in population if not hasattr(ind, 'objectives') or not ind.objectives]
    #
    #     if tasks:
    #         # 并行评估
    #         results = self.pool.map(self.evaluate_individual, tasks)
    #         # 将结果赋值给个体
    #         for ind, objectives in zip(tasks, results):
    #             ind.objectives = objectives

    def calculate_objectives(self, individual):
        """计算个体的目标函数值"""
        if not individual.is_decoded:
            # 解码个体，更新 allocated_nodes
            self.decode_individual(individual, self.pending_Loc)
        # 计算目标函数值
        individual.objectives = [f(individual.allocated_nodes) for f in self.objectives]
        return individual.objectives

    # def __del__(self):
    #     """
    #     清理进程池。
    #     """
    #     self.pool.close()
    #     self.pool.join()