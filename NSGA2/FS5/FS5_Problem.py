import pickle
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed  # 确保导入正确的as_completed

from tqdm import tqdm  # 可选：进度条
from scipy.sparse import dok_matrix

from Algorithm.PathPlanning import Path_Planning
import numpy as np
import geatpy as ea
from NSGA2.FS5.FS5_Individual import Individual
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

        self.num_of_variables = num_of_variables
        self.objectives = [self.f1_weight, self.f2_balanced, self.f3_efficiency]    # 目标函数列表
        self.num_of_objectives = len(self.objectives)     # 目标函数数量
        self.variables_range = variables_range    # 变量取值范围
        self.model = model
        self.TopoGraph = model.combined_graph  # 只读图结构
        self.aisles_dict = aisles_dict
        self.node_sku_map = {}  # {node_id: sku_id}，每次解码时重置
        self.pending_Loc = pending_Loc  # 存储待入库信息
        # self.node_centers = self._precompute_node_centers()  # 预计算所有节点的层中心高度
        self.order_counter = 0  # 添加全局计数器，用于记录分配顺序
        self.path_planning = Path_Planning(model)  # 路径规划器对象
        # 使用列表推导式和内置函数来获取最大列、最大排和最高层

        Lind = self.num_of_variables  #染色体长度
        self.FieldDR = np.array([[0]*Lind, [489]*Lind, [1]*Lind])  # 构造 FieldDR

        self.pos = nx.get_node_attributes(self.TopoGraph, 'pos')
        self.locations = nx.get_node_attributes(self.TopoGraph, 'location')
        rows, cols, layers = zip(*self.pos.values())
        self.max_col = max(cols)        # 最大列
        self.max_row = max(rows)        # 最大排
        self.max_layer = max(layers)    # 最高层
        #节点规格
        self.node_dimensions = nx.get_node_attributes(self.TopoGraph, 'dimension')
        #sku规格
        # self.sku_dimensions = [pending_Loc[sku]['dimension'] for sku in pending_Loc]
        # self.sku_qualities = [pending_Loc[sku]['quality'] for sku in pending_Loc]
        # self.sku_rates = [pending_Loc[sku]['rate'] for sku in pending_Loc]
        self.path_cache = {}  # 路径缓存 {(enter, target): (length, time)}
        self._precompute_paths()  # 初始化时预计算路径长度和时间


    def generate_individual(self):
        """随机生成一个个体"""
        individual = Individual()
        # 获取并且打乱 aisles_dict 的键
        Chrom = ea.crtpp(1,FieldDR=self.FieldDR)  # 创建一个排列编码种群染色体矩阵
        print(" Chrom :",Chrom)
        individual.features = Chrom[0].tolist()
        # shuffled_aisle_ids = [random.randint(0, self.variables_range-1)
        #                        for _ in range(self.num_of_variables)]
        # random.shuffle(shuffled_aisle_ids)
        # individual.features = shuffled_aisle_ids
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
        # 一次性将货道 ID 列表缓存
        aisle_ids = list(self.aisles_dict.keys())
        # 遍历待入库货物
        for sku, sku_data in pending_Loc.items():
            num_needed = sku_data['num']
            # 遍历货道索引(个体方案)
            for aisle_idx in individual.features:
                if num_needed <= 0:
                    break
                aisle_id = aisle_ids[aisle_idx]  # 货道ID
                selected_nodes = self.select_location(sku, aisle_id, num_needed, used_nodes,remaining_capacity)
                if selected_nodes:
                    individual.allocated_nodes.extend(selected_nodes)
                    num_needed -= len(selected_nodes)
            if num_needed > 0:  # 货道容量不足，无法分配
                warnings.warn(f"Not enough nodes for sku {sku}, {num_needed} remaining")
                individual.unallocated[sku] = num_needed  # 记录未分配的 SKU 和数量

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
        # node_dimensions = nx.get_node_attributes(self.TopoGraph, 'dimension')
        # 使用 used_nodes 检查可用节点，而不是依赖 TopoGraph 的 status
        # available_nodes = [node for node in aisle_nodes if node not in used_nodes]
        # 筛选可用节点：未使用且 dimension 匹配
        available_nodes = [
            node for node in aisle_nodes
            if node not in used_nodes and self.node_dimensions.get(node) == sku_dimension
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
        #         #     selected_nodes = sorted(available_nodes, key=lambda node: self.pos[node][1])[:num_available]
        order_counter = 0  # 重置分配顺序计数器,标志入货道的前后顺序
        # 生成新的节点列表格式：[(node_id, sku, order), ...]
        selected = []
        for node in selected_nodes:
            selected.append((node, sku, order_counter))
            self.node_sku_map[node] = sku
            used_nodes.add(node)
            order_counter += 1  # 递增分配顺序计数器
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
        # 转换为NumPy数组,向量化加速计算
        nodes = np.array([node for node, _, _ in allocated_nodes])
        qualities = np.array([self.pending_Loc[sku]['quality'] for _, sku, _ in allocated_nodes])
        z_coords = np.array([self.pos[node][2] for node in nodes])  # 预存z坐标为数组

        total_moment = np.dot(qualities, z_coords)
        total_mass = qualities.sum()
        return round(total_moment / total_mass, 3) if total_mass != 0 else 0.0
        # total_moment = 0
        # total_mass = 0
        # pos = nx.get_node_attributes(self.TopoGraph, 'pos')
        # for node, sku, _ in allocated_nodes:
        #     quality = self.pending_Loc[sku]['quality']
        #     x, y, z = pos[node]
        #     total_moment += quality * z
        #     total_mass += quality
        # if total_mass == 0:
        #     warnings.warn("Total mass is zero, returning default centroid 0.0")
        #     return 0.0
        # return round(total_moment / total_mass, 3)

    def f2_balanced(self, allocated_nodes):
        """
        适配新的 allocated_nodes 结构：[(node_id, sku, order), ...]。
        """
        if not allocated_nodes:
            warnings.warn("f2_balanced-allocated_nodes is empty, return default  0.0")
            return 0.0
        # 初始化稀疏矩阵
        row_rates = dok_matrix((1, self.max_row), dtype=np.float32)
        col_rates = dok_matrix((1, self.max_col), dtype=np.float32)
        layer_rates = dok_matrix((1, self.max_layer), dtype=np.float32)
        #填充数据
        for node, sku, _ in allocated_nodes:
            rate = self.pending_Loc[sku]['rate']
            row, col, layer = self.pos[node]
            # 映射索引 - 1
            row_rates[0, row-1] += rate
            col_rates[0, col-1] += rate
            layer_rates[0, layer-1] += rate
        # 计算标准差函数（适配稀疏矩阵）
        def sparse_std(matrix):
            # 将 dok_matrix 转换为 csr_matrix
            matrix = matrix.tocsr()
            n_nonzero = matrix.getnnz()  # 非零元素数量
            if n_nonzero < 2 :
                return 0.0
            values = matrix.data     # 直接获取非零值数组
            mean = values.mean()     # 计算均值
            # variance = sum((x - mean)**2 for x in values) / (n_nonzero - 1)
            variance = np.var(values, ddof=1)  # ddof=1 表示样本方差
            return np.sqrt(variance)  # 标准差
        #标准差计算
        row_std = sparse_std(row_rates)
        col_std = sparse_std(col_rates)
        layer_std = sparse_std(layer_rates)
        return row_std + col_std + layer_std

    # def safe_standard_deviation(values,max_count):#
        #     n = len(values)
        #     if n < 2:
        #         warnings.warn(f"安全计算标准差（处理样本数不足的情况）!,return 0.0 {values}", UserWarning)
        #         return 0.0
        #     mean = sum(values) / max_count   # 计算均值
        #     variance = sum((x - mean)**2 for x in values) / (max_count - 1)
        #     return np.sqrt(variance)
        # current_x = defaultdict(float)
        # current_y = defaultdict(float)
        # current_z = defaultdict(float)
        # for node, sku, _ in allocated_nodes:
        #     rate = self.pending_Loc[sku]['rate']
        #     row, col, layer = self.pos[node]
        #     current_x[row] += rate    # 按行累计周转率
        #     current_y[col] += rate    # 按列累计周转率
        #     current_z[layer] += rate     # 按层累计周转率
        # x_rates = list(current_x.values())  # 按行累计周转率
        # y_rates = list(current_y.values())  # 按列累计周转率
        # z_rates = list(current_z.values())  # 按层累计周转率
        # # x_rates = [current_x[i] for i in range(self.max_row)]  # 所有排的周转率
        # # y_rates = [current_y[i] for i in range(self.max_col)]  # 所有列的周转率
        # # z_rates = [current_z[i] for i in range(self.max_layer)]  # 所有层的周转率
        #
        # var_x = safe_standard_deviation(x_rates,len(current_x))#self.max_row
        # var_y = safe_standard_deviation(y_rates,len(current_y))
        # var_z = safe_standard_deviation(z_rates,len(current_z))
        # return var_x + var_y + var_z

    def f3_efficiency(self, allocated_nodes):
        """
        适配新的 allocated_nodes 结构：[(node_id, sku, order), ...]。
        """
        if not allocated_nodes:
            warnings.warn("f3_efficiency-allocated_nodes is empty, return default 0.0")
            return 0.0
        # 批量获取所有路径起始点
        enter_nodes = [self.pending_Loc[sku]["enter_node"] for node, sku, _ in allocated_nodes]
        target_nodes = [node for node, _, _ in allocated_nodes]
        rates = [self.pending_Loc[sku]['rate'] for node, sku, _ in allocated_nodes]
        # 批量查询缓存
        time_cost = []
        for enter, target in zip(enter_nodes, target_nodes):
            # 修改为双层字典访问
            if enter in self.path_cache and target in self.path_cache[enter]:
                pl, tc = self.path_cache[enter][target]
                time_cost.append(tc)
            else:
                # 实时计算并缓存
                path,path_length,explored = self.path_planning.improve_A_star(self.TopoGraph, source=enter_node, target=target)
                tc = self.path_planning.cal_path_time(self.TopoGraph, path)
                self.path_cache.setdefault(enter, {})[target] = (pl, tc)
                time_cost.append(tc)
        # 向量化计算
        return np.dot(np.array(time_cost), np.array(rates))  #path_lengths`是[1,2,3]，`rates`是[4,5,6]，点积就是1*4 + 2*5 + 3*6 = 32

        # efficient = 0.0
        # time_cost = 0
        # for node, sku, _ in allocated_nodes:
        #     sku_id = self.node_sku_map[node]
        #     sku_data = self.pending_Loc[sku_id]
        #     enter_node = sku_data["enter_node"]
        #     rate = sku_data["rate"]
        #     if enter_node == node:
        #         warnings.warn("计算效率时，出入库点相同，跳过！")
        #         return 0.0
        #     try:
        #         # path_length = nx.shortest_path_length(self.TopoGraph, source=enter_node, target=node)
        #         path,path_length,explored = self.path_planning.improve_A_star(self.TopoGraph, source=enter_node, target=node)
        #         # time_cost = self.path_planning.cal_path_time(self.TopoGraph, path)
        #     except nx.NetworkXException:
        #         print("路径规划失败，效率值为1e6")
        #         time_cost = 1e6
        #     # efficient += time_cost * rate
        #     efficient += path_length * rate
        # return efficient


    def calculate_objectives(self, individual):
        """计算个体的目标函数值，并添加未分配货物的惩罚"""
        if not individual.is_decoded:
            # 解码个体，更新 allocated_nodes
            self.decode_individual(individual, self.pending_Loc)
        # 计算目标函数值
        objectives = [f(individual.allocated_nodes) for f in self.objectives]
        # 添加惩罚项
        if hasattr(individual, 'unallocated') and individual.unallocated:
            # 基于未分配数量和 SKU 的 rate 加权
            penalty = 0.0
            for sku, num_unallocated in individual.unallocated.items():
                rate = self.pending_Loc[sku]['rate']  # SKU 的重要性
                penalty += num_unallocated * rate  # 按 rate 加权

            penalty_factor = 10e5  # 调整惩罚系数
            penalty *= penalty_factor

            objectives = [obj + penalty for obj in objectives]
        individual.objectives = objectives
        return individual.objectives



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


    def _precompute_paths(self):
        # 尝试从文件加载缓存
        try:
            with open("path_cache.pkl", "rb") as f:
                self.path_cache = pickle.load(f)
            print("从文件加载路径缓存")
        except FileNotFoundError:
            status = nx.get_node_attributes(self.TopoGraph, 'status')
            targets = [target for target in status if status[target] == 0]
            self.path_cache = {}
            for enter in enter_node:
                self._precompute_for_one_enter(enter, targets, batch_size=100)
            # 保存缓存到文件
            with open("path_cache.pkl", "wb") as f:
                pickle.dump(self.path_cache, f)
            print("路径缓存已保存到文件")

    #将目标节点分批提交到进程池，减少同时运行的进程数量，提高效率
    def _precompute_for_one_enter(self, enter, targets, batch_size=100):
        # 确保 self.path_cache[enter] 存在
        if enter not in self.path_cache:
            self.path_cache[enter] = {}

        # 初始化进度条
        progress = tqdm(total=len(targets), desc=f"预计算入口 {enter}", unit="path")
        targets = list(targets)  # 确保 targets 是列表

        # 分批处理
        for i in range(0, len(targets), batch_size):
            batch_targets = targets[i:i + batch_size]
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._compute_single_path, enter, target)
                           for target in batch_targets]
                for future in as_completed(futures):
                    (_, target), (pl, tc) = future.result()
                    self.path_cache[enter][target] = (pl, tc)
                    progress.update(1)
        progress.close()

    def _compute_single_path(self, enter, target):
        """单次路径计算（适配多进程调用）"""
        try:
            # 计算路径
            path, path_length, _ = self.path_planning.improve_A_star(self.TopoGraph, enter, target )
            # 计算时间
            time_cost = round(self.path_planning.cal_path_time(self.TopoGraph, path), 2)
            return (enter, target), (path_length, time_cost)
        except Exception as e:
            print(f"路径计算失败：从 {enter} 到 {target}，错误：{str(e)}")
            return (enter, target), (float('inf'), float('inf'))


