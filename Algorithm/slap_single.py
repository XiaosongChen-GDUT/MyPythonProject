import copy
import math
import random
from collections import defaultdict
from itertools import chain
from random import randint
import matplotlib.pyplot as plt

from Algorithm.PathPlanning import Path_Planning
from Program.DataModel import Model
import numpy as np
import pandas as pd
import networkx as nx
import geatpy as ea
import warnings
row_space = 1.54  # 排间距
h1 = 3.15  # 1楼换层到2楼的高度
h2 = 2.55  # 2楼换层到3楼的高度
h3 = 1.5  # 3楼高度
heights = [h1, h2, h3]  # 各楼层高度
Acc_car = 0.3;  # 四向车加速度
Dec_car = 0.3;  # 四向车减速度
Max_speed_car = 1.2;  # 四向车最大速度
Switching_time = 4;  # 换向时间
Acc_lift = 0.15;  # 提升机加速度
Dec_lift = 0.15;  # 提升机减速度
Max_speed_lift = 1.4;  # 提升机最大速度

out_point = [445, 820, 971, 1156]  # 出口点
enter_node = [51, 348, 636, 925, 1110, 1620]# 入口点
fist_connect_point = [642,  674, 1116, 1148]  # 1楼提升机接驳点
second_connect_point = [2374,  2406, 2844, 2876]  # 2楼接驳点
third_connect_point = [3899, 4135]  # 3楼接驳点
'''测试单目标遗传算法'''
class slap_single(ea.Problem):
    def __init__(self):
        '''
        根据货道进行编解码
        1.设置数据模型set_model
        2.设置路径算法对象path_planning
        3.设置相关性矩阵
        4.创建历史货物数据
        5.创建待入库的货物数据
        3.运行算法

        待入库货物数据：
            sku：编号
            入库节点：ID
            质量：质量
            周转率：周转率
            规格：规格
            数量：数量
        '''

        self.history_Loc = {}   # 历史货物
        self.pending_Loc = {}  # 待入库货物
        self.asiles = {}  # 货道
        self.free_aisles = []  # 新增：存储当前空闲货道ID列表
        self.node_sku_map = {}  # {货位节点ID: SKU_ID}
        #初始化相关性矩阵
        self.correlation_matrix = None
        #历史货物层信息：计算相关性用
        # self.history_layers = []     # id:层号
        # 初始化 历史货物的质量矩,历史货物的总质量
        # self.history_z_moment, self.history_total_mass = 0, 0
        # 初始化货位中心高度
        self.node_centers = {}
        #初始话历史货物的三维周转率分布
        # self.history_rates = {'x': defaultdict(float), 'y': defaultdict(float), 'z': defaultdict(float)} #defaultdict(float) 表示当访问的键不存在时，会自动返回一个默认的浮点数 0.0。
        self.enter_node = enter_node
        self.out_point = out_point
        self.fist_connect_point = fist_connect_point
        self.second_connect_point = second_connect_point
        self.third_connect_point = third_connect_point

    # todo: 货道的编码+剩余货位数？ 深度优先存储
    # 初始化问题
    def initProblem(self):
        if self.pending_Loc == {}:
            warnings.warn("No pending items，cannot init problem!", RuntimeWarning)
            return
        if self.TopoGraph is None:
            warnings.warn("No TopoGraph，cannot init problem!", RuntimeWarning)
            return
        # 步骤1：动态获取空闲货道
        self.free_aisles = [
            aisle_id for aisle_id, aisle in self.asiles.items()
            if aisle['capacity'] > 0
        ]
        print("空闲货道数：", len(self.free_aisles))
        n = len(self.free_aisles)
        total_needed = sum(sku['num'] for sku in self.pending_Loc.values())
        # 步骤2：校验可行性
        total_capacity = sum(aisle['capacity'] for aisle in self.asiles.values())
        if total_capacity < total_needed:
            raise ValueError("库存容量不足! 需求:{} 可用:{}".format(total_needed, total_capacity))


        name = 'LocationAssignment_Aisles'  # 初始化name
        M = 1  # 初始化M,目标维数
        # Dim = len(self.pending_Loc.keys())  # 初始化Dim,决策变量维数=待入库货物数量
        # Dim = len(self.asiles)  # 初始化Dim,决策变量维数=货道数量490,但是其实可以缩小到待入库货物数量
        Dim = n  # 初始化Dim,决策变量维数=100
        varTypes = [1] * Dim  # 初始化varTypes,决策变量的类型，0-连续，1-离散
        maxormins = [1] * M  # 目标最小化
        # pending_items = list(self.pending_Loc.values())
        # 变量范围（货道ID的最小最大值）
        aisle_ids = list(self.asiles.keys())
        lb = [min(aisle_ids)] * Dim  # 变量下界
        ub = [max(aisle_ids)] * Dim   # 变量上界
        lbin = [1] * Dim  # 决策变量包含下边界
        ubin = [1] * Dim  # 决策变量包含上边界

        print("lbin:", len(lbin), 'ubin:', len(ubin), 'varTypes:', len(varTypes))
        # 调用父类构造函数
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    # 目标函数，pop为传入的种群对象
    def aimFunc(self, pop):
        Vars = pop.Phen  # 得到决策变量矩阵,货道编号
        # 建立索引到货道ID的映射
        aisle_ids = np.array(self.free_aisles)
        # 使用 ndim 和 shape 属性获取矩阵的形状
        num_individuals, num_cols = Vars.shape
        obj_values = np.zeros((num_individuals, 4))  # 存储目标函数值
        pop.CV = np.array([[0]] * pop.sizes)  # 每个个体单独设置

        '''=========解码过程=========='''
        # 将货道分配转换为货位分配
        for i in range(num_individuals):
            #分配节点
            nodes = self.node_sku_map.keys()
            for node in nodes:#复位状态
                self.TopoGraph.nodes[node]['status'] = 0
                # print("节点", node, "复位状态")
            self.node_sku_map = {}  # 清空节点-SKU映射表
            free_num = sum(1 for s in nx.get_node_attributes(self.TopoGraph, 'status').values() if s == 0)
            print("空闲货位数：", free_num)
            # 新增：每个个体处理前重置状态
            # 初始化每一代的货位分配结果
            allocated_nodes = []
            permutation  = Vars[i, :]
            unique_aisles = np.unique(permutation)
            decoded_aisles = [aisle_ids[idx] for idx in unique_aisles if idx < len(aisle_ids)] # 遍历每个货物SKU（假设pending_Loc按SKU组织）
            for sku, sku_data in self.pending_Loc.items():
                dimension = sku_data['dimension']
                num_needed = sku_data['num']
                # 遍历解码后的货道序列
                for aisle_id in decoded_aisles:
                    # 分配货位
                    selected = self.select_location(
                        sku, aisle_id, dimension, num_needed
                    )
                    if selected:
                        allocated_nodes.extend(selected)
                        num_needed -= len(selected)
                        if num_needed <= 0:
                            break
                # # 遍历货道序列，分配货位
                # for aisle_id in Vars[i, :]:
                #     # print("当前处理的SKU：", sku, "  货道ID：", aisle_id, "  货物类型：", dimension)
                #     # 选择空闲货位
                #     selected = self.select_location(
                #         sku=sku,
                #         aisle_id=int(aisle_id),
                #         dimension=dimension,
                #         num_needed=num_needed
                #     )
                #     if selected:
                #         allocated_nodes.extend(selected)
                #         num_needed -= len(selected)
                #         if num_needed <= 0:
                #             # print(f"SKU {sku} 已分配完毕，剩余 {num_needed} 个 ",selected)
                #             break

                #未分配货物的数量作为约束违反量
                if num_needed > 0:
                    pop.CV[i, 0] += num_needed  # 约束违反惩罚
                    print(f"SKU {sku} 未完全分配，剩余 {num_needed} 个")
            print("第", i, "个个体分配结果：", allocated_nodes," 货位数：", len(allocated_nodes))

            # 计算目标函数（需传递allocated_nodes）
            if len(allocated_nodes) == 0:
                warnings.warn("No allocated nodes, return default objectives 0.0")
                obj_values[i, :] = [1e8, 1e8, 1e8, 1e8]  # Default value for empty input
                continue
            elif len(allocated_nodes) < num_needed:
                warnings.warn("Allocated nodes less than needed, return default objectives 0.0")
                obj_values[i, :] = [1e8, 1e8, 1e8, 1e8]  # Default value for empty input
                continue
            F1 = self.cal_Center_Gravity_nodes(allocated_nodes)
            F2 = self.cal_Efficiency_nodes(allocated_nodes)
            F3 = self.cal_Balanced_distribution_nodes(allocated_nodes)
            F4 = self.cal_Cargo_relatedness_nodes(allocated_nodes)
            print("\n 第 ",i," 个个体的目标函数值： F1重心:", F1,"F2最短路径:", F2,"F3层内均匀:", F3,"F4层内相关性:", F4)
            # 存储目标函数值
            obj_values[i, :] = [F1, F2, F3, F4]
            # total_objectives[i, :] = [F1+F2+ F3+F4]

        # === 归一化处理 ===
        min_vals = np.min(obj_values, axis=0)
        max_vals = np.max(obj_values, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1e-8  # 防止除以零

        # 处理每个目标的优化方向：
        # F1: 重心-最小化 → 原值
        # F2: 路径时间-最小化 → 原值
        # F3: 方差-最小化 → 原值
        # F4: 相关性-最大化 → 取反
        normalized = np.zeros_like(obj_values)
        normalized[:, 0] = (obj_values[:, 0] - min_vals[0]) / ranges[0]  # F1
        normalized[:, 1] = (obj_values[:, 1] - min_vals[1]) / ranges[1]  # F2
        normalized[:, 2] = (obj_values[:, 2] - min_vals[2]) / ranges[2]  # F3
        normalized[:, 3] = (obj_values[:, 3]-min_vals[3]) / ranges[3]  # F4
        # normalized[:, 3] = (max_vals[3] - obj_values[:, 3]) / ranges[3]  # F4取反

        # === 加权求和（可根据需求调整权重）===
        weights = [0.5, 0.5, 0.25, 0.25]  # 示例权重
        total_objectives = np.sum(normalized * weights, axis=1).reshape(-1, 1)# 按行求和

        pop.ObjV = total_objectives.astype(float)
        # 将计算结果赋值给目标函数矩阵
        # pop.CV = np.zeros((pop.sizes, 4))  # 若无约束，也需初始化空约束矩阵

    """根据货道ID和货物类型,选择分配的货位节点列表（深度优先存储）"""
    def select_location(self,sku, aisle_id, dimension,num_needed):
        """
       从指定货道中选择指定数量的空闲货位
       :param aisle_id: 货道ID
       :param dimension: 货物类型（需与货道type匹配）
       :param num_needed: 需要分配的货位数
       :return: 分配的货位节点列表
       """
        aisle = self.asiles.get(aisle_id, None)
        # capacity = aisle['capacity']  # 货道容量
        aisle_nodes = self.asiles[aisle_id]['nodes']
        status = nx.get_node_attributes(self.TopoGraph, 'status')
        available_nodes = [node for node in aisle_nodes
                           if status[node] == 0]
        capacity = len(available_nodes)  # 货道容量
        if not aisle or capacity == 0:
            # print("货道 ",aisle_id," 容量为0，无法分配！ capacity:",capacity)
            return []
        # # 校验货道类型是否匹配，暂时还没有实现
        # if aisle['dimension'] != dimension:
        #     return []
        pos = nx.get_node_attributes(self.TopoGraph, 'pos')
        # 获取货道内的货位列表
        # 获取货道内未分配的货位（假设已预先生成空闲货位列表）
        type = self.asiles[aisle_id]['type']    # 货道类型
        # 计算可分配的货位数
        num_available = min(len(available_nodes), num_needed)
        # 按深度优先存储分配货位
        # if type == 0:#中位存储
        #      pass
        if type == -1:#向左存储
            # 按列升序选择货位
            selected = sorted(available_nodes, key=lambda node: pos[node][1])[:num_available]
        else:#向右存储
            # 按列降序选择货位
            selected = sorted(available_nodes, key=lambda node: pos[node][1], reverse=True)[:num_available]
        # print("sku: ",sku," num_needed: ",num_needed,"  aisle_id：", aisle_id,"  dimension：", dimension,"  selected: ",selected)
        # 标记货位为已占用
        for node in selected:
            self.TopoGraph.nodes[node]['status'] = 1
            self.node_sku_map[node] = sku  # current_sku为当前处理的SKU ID
            capacity -= 1
        # 更新货道容量
        self.asiles[aisle_id]['capacity'] = capacity
        return selected

    '''形参为分配的货位节点列表，返回目标函数值F2，标量值'''
    def cal_Center_Gravity_nodes(self, allocated_nodes):
        """
        基于货位节点列表计算垂直质心（适配SKU多货物场景）
        :param allocated_nodes: 当前个体分配的货位节点ID列表，如 [101, 205, 307]
        :return: 质心高度（标量值）
        """
        if not allocated_nodes:
            warnings.warn("Gravity-allocated_nodes is empty, return default centroid 0.0")
            return 0.0  # Default value for empty input
        total_moment = 0  # 质量矩
        total_mass = 0    # 总质量
        for node in allocated_nodes:
            sku = self.node_sku_map[node]  # 获取货位对应的 SKU
            quality = self.pending_Loc[sku]['quality']  # 当前 SKU 的质量
            layer_center = self.node_centers[node]  # 货位中心坐标
            total_moment += quality * layer_center
            total_mass += quality
        if total_mass == 0:
            warnings.warn("Total mass is zero, returning default centroid 0.0")
            return 0.0
        centroid = total_moment / total_mass
        return round(centroid, 3)


    def cal_Efficiency_nodes(self, allocated_nodes):
        """
       基于货位节点列表计算出入库效率（适配SKU多货物场景）
       :param allocated_nodes: 分配的货位节点ID列表，如 [101, 205, 307]
       :return: 总加权时间成本（标量值）
       """
        # 效率 = 路径时间 * 周转率
        F2 = 0.0  # 初始化目标函数
        pending_items = list(self.pending_Loc.values())
        # === 遍历每个分配的货位节点 ===
        for node in allocated_nodes:
            # 1. 获取当前货位对应的SKU信息
            sku_id = self.node_sku_map[node]  # 通过预建立的映射表获取SKU ID
            sku_data = self.pending_Loc[sku_id]
            # 2. 提取SKU参数
            enter_node = sku_data["enter_node"]  # 入库点
            rate = sku_data["rate"]              # 周转率
            # 3. 跳过无效路径（入口点与货位相同）
            if enter_node == node:
                warnings.warn("计算效率时，出入库点相同，跳过！")
                continue
            # 4. 计算路径时间
            try:
                path_length = nx.shortest_path_length(self.TopoGraph, source=enter_node, target=node)
            except nx.NetworkXException:  # 路径不存在
                path_length = 1e6  # 高惩罚值

            # 5. 累加效率值（时间 × 周转率）
            F2 += path_length * rate
        return F2

    def cal_Balanced_distribution_nodes(self, allocated_nodes):
        """
        基于货位节点列表计算周转率均衡分布（适配SKU多货物场景）
        :param allocated_nodes: 分配的货位节点ID列表，如 [101, 205, 307]
        :return: 周转率分布方差总和（标量值）
        """
        if not allocated_nodes:
            return 0.0  # Default value for empty input
        # === 计算各维度方差 ===
        def safe_variance(values):
            """安全计算方差（处理样本数不足的情况）"""
            n = len(values)
            if n < 2:
                print(values)
                warnings.warn(f"安全计算方差（处理样本数不足的情况）!,return 0.0 {values}",UserWarning)
                return 0.0
            mean = sum(values) / n
            return sum((x - mean)**2 for x in values) / (n - 1)
        current_x = defaultdict(float)  # X 方向周转率
        current_y = defaultdict(float)  # Y 方向周转率
        current_z = defaultdict(float)  # Z 方向周转率
        for node in allocated_nodes:
            x, y, z = self.pos[node]  # 货位坐标
            sku_id = self.node_sku_map[node]  # SKU ID
            rate = self.pending_Loc[sku_id]['rate']  # 当前 SKU 周转率
            current_x[y] += rate
            current_y[x] += rate
            current_z[z] += rate

        var_x = safe_variance(list(current_x.values()))
        var_y = safe_variance(list(current_y.values()))
        var_z = safe_variance(list(current_z.values()))

        return var_x + var_y + var_z

    def cal_Cargo_relatedness_nodes(self, allocated_nodes):
        """
        基于货位节点列表计算层内SKU相关性（适配SKU多货物场景）
        :param allocated_nodes: 分配的货位节点ID列表，如 [101, 205, 307]
        :return: 层内SKU相关性总和（标量值）
        """
        total_correlation = 0.0
        # === 1. 构建层-SKU分布（含历史数据） ===
        layer_skus = defaultdict(list)  # {层号: [SKU1, SKU2, ...]}
        # 添加当前分配货物
        for node in allocated_nodes:
            layer = self.pos[node][2]
            sku_id = self.node_sku_map[node]  # 通过映射表获取SKU
            layer_skus[layer].append(sku_id)
        # === 2. 计算各层相关性 ===
        for layer, skus in layer_skus.items():
            # 统计SKU出现次数
            sku_counter = defaultdict(int)
            for sku in skus:
                sku_counter[sku] += 1
            # 生成唯一有序SKU列表
            unique_skus = sorted(sku_counter.keys())
            n = len(unique_skus)
            # 遍历所有SKU组合 (i < j)
            for i in range(n):
                sku1 = unique_skus[i]
                count1 = sku_counter[sku1]
                for j in range(i+1, n):
                    sku2 = unique_skus[j]
                    count2 = sku_counter[sku2]
                    # 获取相关性系数
                    corr = self.correlation_matrix[sku1][sku2]
                    total_correlation += corr * count1 * count2
        return total_correlation


    """生成货物相关性矩阵"""
    def _generate_relatedness_matrix(self,sku_items):
        """生成模拟的货物相关性矩阵（0~1，1表示完全相关）"""
        np.random.seed(42)      # 设置随机种子
        correlation = np.random.rand(sku_items, sku_items)  # 随机生成相关性矩阵
        correlation = (correlation + correlation.T) / 2  # 对称矩阵
        np.fill_diagonal(correlation, 0)  # 对角线为0
        self.correlation_matrix = correlation
        return correlation

    """预计算所有节点的层中心高度"""
    def _precompute_node_centers(self):
        node_centers = {}
        for node in self.pos:
            (x, y, z) = self.pos[node]
            cumulative_height = self.cumulative_heights[z - 1]  # 前z-1层总高度
            current_layer_center = cumulative_height + heights[z - 1] / 2  # 前z-1层总高度+当前层高度/2
            node_centers[node] = current_layer_center  # 节点的层中心。在后续处理中，只需要通过节点编号直接获取层中心值，而不需要每次重新计算。
        return node_centers


    '''========================以下为辅助函数=========================='''
    '''根据货道编解码的入库数据生成待入库货物
    {SKU:(enter_node,num,rate,quality,dimension)}
    sku_items: 待入库SKU总数
    Num: 待入库货物数量总数
    '''
    def generateItems_asisles(self,sku_items ,Num):
        # 添加矩阵初始化
        if self.correlation_matrix is None:
            self.correlation_matrix = self._generate_relatedness_matrix(sku_items)  # 示例随机矩阵
        np.random.seed(42)      # 设置随机种子
        pending_Loc = {}
        sku_num = self.correlation_matrix.shape[0]   # SKU种类数
        remaining_Num = Num  # 剩余需要分配的数量
        avg_num = remaining_Num // sku_items  # 平均分配数量
        #将NUM依次随机消耗掉，直到所有SKU都被消耗完
        for i in range(sku_items):
            if i == sku_items-1:
                num = remaining_Num
            else:
                num = np.random.randint(int(avg_num*0.5), avg_num+1)
            sku = random.randint(0, sku_num-1)   # 随机选择SKU
            while sku in pending_Loc:  # 如果重复，重新选择
                sku = random.randint(0, sku_num - 1)
            rate = round(np.random.uniform(0.01, 1), 2)  # 随机生成周转率
            dimension = np.random.choice(['A', 'B', 'C', 'D'])  # 随机选择货物类型
            enter_node = random.choice(self.enter_node)  # 随机选择入库口
            if dimension == 'A':
                quality = np.random.randint(1, 51)  # A，质量值将在 1 到 50 之间随机生成
            elif dimension == 'D':
                quality = np.random.randint(150, 201)  # D，质量值将在 150 到 200 之间随机生成
            else:
                quality = np.random.randint(51, 150)  # 对于 B 和 C，质量值在 51 到 150 之间
            pending_Loc[sku] = {
                'enter_node': enter_node,
                'rate': rate,
                'quality': quality,
                'dimension': dimension,
                'num':num
            }
            remaining_Num -= num  # 更新剩余需要分配的数量
        self.pending_Loc = pending_Loc
        return pending_Loc



    # 设置模型对象
    def set_model(self, model):
        self.model = model
        # 图结构
        self.TopoGraph = model.combined_graph
        # 图中节点的二维画布坐标
        self.location = nx.get_node_attributes(self.TopoGraph, 'location')
        # 图中节点的三维坐标
        self.pos = nx.get_node_attributes(self.TopoGraph, 'pos')
        # 货道
        self.asiles = model.aisles
        # # 初始化换层节点
        # self.cross_nodes = {1: fist_connect_point, 2: second_connect_point,
        #                     3: third_connect_point}  # dict_values([[642, 1116, 674, 1148], [2374, 2844, 2406, 2876], [3899, 4135]])
        # self.cross_nodes_list = list(
        #     chain(*self.cross_nodes.values()))  # [642, 1116, 674, 1148, 2374, 2844, 2406, 2876, 3899, 4135]

        # 预计算每层的累计高度
        self.cumulative_heights = [0] * len(heights)
        for i in range(len(heights)):
            self.cumulative_heights[i] = sum(heights[:i])
        # print(f"cumulative_heights={self.cumulative_heights}")
        # 预计算节点的中心高度
        self.node_centers = self._precompute_node_centers()

    # 设置路径规划对象
    def set_path_planning(self, path_planning):
        self.path_planning = path_planning

    @staticmethod
    def test_cal_Balanced_distribution_nodes():
        # 初始化问题对象
        problem = slap_single()
        problem.set_model(Model())
        problem.set_path_planning(Path_Planning(problem.model))
        correlation = problem._generate_relatedness_matrix(10)  # 生成模拟的货物相关性矩阵（0~1，1表示完全相关）

        # 测试数据
        test_items = {
            0: {'enter_node': 51, 'rate': 0.5, 'quality': 100, 'dimension': 'A', 'num': 10},
            1: {'enter_node': 348, 'rate': 0.8, 'quality': 150, 'dimension': 'D', 'num': 15},
            2: {'enter_node': 636, 'rate': 0.5, 'quality': 200, 'dimension': 'B', 'num': 20}
        }
        problem.pending_Loc = test_items
        problem.initProblem()

        # 模拟分配的货道节点ID列表
        allocated_nodes = [51, 348, 636]

        # 调用目标函数计算周转率均衡分布
        result = problem.cal_Balanced_distribution_nodes(allocated_nodes)

        # 预期结果计算
        expected_x = defaultdict(float)
        expected_y = defaultdict(float)
        expected_z = defaultdict(float)

        for node in allocated_nodes:
            x, y, z = problem.pos[node]
            sku_id = problem.node_sku_map[node]
            rate = problem.pending_Loc[sku_id]['rate']
            expected_x[y] += rate
            expected_y[x] += rate
            expected_z[z] += rate
        print("expected_x.values():",list(expected_x.values()))
        print("expected_y.values():",list(expected_y.values()))
        print("expected_z.values():",list(expected_z.values()))
        expected_var_x = problem.safe_variance(list(expected_x.values()))
        expected_var_y = problem.safe_variance(list(expected_y.values()))
        expected_var_z = problem.safe_variance(list(expected_z.values()))

        expected_result = expected_var_x + expected_var_y + expected_var_z

        # 验证结果是否符合预期
        assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

        print("Test passed!")

if __name__ == '__main__':
    # 读取数据
    model = Model()
    path_planning = Path_Planning(model)
    problem = slap_single()              # 初始化问题对象
    problem.set_model(model)                    # 设置模型对象
    problem.set_path_planning(path_planning)    # 设置路径规划对象
    correlation = problem._generate_relatedness_matrix(100)  # 生成模拟的货物相关性矩阵（0~1，1表示完全相关）
    test_items = problem.generateItems_asisles(10,100)      # 生成待入库货物信息
    for item in test_items:
        print(item, test_items[item])
    # 测试数据
    # test_items = {0: {'enter_node': 51, 'rate': 0.5, 'quality': 100, 'dimension': 'A', 'num': 10},
    #               1: {'enter_node': 348, 'rate': 0.8, 'quality': 150, 'dimension': 'D', 'num': 15},
    #               2: {'enter_node': 636, 'rate': 0.5, 'quality': 200, 'dimension': 'B', 'num': 20}}
    # problem.pending_Loc = test_items
    # problem.initProblem()  # 初始化问题
    # for item in test_items:
    #     print(item, test_items[item])
    #
    # # 快速构建算法
    # algorithm = ea.soea_SGA_templet(
    #     problem,
    #     ea.Population(Encoding='RI', NIND=50),
    #     MAXGEN=100,  # 最大进化代数
    #     logTras=1,
    #     verbose=True,
    #     record_mode=True,
    #     save_mode=True,
    #     plot_mode=True,)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # res = ea.optimize(algorithm,
    #                   verbose=True,
    #                   drawing=1,
    #                   outputMsg=True,
    #                   logTras=1,
    #                   drawLog=True)
    # print('最佳目标函数值：%s' % res['ObjV'][0][0])
    # print('最佳可行解：%s' % res['X'][0])