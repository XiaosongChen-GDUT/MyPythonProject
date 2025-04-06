import copy
import math
import random
from collections import defaultdict
from itertools import chain
from random import randint
import matplotlib.pyplot as plt
import plotly.express as px
from Algorithm.PathPlanning import Path_Planning
from Program.DataModel import Model
import numpy as np
import pandas as pd
import networkx as nx
import geatpy as ea
import warnings

h1 = 3.15  # 1楼换层到2楼的高度
h2 = 2.55  # 2楼换层到3楼的高度
h3 = 1.5  # 3楼高度
heights = [h1, h2, h3]  # 各楼层高度
out_point = [445, 820, 971, 1156]  # 出口点
enter_node = [51, 348, 636, 925, 1110, 1620]# 入口点
fist_connect_point = [642,  674, 1116, 1148]  # 1楼提升机接驳点
second_connect_point = [2374,  2406, 2844, 2876]  # 2楼接驳点
third_connect_point = [3899, 4135]  # 3楼接驳点
'''不纳入历史物资的影响'''
class slap(ea.Problem):
    def __init__(self, model,path_planning):
        '''
        根据货道进行编解码
        1.设置数据模型set_model
        2.设置路径算法对象path_planning
        3.设置相关性矩阵
        4.创建待入库的货物数据
        5.运行算法

        待入库货物数据：
            sku：编号
            入库节点：ID
            质量：质量
            周转率：周转率
            规格：规格
            数量：数量
        '''
        self.set_model(model)
        self.path_planning = path_planning
        # self.pending_Loc = {}  # 待入库货物
        self.node_sku_map = {}  # {货位节点ID: SKU_ID}
        #初始化相关性矩阵
        # self.correlation_matrix = None
        # 初始化货位中心高度
        # self.node_centers = {}
        #初始话历史货物的三维周转率分布
        # self.history_rates = {'x': defaultdict(float), 'y': defaultdict(float), 'z': defaultdict(float)} #defaultdict(float) 表示当访问的键不存在时，会自动返回一个默认的浮点数 0.0。
        self.enter_node = enter_node
        self.out_point = out_point
        self.fist_connect_point = fist_connect_point
        self.second_connect_point = second_connect_point
        self.third_connect_point = third_connect_point

        self._generate_relatedness_matrix(10)  # 生成模拟的货物相关性矩阵（0~1，1表示完全相关）
        self.test_items = self.generateItems_asisles(10,100)      # 生成待入库货物信息

        # 步骤1：获取空闲货道列表,其索引对应货道的序号
        self.free_aisles_keys = [
            aisle_id for aisle_id, aisle in self.asiles.items()
            if aisle['capacity'] > 0
        ]
        # print("空闲货道列表：", self.free_aisles_keys)
        name = 'LocationAssignment_Aisles'  # 初始化name
        self.M = 4  # 初始化M,目标维数
        M = self.M# 初始化M,目标维数
        Dim = len(self.free_aisles_keys)  # 初始化Dim,决策变量维数=空闲货道数量
        varTypes = [1] * Dim  # 初始化varTypes,决策变量的类型，0-连续，1-离散
        maxormins = [1] * M  # 目标最小化
        lb = [1] * Dim  # 变量下界
        ub = [len(self.free_aisles_keys)] * Dim  # 变量上界
        lbin = [1] * Dim  # 决策变量包含下边界
        ubin = [1] * Dim  # 决策变量包含上边界
        # 变量范围（货道ID的最小最大值）
        # print("lb: ",lb," ub: ", ub, "lbin:", len(lbin), 'ubin:', len(ubin), 'varTypes:', len(varTypes))
        # 调用父类构造函数
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    # # 初始化问题
    # def initProblem(self):
    #     if self.pending_Loc == {}:
    #         warnings.warn("No pending items，cannot init problem!", RuntimeWarning)
    #         return
    #     if self.TopoGraph is None:
    #         warnings.warn("No TopoGraph，cannot init problem!", RuntimeWarning)
    #         return
    #     # 步骤1：动态获取空闲货道
    #     self.free_aisles = [
    #         aisle_id for aisle_id, aisle in self.asiles.items()
    #         if aisle['capacity'] > 0
    #     ]
    #     n = len(self.free_aisles)
    #     total_needed = sum(sku['num'] for sku in self.pending_Loc.values())
    #     # 步骤2：校验可行性
    #     total_capacity = sum(aisle['capacity'] for aisle in self.asiles.values())
    #     if total_capacity < total_needed:
    #         raise ValueError("库存容量不足! 需求:{} 可用:{}".format(total_needed, total_capacity))
    #
    #
    #     name = 'LocationAssignment_Aisles'  # 初始化name
    #     M = 4  # 初始化M,目标维数
    #     # Dim = len(self.pending_Loc.keys())  # 初始化Dim,决策变量维数=待入库货物数量
    #     # Dim = len(self.asiles)  # 初始化Dim,决策变量维数=货道数量490,但是其实可以缩小到待入库货物数量
    #     Dim = n  # 初始化Dim,决策变量维数=100
    #     varTypes = [1] * Dim  # 初始化varTypes,决策变量的类型，0-连续，1-离散
    #     maxormins = [1] * M  # 目标最小化
    #     # pending_items = list(self.pending_Loc.values())
    #     # 变量范围（货道ID的最小最大值）
    #     aisle_ids = list(self.asiles.keys())
    #     array = np.array(aisle_ids)
    #     lb = [min(aisle_ids)] * Dim  # 变量下界
    #     ub = [max(aisle_ids)] * Dim   # 变量上界
    #     lbin = [1] * Dim  # 决策变量包含下边界
    #     ubin = [1] * Dim  # 决策变量包含上边界
    #
    #     print("lbin:", len(lbin), 'ubin:', len(ubin), 'varTypes:', len(varTypes))
    #     # 调用父类构造函数
    #     ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    # 目标函数，pop为传入的种群对象
    def aimFunc(self, pop):
        Vars = pop.Phen  # 得到决策变量矩阵,空闲货道列表的索引
        # 使用 ndim 和 shape 属性获取矩阵的形状
        num_individuals, num_cols = Vars.shape
        total_objectives = np.zeros((num_individuals, self.M))  # 存储四个目标函数值

        '''=========解码过程=========='''
        # 将货道分配转换为货位分配
        for i in range(num_individuals):
            if self.node_sku_map:  # 若节点-SKU映射表不为空，则先重置状态
                for node in self.node_sku_map.keys():
                    #复位地图节点状态
                    self.TopoGraph.nodes[node]['status'] = 0
            self.node_sku_map = {}  # 清空节点-SKU映射表
            # free_num = sum(1 for s in nx.get_node_attributes(self.TopoGraph, 'status').values() if s == 0)
            # print("空闲货位数：", free_num)
            # 初始化每一代的货位分配结果
            allocated_nodes = []
            permutation  = Vars[i, :]   # 当前个体的货位分配方案：存放空闲货道序号的数组
            # print("第", i, "个个体方案：", permutation)
            # 每个方案都遍历待入库货物
            for sku, sku_data in self.pending_Loc.items():
                num_needed = sku_data['num']
                # 遍历解码后的货道序列
                for aisle_id in permutation:
                    # 分配货位
                    selected = self.select_location(sku, aisle_id, num_needed)
                    if selected:
                        allocated_nodes.extend(selected)
                        num_needed -= len(selected)
                        if num_needed <= 0:
                            break
                # 未分配完的SKU记录惩罚
                if num_needed > 0:
                    pop.CV[i, 0] += num_needed  # 记录约束违反值
                    print(f"SKU {sku} 未完全分配，剩余 {num_needed} 个")
            #解码完成，记录分配结果
            # print("第", i, "个个体分配结果：", allocated_nodes," 货位数：", len(allocated_nodes))
            # 计算目标函数（需传递allocated_nodes）
            if len(allocated_nodes) == 0:
                warnings.warn("No allocated nodes, return default objectives 0.0")
                total_objectives[i, :] = [1e6, 1e5, 1e4, 1e3]  # Default value for empty input
                continue
            elif len(allocated_nodes) < num_needed:
                warnings.warn("Allocated nodes less than needed, return default objectives 0.0")
                total_objectives[i, :] = [1e6, 1e5, 1e4, 1e3]  # Default value for empty input
                continue
            # 计算目标函数值
            F1 = self.cal_Center_Gravity_nodes(allocated_nodes)
            F2 = self.cal_Efficiency_nodes(allocated_nodes)
            F3 = self.cal_Balanced_distribution_nodes(allocated_nodes)
            F4 = self.cal_Cargo_relatedness_nodes(allocated_nodes)
            # 存储目标函数值
            # total_objectives[i, :] = [F1, F2, F3]
            total_objectives[i, :] = [F1, F2, F3, F4]
            # print("\n 第 ",i," 个个体的目标函数值： F1重心:", F1,"F2最短路径:", F2,"F3层内均匀:", F3,"F4层内相关性:", F4)
        # #对该种群的目标函数值进行归一化
        # normalized = np.zeros_like(total_objectives)
        # epsilon = 1e-10  # 小常数，避免除以零
        # for i in range(self.M):
        #     #获取该目标函数的列
        #     obj_col = total_objectives[:, i]
        #     #可能因目标值全相同而导致NAN
        #     if np.all(total_objectives[:, i] == total_objectives[0, i]):  # 所有值相等
        #         print("所有个体的第：", i, "个目标函数值均相同，无法计算归一化，返回默认值 0")
        #         normalized[:, i] = 0
        #     else:
        #         min_val = obj_col.min()
        #         max_val = obj_col.max()
        #         if max_val - min_val < epsilon:  # 范围太小
        #             normalized[:, i] = 0.5
        #         else:
        #             normalized[:, i] = (obj_col - min_val) / (max_val - min_val)
                    # pop.ObjV = normalized.astype(float)
        # 将归一化结果赋值给目标函数矩阵
        # pop.ObjV = normalized.astype(float)  # 确保数据类型为浮点型
        pop.ObjV = total_objectives.astype(float)  # 确保数据类型为浮点型
        # pop.CV = np.zeros((pop.sizes, 4))  # 若无约束，也需初始化空约束矩阵

    """根据货道ID和货物类型,选择分配的货位节点列表（深度优先存储）"""
    def select_location(self,sku, aisle_id,num_needed):
        """
       从指定货道中选择指定数量的空闲货位
       :param aisle_id: 货道ID
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
            # print("分配sku ",sku," 货道 ",aisle_id," 容量为0，无法分配！ capacity:",capacity)
            return []
        pos = nx.get_node_attributes(self.TopoGraph, 'pos')
        # 获取货道内未分配的货位（假设已预先生成空闲货位列表）
        type = self.asiles[aisle_id]['type']    # 货道类型
        # 计算可分配的货位数
        num_available = min(len(available_nodes), num_needed)
        # 按深度优先存储分配货位
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
            # capacity -= 1
        # # 更新货道容量
        # self.asiles[aisle_id]['capacity'] = capacity
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
                return 0.0   # 无效路径，返回默认值
            # 4. 计算路径时间
            try:
                # path, _, _ = self.path_planning.A_star(
                #     self.TopoGraph,
                #     enter_node,
                #     node
                # )
                # time = self.path_planning.cal_path_time(self.TopoGraph, path)
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
                num = np.random.randint(int(avg_num*0.8), avg_num+1)
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
        # 初始化换层节点
        self.cross_nodes = {1: fist_connect_point, 2: second_connect_point,
                            3: third_connect_point}  # dict_values([[642, 1116, 674, 1148], [2374, 2844, 2406, 2876], [3899, 4135]])
        self.cross_nodes_list = list(
            chain(*self.cross_nodes.values()))  # [642, 1116, 674, 1148, 2374, 2844, 2406, 2876, 3899, 4135]

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



if __name__ == '__main__':
    ref_front = np.array([[0.5, 10.0, 100.0, 50.0],  # 一个理想点
                          [1.0, 20.0, 80.0, 40.0],
                          [1.5, 30.0, 60.0, 30.0],
                          ])
    # 读取数据
    model = Model()
    path_planning = Path_Planning(model)
    problem = slap(model, path_planning)              # 初始化问题对象

    for item in problem.test_items:
        print(item, problem.test_items[item])
    """======================遗传算法参数设置="""
    Encoding = 'RI' # 编码方式
    NIND = 50 # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
                      problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)
    # if population.Encoding == 'RI':
    #     myAlgorithm.recOper = ea.Recndx(XOVR = 1) # 生成正态分布交叉算子对象
    myAlgorithm.mutOper.Pm = 0.2 # 修改变异算子的变异概率
    myAlgorithm.recOper.XOVR = 0.9 # 修改交叉算子的交叉概率
    myAlgorithm.MAXGEN = 100 # 最大进化代数
    myAlgorithm.logTras = 1 # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1 #设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画； 3：绘制决策空间过程动画）
    myAlgorithm.paretoFront = ref_front  # 使用最终解集作为参考前沿
    [NDSet, population] = myAlgorithm.run()
    print('用时：%s 秒' % myAlgorithm.passTime)
    print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else  print('没有找到可行解！')
    print('最优解集：', NDSet)
    if myAlgorithm.log is not None and NDSet.sizes != 0:
        x = print('GD', myAlgorithm.log['gd'][-1])
    print('IGD', myAlgorithm.log['igd'][-1])
    print('HV', myAlgorithm.log['hv'][-1])
    print('Spacing', myAlgorithm.log['spacing'][-1])
    if NDSet.sizes != 0:
        print('最优解：', NDSet.ObjV[0][0])
        print('最优解对应的染色体：')
        for i in range(NDSet.Phen.shape[1]):
            print(NDSet.Phen[0][i])
    """======================进化过程指标追踪分析=================="""
    metricName = [['igd'], ['hv'],['spacing']]
    Metrics = np.array([myAlgorithm.log[metricName[i][0]] for i in
                        range(len(metricName))]).T
    # 绘制指标追踪分析图
    ea.trcplot(Metrics, labels=metricName, titles=metricName)
    fig = px.scatter_3d(NDSet.ObjV, x=0, y=1, z=2, labels={'x':'F1', 'y':'F2', 'z':'F3'})
    fig.show()
