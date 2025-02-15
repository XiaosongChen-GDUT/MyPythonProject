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
fist_connect_point = [642, 1116, 674, 1148]  # 1楼提升机接驳点
second_connect_point = [2374, 2844, 2406, 2876]  # 2楼接驳点
third_connect_point = [3899, 4135]  # 3楼接驳点

class LocationAssignment_Aisles(ea.Problem):
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

            货物（托盘)：ID,
            'enter_node'(入库节点)：ID,
            'rate': rate,
            'quality': quality,
            'dimension': dimension
            'sku': sku = 0,1,2,3....

        历史货物数据格式：
            货物编号：货位id、质量、周转率、规格、sku
            'location_node':货位ID
            'rate': rate,
            'quality': quality,
            'dimension': dimension
            'sku': sku
        '''

        self.history_Loc = {}   # 历史货物
        self.pending_Loc = {}  # 待入库货物
        self.asiles = {}  # 货道
        #初始化相关性矩阵
        self.correlation_matrix = None
        #历史货物层信息：计算相关性用
        self.history_layers = []     # id:层号
        # 初始化 历史货物的质量矩,历史货物的总质量
        self.history_z_moment, self.history_total_mass = 0, 0
        # 初始化货位中心高度
        self.node_centers = {}
        #初始话历史货物的三维周转率分布
        self.history_rates = {'x': defaultdict(float), 'y': defaultdict(float), 'z': defaultdict(float)} #defaultdict(float) 表示当访问的键不存在时，会自动返回一个默认的浮点数 0.0。
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

        # 图中节点的状态：-1-储货点，0-空闲，1-占用
        self.status = nx.get_node_attributes(self.TopoGraph, 'status')
        # 空闲货位集合
        self.free_status = [i for i in self.status if self.status[i] == 0]
        # 空闲货位,按维度分类A\B\C\D
        self.free_Loc = {}
        for node in self.TopoGraph.nodes():
            if self.TopoGraph.nodes[node]['status'] == 0:
                dimension = self.TopoGraph.nodes[node]['dimension']
                if dimension not in self.free_Loc:
                    self.free_Loc[dimension] = []
                self.free_Loc[dimension].append(node)

        min_index = {'A': min(self.free_Loc['A']), 'B': min(self.free_Loc['B']), 'C': min(self.free_Loc['C']),
                     'D': min(self.free_Loc['D'])}
        max_index = {'A': max(self.free_Loc['A']), 'B': max(self.free_Loc['B']), 'C': max(self.free_Loc['C']),
                     'D': max(self.free_Loc['D'])}

        name = 'LocationAssignment'  # 初始化name
        M = 4  # 初始化M,目标维数
        Dim = len(self.pending_Loc)  # 初始化Dim,决策变量维数=待入库货物数量
        varTypes = [1] * Dim  # 初始化varTypes,决策变量的类型，0-连续，1-离散
        maxormins = [1] * M  # 目标最小化
        pending_items = list(self.pending_Loc.values())
        lb = [min_index[item['dimension']] for item in pending_items]
        ub = [max_index[item['dimension']] for item in pending_items]
        lbin = [1] * Dim  # 决策变量包含下边界
        ubin = [1] * Dim  # 决策变量包含上边界

        # print('min_index:', min_index)
        # print('max_index:', max_index)
        # print("lb:", lb)
        # print("ub:", ub)
        # print("lbin:", len(lbin), 'ubin:', len(ubin), 'varTypes:', len(varTypes))
        # 调用父类构造函数
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    # 目标函数，pop为传入的种群对象
    def aimFunc(self, pop):
        Vars = pop.Phen  # 得到决策变量矩阵
        # 使用 ndim 和 shape 属性获取矩阵的形状
        # num_rows, num_cols = Vars.shape

        F1 = self.cal_Center_Gravity(Vars)          # 计算质量重心
        F2 = self.cal_Efficiency(Vars)              # 效率
        F3 = self.cal_Balanced_distribution(Vars)   # 均衡分布
        F4 = self.cal_Cargo_relatedness(Vars)       # 货物相关性
        # print("F1:",F1)
        # print("F2:",F2)
        # print("F3:",F3)
        # print("F4:",F4)
        # 合并目标函数值
        pop.ObjV = np.hstack([F1, F2, F3, F4])  #水平堆叠数组（即按列方向堆叠）
        # pop.CV = [0] * pop.sizes            # 约束条件值

    '''计算仓库中所有货物的重心,返回目标函数值F1，列向量'''
    def cal_Center_Gravity(self, Vars):
        """
       计算种群中每个个体的垂直重心
       :param Vars: 决策变量矩阵 (num_individuals, num_pending)
       :return: 重心数组 (num_individuals, 1)
       """
        # 使用 ndim 和 shape 属性获取矩阵的形状
        # num_rows, num_cols = Vars.shape
        # 创建num_rows行的矩阵，每行元素初始化为0，存储目标函数值
        # F1 = np.zeros((num_rows, 1))
        # print('history_z_moment:', self.history_z_moment, '  history_total_mass:', self.history_total_mass)

        # 决策变量的解析
        pending_items = list(self.pending_Loc.values())  # 获取待入库货物列表
        pending_qualities = np.array([item['quality'] for item in pending_items])
        total_pending_mass = np.sum(pending_qualities)
        # 计算总质量
        total_mass = self.history_total_mass + total_pending_mass
        if total_mass == 0:
            warnings.warn("Total mass is 0，cal_Center_Gravity() return n行1列的0矩阵!")
            return np.zeros((Vars.shape[0], 1))  # 所有重心为0
        # 将Vars中的节点转换为层中心的矩阵
        layer_centers = np.vectorize(lambda node: self.node_centers[node])(
            Vars)  # 使用了 numpy 库中的 vectorize 函数来简化对 Vars 矩阵中每个元素的操作
        # 计算每个个体的总质量矩（待入库部分）
        z_moments = np.dot(layer_centers, pending_qualities)
        # print('layer_centers:', layer_centers)
        # print('pending_qualities:', pending_qualities, 'z_moments:', z_moments)
        # 个体的质量矩 + 历史质量矩
        total_z_moments = z_moments + self.history_z_moment
        # 计算重心 保留3位小数 -1 表示根据数组的长度自动计算行数，而列数被显式地指定为 1。
        F1 = np.round(total_z_moments / total_mass, 3).reshape((-1, 1))  # 保留3位小数，并转为列向量
        return F1

    # todo '''均衡分布：货道X,Y,Z方向上的周转率之和表示拥挤程度，标准差反应均衡程度'''
    def cal_Balanced_distribution(self, Vars):
        num_individuals = Vars.shape[0]  # 种群规模
        F3 = np.zeros(num_individuals)  # 初始化目标函数
        # 获取待入库货物的周转率列表
        pending_items = list(self.pending_Loc.values())
        # pending_rates = [item['rate'] for item in pending_items]
        def calculate_variance(values):
            """计算方差（样本方差，分母为n-1）"""
            if len(values) < 2:
                # print("样本数小于2，无法计算方差！")
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            return variance
        # 解析决策变量矩阵
        for i in range(num_individuals):
            # 复制历史数据
            current_x = defaultdict(float, self.history_rates['x'])
            current_y = defaultdict(float, self.history_rates['y'])
            current_z = defaultdict(float, self.history_rates['z'])
            # 解析当前个体的分配方案
            for j in range(Vars.shape[1]):          # 遍历一行中货位分配方案
                node = Vars[i, j]                   # 分配的货位节点
                rate = pending_items[j]['rate']     # 周转率
                x, y, z = self.pos[node]            # 获取排(x)、列(y)、层(z)坐标
                # 累加周转率
                current_x[y] += rate
                current_y[x] += rate
                current_z[z] += rate
                # print("分配货位：",node," 周转率：",rate," 坐标：",x,y,z," 当前周转率：",current_x[y],current_y[x],current_z[z])
            # 计算方差（标准差公式中的分子部分）
            s_x = calculate_variance(list(current_x.values()))
            s_y = calculate_variance(list(current_y.values()))
            s_z = calculate_variance(list(current_z.values()))
            F3[i] = s_x + s_y + s_z
            # print("个体：", i, " 的均衡分布："," s_x:",s_x," s_y:",s_y," s_z:",s_z," F3:", F3[i])
        return F3.reshape((-1, 1))  # 转为列向量

    # '''货物相关性：降低高相关货物层间相关性，分层存储'''
    def cal_Cargo_relatedness(self, Vars):
        num_individuals = Vars.shape[0]  # 种群规模
        F4 = np.zeros(num_individuals)  # 初始化目标函数
        for i in range(num_individuals):
            allocation = {j: Vars[i, j] for j in range(Vars.shape[1])}  # 解析当前个体的分配方案  入库列表索引：分配货位节点ID
            # print("分配方案：", allocation)
            #layer_items每层存储的是SKU列表，后续计算时可以直接使用这些SKU进行相关性查询
            layer_skus = defaultdict(list)  #  # 按层存储SKU   层 1 ：[1,2,2,3,4,1...]；层 2 ：[1,1,3,4,4,...]
            # # 添加历史货物（直接使用预计算的层信息）
            for item_id, item_data in self.history_Loc.items():
                node = item_data['location_node']    # 分配的货位节点
                sku = item_data['sku']
                layer = self.pos[node][2]
                layer_skus[layer].append(sku)
            # 添加待入库货物   在处理历史货物时，应该获取其SKU，而不是直接使用item_id
            for j, node in allocation.items():
                # print("待入库货物：",j," 分配节点：",node)
                layer = self.pos[node][2]
                sku = self.pending_Loc[j]['sku']
                layer_skus[layer].append(sku)
            #计算每层的相关性总和（仅遍历n1 < n2）
            total_correlation = 0.0
            for layer, skus in layer_skus.items():
                # print("layer:",layer, "skus:", skus)
                # 统计当前层各SKU的出现次数
                sku_counts = defaultdict(int)
                for sku in skus:
                    sku_counts[sku] += 1
                # 将SKU列表转为唯一有序列表，避免重复计算
                unique_skus = sorted(sku_counts.keys())
                n_skus = len(unique_skus)
                # 计算不同SKU间的组合相关性
                for i_idx in range(n_skus):
                    sku1 = unique_skus[i_idx]
                    count1 = sku_counts[sku1]
                    for j_idx in range(i_idx + 1, n_skus):
                        sku2 = unique_skus[j_idx]
                        count2 = sku_counts[sku2]
                        # 获取相关性，处理缺失值
                        # corr = self.correlation_matrix.get(sku1, {}).get(sku2, 0)
                        corr = self.correlation_matrix[sku1][sku2]
                        total_correlation += corr * count1 * count2
                # print("layer:",layer , total_correlation)
            F4[i] = total_correlation
            # print("个体：", i, " 的相关性：", total_correlation)
        return F4.reshape((-1, 1))  # 转为列向量


    # todo: 提升机选择、出入库口选择、路径规划考虑冲突？
    '''货物出入库时间乘以周转率表征出入库效率'''
    def cal_Efficiency(self, Vars):
        # 路径规划--》选择提升机、出入库口（排队论？）
        # 效率 = 路径时间 * 周转率
        num_individuals = Vars.shape[0]  # 种群规模
        F2 = np.zeros(num_individuals)  # 初始化目标函数
        pending_items = list(self.pending_Loc.values())
        # pending_rates = np.array([item['rate'] for item in pending_items])
        # 解析决策变量矩阵
        for i in range(num_individuals):
            total_efficiency = 0
            for j in range(Vars.shape[1]):  # 遍历一行中货位分配方案
                item = pending_items[j]  # 待入库货物
                rate = item['rate']  # 周转率
                enter_node = item['enter_node']  # 入库节点
                loc_node = Vars[i, j]  # 分配的货位节点
                if enter_node == loc_node:
                    continue  # 入口点和货位相同，无需移动
                try:
                    # todo 寻路算法有待替换
                    # path = nx.shortest_path(self.TopoGraph, enter_node, loc_node, weight='weight')
                    path, cost, explored = self.path_planning.A_star(self.TopoGraph, enter_node, loc_node,heuristic_index = 1, weight='weight')
                except nx.NetworkXException:
                    total_time = 1e6  # 路径不存在，赋予高惩罚值
                else:
                    total_time = self.path_planning.cal_path_time(self.TopoGraph, path)  # 路径时间
                    # print(
                    #     f"item：{item}，入库节点{enter_node}到货位{loc_node}的最短路径{path}，时间：{total_time}，周转率：{rate}，效率：{total_time * rate}")
                # 累加效率（时间乘以周转率）
                total_efficiency += total_time * rate
            # print("个体：", i, " 的效率：", total_efficiency)
            F2[i] = total_efficiency
        return F2.reshape((-1, 1))  # 转为列向量:-1 表示 NumPy 会根据数组的总元素数量自动计算合适的行数;1 表示重塑后的数组有1列。

        # 计算路径时间=响应成本 + 执行成本：
        # 非1楼需计算提升机时间=响应成本 + 执行成本
    '''计算路径时间成本'''
    # def cal_path_time(self, graph, path, Vmax,Acc,Dcc):
    #     time_cost = 0
    #     t_acc = Vmax/Acc   #加速时间
    #     t_dcc = Vmax/Dcc   #减速时间
    #     S_acc = Vmax**2/(2*Acc)   #从0加速到最大速度所需距离
    #     S_dcc = Vmax**2/(2*Dcc)    #从最大速度减速到0所需距离
    #     S0 = S_dcc + S_acc   #临界距离
    #     # print(f"t_acc={t_acc},t_dcc={t_dcc},S_acc={S_acc},S_dcc={S_dcc}，S0={S0}")
    #     '''返回转向节点列表'''
    #     def get_turn_nodes(graph,path):
    #         turn_nodes = []
    #         location = nx.get_node_attributes(graph, 'location')
    #         threshold = 1.0
    #         for i in range(1, len(path) - 1):
    #             parent = path[i - 1]      # 前一个节点
    #             current_node = path[i]    # 当前节点
    #             next_node = path[i + 1]   # 下一个节点
    #             parent_pos = location[parent]
    #             next_pos = location[next_node]
    #             delta_x = abs(parent_pos[0] - next_pos[0])  # x轴偏差
    #             delta_y = abs(parent_pos[1] - next_pos[1])  # y轴偏差
    #             # 判断是否在 x 和 y 轴上都超过阈值
    #             if delta_x > threshold and delta_y > threshold:
    #                 # print("delta_x :",delta_x,"delta_y :",delta_y)
    #                 turn_nodes.append(current_node)
    #         return turn_nodes
    #
    #     '''计算一段子路径时间'''
    #     def cal_time(sub_path):
    #         path_time = 0
    #         length =  nx.path_weight(graph, sub_path, 'weight')
    #         if length >= S0:      # 路径长度超过了从0加速到最大速度所需距离+从最大速度减速到0所需距离
    #             constant_time = (length - S0)/Vmax   # 计算匀速时间
    #             path_time += constant_time + t_acc + t_dcc   # 路径时间 = 匀速时间 + 加速时间 + 减速时间
    #             print(f"足够长的距离加速,sub_path={sub_path},length={length},constant_time={constant_time},path_time={path_time}")
    #         else: #匀加速到非最大速度后，立即做匀减速运动
    #             t = math.sqrt(2*length*(1/Acc + 1/Dcc))
    #             path_time += t
    #             print(f"非最大速度，sub_path={sub_path},length={length},t={t},path_time={path_time}")
    #         return path_time
    #
    #     #二维列表，存储划分后的路径
    #     result = []
    #     #引入转向节点列表
    #     turn_nodes = get_turn_nodes(graph,path)
    #     start_index = 0
    #     #遍历分割点
    #     for point in turn_nodes:
    #         # 找到分割点在原列表中的索引
    #         index = path.index(point,start_index)   #从start_index开始查找
    #         # 将从上次开始索引到当前分割点的子列表添加到结果列表中
    #         sub_path = path[start_index:index+1]       #加1，以包含 index 所指向的元素。
    #         result.append(sub_path)
    #         # 计算子路径时间
    #         time_cost += cal_time(sub_path)
    #         # # 更新开始索引
    #         start_index = index
    #     if start_index < len(path):  # 处理最后一段路径.（即 start_index 仍然指向路径中的某个位置）
    #         sub_path = path[start_index:]
    #         result.append(sub_path)
    #         time_cost += cal_time(sub_path)
    #     turn_count = len(turn_nodes)  # 计算转向次数
    #     turn_time = turn_count * Switching_time  # 计算转向时间
    #     time_cost += turn_time  # 路径时间 = 路径时间 + 转向时间
    #     # print(f"分割点列表：{result}"," , 转向次数：{turn_count} ",turn_count,"，转向时间：{turn_time}",turn_time,"，路径时间：{time_cost}",time_cost)
    #     return time_cost

    # todo: 提升机的定义与初始话！

    '''计算提升机时间成本=当前楼层-》取货楼层-》放货楼层'''
    def cal_ele_time(self, current_floor, get_floor, put_floor, Vmax, Acc, Dcc):
        time_cost = 0
        # 计算提升机时间
        t_acc = Vmax / Acc  # 加速时间
        t_dcc = Vmax / Dcc  # 减速时间
        S_acc = Vmax ** 2 / (2 * Acc)  # 从0加速到最大速度所需距离
        S_dcc = Vmax ** 2 / (2 * Dcc)  # 从最大速度减速到0所需距离
        S0 = S_dcc + S_acc  # 临界高度
        # 获取当前楼层、取货楼层、放货楼层的高度
        current_height = self.cumulative_heights[current_floor - 1]
        get_height = self.cumulative_heights[get_floor - 1]
        put_height = self.cumulative_heights[put_floor - 1]

        # print(f"t_acc={t_acc},t_dcc={t_dcc},S_acc={S_acc},S_dcc={S_dcc}，S0={S0}")
        def cal_time(height_diff):
            if height_diff >= S0:  # 高度超过了从0加速到最大速度所需距离+从最大速度减速到0所需距离
                constant_time = (height_diff - S0) / Vmax  # 计算匀速时间
                time_cost = constant_time + t_acc + t_dcc  # 路径时间 = 匀速时间 + 加速时间 + 减速时间
                # print(f"足够长的高度加速,height_diff={height_diff},constant_time={constant_time},time_cost={time_cost}")
            else:  # 匀加速到非最大速度后，立即做匀减速运动
                time_cost = math.sqrt(2 * height_diff * (1 / Acc + 1 / Dcc))
                # print(f"非足够高度，height_diff={height_diff},time_cost={time_cost}")
            return time_cost

        # 计算响应成本：当前楼层-》取货楼层
        if current_floor == get_floor:  # 当前楼层等于取货楼层
            phase1 = 0
        else:  # 当前楼层不等于取货楼层
            height_diff = abs(get_height - current_height)
            phase1 = cal_time(height_diff)
        # 计算执行成本：取货楼层-》放货楼层
        height_diff = abs(put_height - get_height)
        phase2 = cal_time(height_diff)
        # 总时间
        time_cost = phase1 + phase2
        return time_cost

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

    '''预计算历史货物层信息'''
    def _precompute_history_layers(self):
        """预计算历史货物所在层（z坐标）"""
        # history_layers = []  # 历史货物所在层（z坐标）
        for item_id, item_data in self.history_Loc.items():
            node = item_data['location_node']    # 分配的货位节点
            z = self.pos[node][2]  # 获取层信息（z坐标）
            self.history_layers[item_id] = z
        return self.history_layers

    """预计算历史货物在各轴的周转率分布"""
    def _precompute_history_rates(self):
        history_rates = {'x': defaultdict(float), 'y': defaultdict(float), 'z': defaultdict(float)}
        #遍历历史货物，计算各轴的周转率分布
        for item in self.history_Loc.values():
            node = item['location_node']
            rate = item['rate']
            x, y, z = self.pos[node]
            history_rates['x'][y] += rate    # 累加 x 轴的周转率
            history_rates['y'][x] += rate    # 累加 y 轴的周转率
            history_rates['z'][z] += rate    # 累加 z 轴的周转率
        return history_rates

    """预计算历史货物的质量矩和总质量"""
    def _precompute_history(self):
        history_z_moment = 0  # 历史货物质量矩
        history_total_mass = 0  # 历史货物总质量
        # 计算历史货物的质量
        for item in self.history_Loc.values():
            quality = item['quality']
            node = item['location_node']
            current_layer_center = self.node_centers[node]  # 前z-1层总高度+当前层高度/2
            history_z_moment += quality * current_layer_center
            history_total_mass += quality
        return history_z_moment, history_total_mass

    '''========================以下为辅助函数=========================='''
    # 生成入库数据,随机生成Num个待入库货物
    def generateItems(self, Num):
        np.random.seed(42)      # 设置随机种子
        pending_Loc = {}
        # item_id = random.randint(1, 1000)       #随机起始ID
        item_id = 0
        sku_num = self.correlation_matrix.shape[0]   # SKU种类数
        # 随机生成Num个待入库货物
        for i in range(Num):
            rate = round(np.random.uniform(0.01, 1), 2)  # 随机生成周转率
            dimension = np.random.choice(['A', 'B', 'C', 'D'])  # 随机选择货物类型
            enter_node = random.choice(self.enter_node)  # 随机选择入库口
            if dimension == 'A':
                quality = np.random.randint(1, 51)  # A，质量值将在 1 到 50 之间随机生成
            elif dimension == 'D':
                quality = np.random.randint(150, 201)  # D，质量值将在 150 到 200 之间随机生成
            else:
                quality = np.random.randint(51, 150)  # 对于 B 和 C，质量值在 51 到 150 之间
            pending_Loc[item_id] = {
                'enter_node': enter_node,
                'rate': rate,
                'quality': quality,
                'dimension': dimension,
                'sku': np.random.randint(0, sku_num)  # 随机选择SKU
            }
            item_id += 1
        self.pending_Loc = pending_Loc
        return pending_Loc

    # 添加新的历史记录：质量、周转率、类型和存储货位节点
    def add_history(self, item_id, quality, turnover_rate, dimension, location_node):
        self.history_Loc[item_id] = {
            'quality': quality,
            'rate': turnover_rate,
            'dimension': dimension,
            'location_node': location_node
        }

    # 根据货物id删除特定货物的历史信息
    def remove_item_history(self, item_id):
        if item_id in self.history_Loc:
            del self.history_Loc[item_id]
        else:
            print("No such item in history，cannot delete!")

    # 设置历史已存储信息
    def set_history_Loc(self, history_Loc):
        self.history_Loc = history_Loc
        # 预计算 历史货物的质量矩,历史货物的总质量
        self.history_z_moment, self.history_total_mass = self._precompute_history()
        # 预计算历史周转率分布
        self.history_rates = self._precompute_history_rates()
        # 预计算历史货物所在层
        self._precompute_history_layers()

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
        # 预计算节点的中心高度
        self.node_centers = self._precompute_node_centers()

    # 设置路径规划对象
    def set_path_planning(self, path_planning):
        self.path_planning = path_planning



if __name__ == '__main__':
    # 读取数据
    model = Model()
    path_planning = Path_Planning(model)
    loc = LocationAssignment_Aisles()              # 初始化问题对象
    loc.set_model(model)                    # 设置模型对象
    loc.set_path_planning(path_planning)    # 设置路径规划对象
    correlation = loc._generate_relatedness_matrix(10)  # 生成模拟的货物相关性矩阵（0~1，1表示完全相关）
    test_items = loc.generateItems(10)      # 生成待入库货物信息
    # print(correlation)                    # 打印相关性矩阵
    # print(test_items)                     # 打印待入库货物信息
    # aisles = model.combined_graph.graph['aisles']
    print(loc.asiles)
    # '''========================初始化问题================'''
    # problem = loc.initProblem()  # 初始化问题
    #
    # '''========================种群设置================'''
    # Encoding = 'RI'  # 编码方式: 'BG'表示采用二进制/格雷编码，'RI'表示采用离散/整数编码;'P' 排列编码，即染色体每一位的元素都是互异的
    # NIND = 100 # 种群规模
    # Field = ea.crtfld(Encoding, loc.varTypes, loc.ranges, loc.borders) #译码矩阵 创建区域描述器
    # population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）
    # """===========================算法参数设置=========================="""
    # #实例化一个算法模板对象
    # algorithm = ea.moea_NSGA2_templet(loc, population)
    # algorithm.MAXGEN = 1000        # 最大进化代数
    # algorithm.mutOper.CR = 0.2     # 修改变异算子的变异概率
    # algorithm.recOper.XOVR = 0.9   # 修改交叉算子的交叉概率
    # algorithm.logTras = 1          # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    # algorithm.verbose = True       # 设置是否打印输出日志信息
    # algorithm.drawing = 1          # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    # """==========================调用算法模板进行种群进化==============="""
    # '''调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。
    # NDSet是一个种群类Population的对象。
    # NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。'''
    # [NDSet, population] = algorithm.run() #执行算法模板，得到非支配种群以及最后一代种群
    # NDSet.save() # 把非支配种群的信息保存到文件中
    #
    # """=================================输出结果======================="""
    # print('用时：%s 秒' % algorithm.passTime)
    # print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
    # if algorithm.log is not None and NDSet.sizes != 0:
    #     print('GD', algorithm.log['gd'][-1])
    #     print('IGD', algorithm.log['igd'][-1])
    #     print('HV', algorithm.log['hv'][-1])
    #     print('Spacing', algorithm.log['spacing'][-1])
    # """======================进化过程指标追踪分析=================="""
    # metricName = [['igd'], ['hv']]
    # Metrics = np.array([algorithm.log[metricName[i][0]] for i in
    #                     range(len(metricName))]).T
    # # 绘制指标追踪分析图
    # ea.trcplot(Metrics, labels=metricName, titles=metricName)
