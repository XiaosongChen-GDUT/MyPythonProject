import random
from random import randint
import matplotlib.pyplot as plt
from Program.DataModel import Model
import numpy as np
import pandas as pd
import networkx as nx
import geatpy as ea
import warnings

row_space = 1.54  #排间距
h1 = 3.15                                     #1楼换层到2楼的高度
h2 = 2.55                                     #2楼换层到3楼的高度
h3 = 1.5                                      #3楼高度
heights = [h1, h2, h3]                         #各楼层高度
Acc_car = 0.3;                                #四向车加速度
Dec_car = 0.3;                                #四向车减速度
Max_speed_car = 1.2;                          #四向车最大速度
Switching_time = 4;                           #换向时间
Acc_lift = 0.15;                              #提升机加速度
Dec_lift = 0.15;                              #提升机减速度
Max_speed_lift = 1.4;                         #提升机最大速度

enter_point = [51,348,636,925,1110,1620]    #入口点
out_point = [445,820,971,1156]  #出口点
fist_connect_point = [642,1116,674,1148]        #1楼提升机接驳点
second_connect_point = [2374,2844,2406,2876]    #2楼接驳点
third_connect_point = [3899,4135]               #3楼接驳点
class LocationAssignment(ea.Problem):
    def __init__(self, model):
        #图结构
        self.TopoGraph = model.combined_graph
        #图中节点的二维画布坐标
        self.location = nx.get_node_attributes(self.TopoGraph, 'location')
        #图中节点的三维坐标
        self.pos = nx.get_node_attributes(self.TopoGraph, 'pos')

        self.history_Loc = {}
        self.pending_Loc = {}

        # 预计算每层的累计高度
        self.cumulative_heights = [0] * len(heights)
        for i in range(len(heights)):
            self.cumulative_heights[i] = sum(heights[:i])
        # 预计算节点的中心高度
        self.node_centers = self._precompute_node_centers()


    # 初始化问题
    def initProblem(self):
        if self.pending_Loc == {}:
            warnings.warn("No pending items，cannot init problem!",RuntimeWarning)
            return
        if self.TopoGraph is None:
            warnings.warn("No TopoGraph，cannot init problem!",RuntimeWarning)
            return
        # 初始话 历史货物的质量矩,历史货物的总质量
        self.history_z_moment, self.history_total_mass = self._precompute_history()
        #图中节点的状态：-1-储货点，0-空闲，1-占用
        self.status = nx.get_node_attributes(self.TopoGraph,'status')
        #空闲货位集合
        self.free_status = [i for i in self.status if self.status[i] == 0]
        #空闲货位,按维度分类A\B\C\D
        self.free_Loc = {}
        for node in self.TopoGraph.nodes():
            if self.TopoGraph.nodes[node]['status'] == 0:
                dimension = self.TopoGraph.nodes[node]['dimension']
                if dimension not in self.free_Loc:
                    self.free_Loc[dimension] = []
                self.free_Loc[dimension].append(node)

        min_index = {'A':min(self.free_Loc['A']), 'B':min(self.free_Loc['B']), 'C':min(self.free_Loc['C']), 'D':min(self.free_Loc['D'])}
        max_index = {'A':max(self.free_Loc['A']), 'B':max(self.free_Loc['B']), 'C':max(self.free_Loc['C']), 'D':max(self.free_Loc['D'])}

        name = 'LocationAssignment'  # 初始化name
        M = 3                        # 初始化M,目标维数
        Dim = len(self.pending_Loc)  # 初始化Dim,决策变量维数=待入库货物数量
        varTypes = [1] * Dim         # 初始化varTypes,决策变量的类型，0-连续，1-离散
        maxormins = [1] * M                # 目标最小化
        pending_items = list(self.pending_Loc.values())
        lb = [min_index[item['dimension']] for item in pending_items]
        ub = [max_index[item['dimension']] for item in pending_items]
        lbin = [1] * Dim             # 决策变量包含下边界
        ubin = [1] * Dim             # 决策变量包含上边界

        print('min_index:', min_index)
        print('max_index:', max_index)
        print("lb:", lb)
        print("ub:", ub)
        print("lbin:", len(lbin),'ubin:', len(ubin), 'varTypes:', len(varTypes))
        # 调用父类构造函数
        ea.Problem.__init__(self, name, M, maxormins,Dim, varTypes, lb, ub, lbin, ubin)

    # 目标函数，pop为传入的种群对象
    def aimFunc(self, pop):
        Vars = pop.Phen  # 得到决策变量矩阵
        # 使用 ndim 和 shape 属性获取矩阵的形状
        num_rows, num_cols = Vars.shape


        F1 = self.cal_Center_Gravity(Vars)  # 计算质量重心
        F2 = 0  #效率
        F3 = 0  #相关性

        pop.ObjV = [F1, F2, F3]  # 目标函数值
        pop.CV = [0] * pop.sizes  # 约束条件值



    #计算仓库中所有货物的重心,返回目标函数值F1，列向量
    def cal_Center_Gravity(self,Vars):
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

        #todo:决策变量的解析
        pending_items = list(self.pending_Loc.values())  # 获取待入库货物列表
        pending_qualities = np.array([item['quality'] for item in pending_items])
        total_pending_mass = np.sum(pending_qualities)
        # 计算总质量
        total_mass = self.history_total_mass + total_pending_mass
        if total_mass == 0:
            warnings.warn("Total mass is 0，cal_Center_Gravity() return n行1列的0矩阵!")
            return np.zeros((Vars.shape[0], 1))  # 所有重心为0
        # 将Vars中的节点转换为层中心的矩阵
        layer_centers = np.vectorize(lambda node: self.node_centers[node])(Vars)    #使用了 numpy 库中的 vectorize 函数来简化对 Vars 矩阵中每个元素的操作
        # 计算每个个体的总质量矩（待入库部分）
        z_moments = np.dot( layer_centers,pending_qualities)
        # print('layer_centers:', layer_centers)
        # print('pending_qualities:', pending_qualities, 'z_moments:', z_moments)
        # 个体的质量矩 + 历史质量矩
        total_z_moments = z_moments + self.history_z_moment
        # 计算重心 保留3位小数 -1 表示根据数组的长度自动计算行数，而列数被显式地指定为 1。
        F1 = np.round(total_z_moments / total_mass,3).reshape((-1,1))   # 保留3位小数，并转为列向量
        return F1


        # # 遍历所有行
        # for i in range(num_rows):
        #     z_moment = 0         #初始话质量矩
        #     total_mass = 0       #总质量
        #     row = Vars[i]                                 #当前行的决策变量对应的节点
        #     # 遍历当前行的每个元素
        #     for j in range(num_cols):                     # j 为决策变量的下标：0，1，2，3，4，5，6，7，8，9
        #         node = row[j]                             #当前决策变量对应的空闲货位节点
        #         quality = pending_items[j]['quality']     # j 对应的待入库货物列表 pending_items 的索引
        #         current_layer_center = self.node_centers[node]    # 前z-1层总高度+当前层高度/2
        #         z_moment += quality * current_layer_center
        #         total_mass += quality
        #         print('node:', node, 'quality:', quality,'current_layer_center:', current_layer_center, 'z_moment:', z_moment, 'total_mass:', total_mass)
        #     # 计算该行的质量重心
        #     if total_mass == 0:
        #         F1[i] = 0  # 避免除以零
        #     else:
        #         center_z = round((z_moment + self.history_z_moment ) / (total_mass + self.history_total_mass), 3)  # 计算垂直方向重心,保留3位小数
        #         F1[i] = center_z
        # return F1

    def

    """预计算所有节点的层中心高度"""
    def _precompute_node_centers(self):
        node_centers = {}
        for node in self.pos:
            (x, y, z) = self.pos[node]
            cumulative_height = self.cumulative_heights[z-1]          # 前z-1层总高度
            current_layer_center = cumulative_height + heights[z-1]/2    # 前z-1层总高度+当前层高度/2
            node_centers[node] = current_layer_center   # 节点的层中心。在后续处理中，只需要通过节点编号直接获取层中心值，而不需要每次重新计算。
        return node_centers

    """预计算历史货物的质量矩和总质量"""
    def _precompute_history(self):
        z_moment = 0
        total_mass = 0
        #计算历史货物的质量
        for item in self.history_Loc.values():
            quality = item['quality']
            node = item['node']
            current_layer_center = self.node_centers[node]    # 前z-1层总高度+当前层高度/2
            z_moment += quality * current_layer_center
            total_mass += quality
        return z_moment, total_mass




    '''========================以下为辅助函数=========================='''
    #生成入库数据
    def generateItems(self,Num):
        pending_Loc = {}
        item_id = random.randint(1, 1000)
        # 随机生成Num个待入库货物
        for i in range(Num):
            item_id += 1
            rate = round(np.random.uniform(0.01, 1), 2)
            dimension = np.random.choice(['A','B','C','D'])
            if dimension == 'A':
                quality = np.random.randint(1, 51)      #A，质量值将在 1 到 50 之间随机生成
            elif dimension == 'D':
                quality = np.random.randint(150, 201)   #D，质量值将在 150 到 200 之间随机生成
            else:
                quality = np.random.randint(51, 150)    # 对于 B 和 C，质量值在 51 到 150 之间
            pending_Loc[item_id] = {
                'rate': rate,
                'quality': quality,
                'dimension': dimension
            }
        self.pending_Loc = pending_Loc
        return self.pending_Loc


    # 添加新的历史记录：质量、周转率、类型和存储货位节点
    def add_history(self, item_id, quality, turnover_rate, dimension, location_node):
        self.history_Loc[item_id] = {
            'quality': quality,
            'rate': turnover_rate,
            'dimension':dimension,
            'node': location_node
        }

    # 根据货物id删除特定货物的历史信息
    def remove_item_history(self, item_id):
        if item_id in self.history_Loc:
            del self.history_Loc[item_id]
        else:
            print("No such item in history，cannot delete!")

    #设置历史已存储信息
    def set_history_Loc(self, history_Loc):
        self.history_Loc = history_Loc

    #设置待入库货物信息
    def set_pending_Loc(self, pending_Loc):
        self.pending_Loc = pending_Loc


if __name__ == '__main__':
    # 读取数据
    model = Model()
    loc = LocationAssignment(model)  # 初始化问题对象
    loc.set_history_Loc({})                     # 设置历史已存储信息
    loc.set_pending_Loc(loc.generateItems(3))   # 生成待入库货物信息
    print(loc.pending_Loc)
    loc.initProblem()                           # 初始化问题

    # 定义outFunc()函数
    def outFunc(alg, pop):  # alg 和 pop为outFunc的固定输入参数，分别为算法对象和每次迭代的种群对象。
        print('进化代数：%d' % alg.currentGen)
        print('决策变量维数：',pop.Phen.shape)
        print('最优个体：')
        print(pop.Phen)
        print('最优个体的目标函数值：')
        print(pop.ObjV)
        print('最优个体的基因型：')
        print(pop.Gen)
        print('最优个体的适应度值：')
        print(pop.Aptitude)

    #构建算法
    algorithm = ea.moea_NSGA2_templet(
        loc,
        ea.Population(Encoding='RI', NIND=100),# 种群规模100
        MAXGEN=300,  # 最大进化代数
        logTras=1,  # 每隔多少代记录日志，0表示不记录。
        outFunc=outFunc,  # 自定义输出函数
        )
    print(algorithm.problem)
    Vars = np.array([[1,2,3]])
    gravity = loc.cal_Center_Gravity(Vars)
    print(gravity)
    # 求解
    # res = ea.optimize(algorithm,
    #                   verbose=False,
    #                   drawing=1,
    #                   outputMsg=True,
    #                   drawLog=True,
    #                   saveFlag=False)
    # print(res)



