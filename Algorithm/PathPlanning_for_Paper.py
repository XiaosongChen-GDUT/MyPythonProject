# This file contains the implementation of the A* algorithm for path planning
import heapq
import os

from networkx.algorithms.shortest_paths.weighted import _weight_function, _dijkstra
from Program.DataModel import Model
from heapq import heappop, heappush
from itertools import count  # count object for the heap堆的计数对象
from itertools import chain
import networkx as nx
import random
import pandas as pd
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体为宋体，数字字体为Times Roman
plt.rcParams['font.sans-serif'] = ['SimSun']  # 中文使用宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# plt.rcParams['font.serif'] = ['Times New Roman']  # 数字使用Times Roman

# 定义地标点
lanmarks_1 = [381, 369, 357,857,845,833,1324,1300,1629,1661]  # 关键十字路口地标点
lanmarks_2 = [642,674,1116,1148]    #1楼提升机接驳点
lanmarks_3 = [357,381,1300,1324]    #较分散的十字路口地标点
lanmarks_4 = [1,50,1708,1748]  # 地图的四个角落地标点

first_landmarks = [1,50,1708,1748]               # 1楼地标点
# second_landmarks = [1749,1795,3528,3578]         # 2楼地标点
second_landmarks = [1749,3578]         # 2楼地标点
third_landmarks = [3579,3602,4443,4466]          # 3楼地标点
fist_connect_point = [642,1116,674,1148]        #1楼提升机接驳点
second_connect_point = [2374,2844,2406,2876]    #2楼接驳点
third_connect_point = [3899,4135]               #3楼接驳点
enter_point = [51,348,636,925,1110,1620]    #入口点
out_point = [445,820,971,1156]  #出口点
'''做路径规划对比试验，出图'''
class Path_Planning:
    def __init__(self,model):
        #图结构
        self.TopoGraph = model.combined_graph
        self.first_floor = model.floor1
        self.second_floor = model.floor2
        self.third_floor = model.floor3
        # 地标点
        # 组合所有地标点的索引
        self.first_landmarks_index =  first_landmarks  # 地标点的索引
        self.second_landmarks_index = second_landmarks
        self.third_landmarks_index = third_landmarks
        # 地标点的坐标
        self.first_landmarks = {}  # 初始化字典
        self.second_landmarks = {}
        self.third_landmarks = {}
        # # 计算每个地标点到每个节点的距离
        for index in self.first_landmarks_index:
            self.first_landmarks[index] = {}
                # 计算当前节点到地标的距离
            distance = nx.shortest_path_length(self.first_floor,source=index,weight='weight')   #返回的是字典
            self.first_landmarks[index] = distance
        #添加地标点数据到图中
        self.first_floor.graph['landmarks'] = self.first_landmarks
        for index in self.second_landmarks_index:
            self.second_landmarks[index] = {}
                # 计算当前节点到地标的距离
            distance = nx.shortest_path_length(self.second_floor,source=index,weight='weight')   #返回的是字典
            self.second_landmarks[index] = distance
        #添加地标点数据到图中
        self.second_floor.graph['landmarks'] = self.second_landmarks
        for index in self.third_landmarks_index:
            self.third_landmarks[index] = {}
                # 计算当前节点到地标的距离
            distance = nx.shortest_path_length(self.third_floor,source=index,weight='weight')   #返回的是字典
            self.third_landmarks[index] = distance
        #添加地标点数据到图中
        self.third_floor.graph['landmarks'] = self.third_landmarks

        #初始化换层节点
        self.cross_nodes = {1:fist_connect_point,2:second_connect_point, 3:third_connect_point}     #dict_values([[642, 1116, 674, 1148], [2374, 2844, 2406, 2876], [3899, 4135]])
        self.cross_nodes_list = list(chain(*self.cross_nodes.values()))     # [642, 1116, 674, 1148, 2374, 2844, 2406, 2876, 3899, 4135]
        #初始化换层节点的地标点
        cross_landmarks = {}
        for index in second_connect_point:
            cross_landmarks[index] = {}
            distance = nx.shortest_path_length(self.TopoGraph,source=index,weight='weight')   #返回的是字典
            cross_landmarks[index] = distance
        self.TopoGraph.graph['landmarks'] = cross_landmarks



        '''分析路径:返回搜索时间，路径，探索过的节点，路径长度，转向次数，路径时间成本'''
    '''分析路径:返回搜索时间，路径，探索过的节点，路径长度，转向次数，路径时间成本'''
    def Analyze_Path(self,graph, source, target, algorithm_name ):
        # turn_count = 0  # 初始化转向计数
        start_time = time.time()
        path, cost, explored = self.run(graph, source, target, algorithm_name)
        take_time = round( (time.time() - start_time)*1000,2)  # 计算运行时间
        turn_count = self.Turn_Count(graph, path)  # 计算转向次数
        cal_path_time = self.cal_path_time(graph, path)  # 计算路径时间成本
        return take_time, path, explored, cost, turn_count,cal_path_time

    ''' path, explored, cost'''
    def run(self, graph, source, target, algorithm_name):
        try:
            weight='weight'
            if algorithm_name == "Dijkstra":# Dijkstra算法
                return self.Dijkstra(graph, source, target)
            elif algorithm_name == "A*":# :# A*算法
                return self.A_star(graph, source, target)
            elif algorithm_name == "ATL_star":# ATL_star
                return self.ATL_star(graph, source, target,  lanmarks_4)
            elif algorithm_name == "改进A*":#改进的A*算法
                return self.improve_A_star(graph, source, target)
            elif algorithm_name == "本文方法":
                return self.abs_ATL_star(graph, source, target, lanmarks_4)
            elif algorithm_name == "双向A*":# 双向A*
                return self.bidirectional_Astar(graph, source, target)
            elif algorithm_name == "D*":
                return self.DStar(graph, source, target)
            elif algorithm_name == "D*Lite":
                return self.DStarLite_optimized(graph, source, target)
            else:
                raise ValueError("class-> Path_Planning -> run : 没匹配到算法名称：",algorithm_name)
                return None, None, None
        except ValueError as ve:
            print(f"ValueError: {ve}")

        '''遍历节点，对比算法的实验'''
    def test_All_Nodes(self,graph):
        # nodes = list(graph.nodes)
        # source = random.choice(nodes)  # 随机选择源节点
        # source = 51 # 随机选择源节点
        # nodes.remove(source)  # 去掉源节点
        # 随机选择200个目标节点
        # targets = random.sample(nodes, 10)
        sources = [51,348,925,1110,1620]
        # sources = [1758,2040,1749,2886,3537,3132]
        targets = [26,323,50,1399,1727,1748]
        # targets = [1773,1789,3552,3162,2071]
        algorithm_results = {}  # 存储每个算法的结果
        # 初始化累计变量
        # 定义算法名称和对应的函数
        algorithms = {
            # "Dijkstra": self.Dijkstra,
            "D*Lite": self.DStarLite,
            "改进A*": self.improve_A_star,
            # "D*": self.DStar,
            "双向A*": self.bidirectional_Astar,
            "本文方法": self.abs_ATL_star
        }

        for algorithm in algorithms:
            algorithm_results[algorithm] = {
                'take_time': [],      # 累计耗时
                'explored': [],       # 累计探索节点数
                'cost': [],           # 累计成本
                'turn_count': [],      # 累计转向次数
                'cal_path_time': []   # 累计路径时间
            }
        status = nx.get_node_attributes(graph, 'status')  # 获取节点状态
        for source in sources:
            for target in targets:
                # 遍历每个算法
                for algo_name, algo_func in algorithms.items():
                    # time_start = time.time()
                    # 调用算法函数
                    take_time, path, explored, cost, turn_count,cal_path_time = self.Analyze_Path(graph, source, target, algo_name)

                    # print(f"算法：{algo_name}，源点：{source}，目标点：{target}，路径：{path}，搜索耗时：{take_time}ms，探索节点数：{len(explored)}，成本：{cost}，转弯次数：{turn_count},计算路径耗时：{cal_path_time}ms")
                    # 存储结果
                    algorithm_results[algo_name]['take_time'].append(take_time)
                    algorithm_results[algo_name]['explored'].append(len(explored))
                    algorithm_results[algo_name]['cost'].append(cost)
                    algorithm_results[algo_name]['turn_count'].append(turn_count)
                    algorithm_results[algo_name]['cal_path_time'].append(cal_path_time)

        # # 准备写入Excel的数据
        # data_list = []
        # for algo_name, results in algorithm_results.items():
        #     for i in range(len(results['take_time'])):
        #         data_list.append({
        #             '序号': i + 1,
        #             '算法': algo_name,
        #             '起始点': sources[i // len(targets)],
        #             '终点': targets[i % len(targets)],
        #             '探索节点数': results['explored'][i],
        #             '搜索耗时': results['take_time'][i],
        #             '路径耗时': results['cal_path_time'][i]
        #         })

        # 创建DataFrame
        # df = pd.DataFrame(data_list)

        # 写入Excel文件
        # df.to_excel('algorithm_results.xlsx', index=False)
        #优化率计算
        # self.cal_optimity(algorithm_results)

        return algorithm_results

    """对比测试不同的地标点的数量、位置"""
    def test_ATL_star(self, graph, source, target, heuristic_index=2, weight='weight'):
        canvas =  graph.graph['canvas']
        # targets = [26,50,323,1399,1727,1748]  # 测试目标点
        nums = 200  # 测试目标点数量
        targets = random.sample(list(graph.nodes), nums)  # 随机选择不重复的目标点
        source = random.choice(list(graph.nodes))  # 随机选择源节点
        algorithm_results = {}  # 存储每个算法的结果

        path_cost = 0  # 路径长度
        take_time = 0  # 耗时
        turn_count = 0  # 转弯次数
        explored_count = 0  # 探索节点数
        for target in targets:
            start_time = time.time()
            path, cost, explored=self.ATL_star(graph, source, target, heuristic_index, lanmarks_1,weight)
            take_time += round((time.time() - start_time)*1000,2)
            path_cost += cost
            turn_count += self.Turn_Count(graph,path)
            explored_count += len(explored)

            # 存储结果
        algorithm_results["lanmarks_1"] = {
            'take_time': take_time,
            'explored': explored_count,
            'cost': path_cost,
            'turn_count': turn_count

            # 'take_time': take_time/nums,
            # 'explored': explored_count/nums,
            # 'cost': path_cost/nums,
            # 'turn_count': turn_count/nums
        }
        # canvas.show_visited_process(graph, explored)    #显示探索过的节点
        # canvas.show_path_with_color(graph, path,color='pink',name='lanmarks_1')  # 显示路径
        # canvas.save_image(source, target, "ATL_star", "lanmarks_1")
        # canvas.reset_canvas()  # 重置画布

        path_cost = 0  # 路径长度
        take_time = 0  # 耗时
        turn_count = 0  # 转弯次数
        explored_count = 0  # 探索节点数
        for target in targets:
            start_time = time.time()
            path, cost, explored=self.ATL_star(graph, source, target, heuristic_index, lanmarks_2,weight)
            take_time += round((time.time() - start_time)*1000,2)
            path_cost += cost
            turn_count += self.Turn_Count(graph,path)
            explored_count += len(explored)
        # 存储结果
        algorithm_results["lanmarks_2"] = {
            'take_time': take_time,
            'explored': explored_count,
            'cost': path_cost,
            'turn_count': turn_count
            # 'take_time': take_time/nums,
            # 'explored': explored_count/nums,
            # 'cost': path_cost/nums,
            # 'turn_count': turn_count/nums
        }
        # canvas.show_visited_process(graph, explored)    #显示探索过的节点
        # canvas.show_path_with_color(graph, path,color='brown',name='lanmarks_2')  # 显示路径
        # canvas.save_image(source, target, "ATL_star", "lanmarks_2")
        # canvas.reset_canvas()  # 重置画布

        path_cost = 0  # 路径长度
        take_time = 0  # 耗时
        turn_count = 0  # 转弯次数
        explored_count = 0  # 探索节点数
        for target in targets:
            start_time = time.time()
            path, cost ,explored=self.ATL_star(graph, source, target, heuristic_index, lanmarks_3,weight)
            take_time += round((time.time() - start_time)*1000,2)
            path_cost += cost
            turn_count += self.Turn_Count(graph,path)
            explored_count += len(explored)
        # 存储结果
        algorithm_results["lanmarks_3"] = {
            'take_time': take_time,
            'explored': explored_count,
            'cost': path_cost,
            'turn_count': turn_count
            # 'take_time': take_time/nums,
            # 'explored': explored_count/nums,
            # 'cost': path_cost/nums,
            # 'turn_count': turn_count/nums
        }
        # canvas.show_visited_process(graph, explored)    #显示探索过的节点
        # canvas.show_path_with_color(graph, path,color='purple',name='lanmarks_3')  # 显示路径
        # canvas.save_image(source, target, "ATL_star", "lanmarks_3")
        # canvas.reset_canvas()  # 重置画布

        path_cost = 0  # 路径长度
        take_time = 0  # 耗时
        turn_count = 0  # 转弯次数
        explored_count = 0  # 探索节点数
        for target in targets:
            start_time = time.time()
            path, cost ,explored=self.ATL_star(graph, source, target, heuristic_index, lanmarks_4,weight)
            take_time += round((time.time() - start_time)*1000,2)
            path_cost += cost
            turn_count += self.Turn_Count(graph,path)
            explored_count += len(explored)
        # 存储结果
        algorithm_results["lanmarks_4"] = {
            'take_time': take_time,
            'explored': explored_count,
            'cost': path_cost,
            'turn_count': turn_count
            # 'take_time': take_time/nums,
            # 'explored': explored_count/nums,
            # 'cost': path_cost/nums,
            # 'turn_count': turn_count/nums
        }
        # canvas.show_visited_process(graph, explored)    #显示探索过的节点
        # canvas.show_path_with_color(graph, path,color='black',name='lanmarks_4')  # 显示路径
        # canvas.save_image(source, target, "ATL_star", "lanmarks_4")
        # canvas.reset_canvas()  # 重置画布

        path_cost = 0  # 路径长度
        take_time = 0  # 耗时
        turn_count = 0  # 转弯次数
        explored_count = 0  # 探索节点数
        for target in targets:
            start_time = time.time()
            path, cost, explored=self.A_star(graph, source, target, heuristic_index,weight)
            take_time += round((time.time() - start_time)*1000,2)
            path_cost += cost
            turn_count += self.Turn_Count(graph,path)
            explored_count += len(explored)
        # 存储结果
        algorithm_results["A*"] = {
            'take_time': take_time,
            'explored': explored_count,
            'cost': path_cost,
            'turn_count': turn_count
            # 'take_time': take_time/nums,
            # 'explored': explored_count/nums,
            # 'cost': path_cost/nums,
            # 'turn_count': turn_count/nums
        }
        # canvas.show_visited_process(graph, explored)    #显示探索过的节点
        # canvas.show_path_with_color(graph, path,"green",name='A*')  # 显示路径
        # canvas.save_image(source, target, "A*", "曼哈顿距离")
        # canvas.reset_canvas()  # 重置画布

        # path_cost = 0  # 路径长度
        # take_time = 0  # 耗时
        # turn_count = 0  # 转弯次数
        # explored_count = 0  # 探索节点数
        # for target in targets:
        #     start_time = time.time()
        #     path, cost, explored=self.improve_A_star(graph, source, target, heuristic_index,weight)
        #     take_time += round((time.time() - start_time)*1000,2)
        #     path_cost += cost
        #     turn_count += self.Turn_Count(graph,path)
        #     explored_count += len(explored)
        # # 存储结果
        # algorithm_results["改进A*"] = {
        #     'take_time': take_time,
        #     'explored': explored_count,
        #     'cost': path_cost,
        #     'turn_count': turn_count
        #     # 'take_time': take_time/nums,
        #     # 'explored': explored_count/nums,
        #     # 'cost': path_cost/nums,
        #     # 'turn_count': turn_count/nums
        # }
        # canvas.show_visited_process(graph, explored)    #显示探索过的节点
        # canvas.show_path_with_color(graph, path,color='orange',name='改进A*')  # 显示路径
        # canvas.save_image(source, target, "改进A*", "曼哈顿距离")
        # canvas.reset_canvas()  # 重置画布

        # path_cost = 0  # 路径长度
        # take_time = 0  # 耗时
        # turn_count = 0  # 转弯次数
        # explored_count = 0  # 探索节点数
        # for target in targets:
        #     start_time = time.time()
        #     path, cost, explored=self.Dijkstra(graph, source, target,weight)
        #     take_time += round((time.time() - start_time)*1000,2)
        #     path_cost += cost
        #     turn_count += self.Turn_Count(graph,path)
        #     explored_count += len(explored)
        # # 存储结果
        # algorithm_results["Dijkstra"] = {
        #     'take_time': take_time,
        #     'explored': explored_count,
        #     'cost': path_cost,
        #     'turn_count': turn_count
        #     # 'take_time': take_time/nums,
        #     # 'explored': explored_count/nums,
        #     # 'cost': path_cost/nums,
        #     # 'turn_count': turn_count/nums
        # }

        # canvas.show_visited_process(graph, explored)    #显示探索过的节点
        # canvas.show_path_with_color(graph, path,color='blue',name='Dijkstra算法')  # 显示路径
        # canvas.save_image(source, target, "dijkstra算法", "")
        # canvas.reset_canvas()  # 重置画布
        return algorithm_results


    """=============算法部分=================="""
    class DStarNode:
        """D*算法节点状态容器"""
        __slots__ = ['g', 'rhs', 'parent', 'state']

        def __init__(self):
            self.g = float('inf')    # 实际代价估计
            self.rhs = float('inf')  # 右侧启发式值
            self.parent = None       # 父节点指针
            self.state = 'NEW'       # 状态: NEW/OPEN/CLOSED

    def DStarLite_optimized2(self, graph, start, target):
        """
        D* Lite算法优化版
        :param graph: NetworkX加权无向图
        :param start: 起始节点ID
        :param target: 目标节点ID
        :return: (路径列表, 路径成本, 探索节点字典)
        """
        # --------------- 初始化数据结构 ---------------
        location = nx.get_node_attributes(graph, 'location')  # 节点坐标字典
        weight_func = _weight_function(graph, 'weight')       # 权重计算函数
        G_succ = graph._adj                                   # 邻接表

        # 初始化节点状态字典
        g_values = {node: float('inf') for node in graph.nodes}  # 实际成本估计
        rhs_values = {node: float('inf') for node in graph.nodes} # 右侧启发值
        parent_map = {}                                         # 父节点回溯字典

        # 优先队列（按优先级排序）
        open_queue = []
        # --------------- 工具函数定义 ---------------
        def heuristic(u, v):
            """改进的欧几里得距离启发式（兼容二维位置）"""
            dx = location[u][0] - location[v][0]
            dy = location[u][1] - location[v][1]
            return math.hypot(dx, dy) * 0.98  # 保证启发式可采纳性

        def calculate_key(node):
            """优先队列键值计算（决定节点处理顺序）"""
            return (
                min(g_values[node], rhs_values[node]) + heuristic(node, start),
                min(g_values[node], rhs_values[node])
            )

        def update_node(node):
            """更新节点状态并维护优先队列"""
            key = calculate_key(node)
            if g_values[node] != rhs_values[node]:
                heapq.heappush(open_queue, (key, node))
            else:
                # 移除过时节点
                if (key, node) in open_queue:
                    open_queue.remove((key, node))
                    heapq.heapify(open_queue)

        # --------------- 算法初始化 ---------------
        rhs_values[target] = 0
        heapq.heappush(open_queue, (calculate_key(target), target))
        parent_map[target] = None

        # --------------- 主计算循环 ---------------
        while open_queue:
            current_key, current = heapq.heappop(open_queue)

            # 终止条件：到达起点且路径一致
            if current == start and g_values[start] == rhs_values[start]:
                break

            # 跳过过时节点（已更新更优路径）
            if current_key > calculate_key(current):
                continue

            # 节点状态转换
            if g_values[current] > rhs_values[current]:
                # 更新实际成本
                g_values[current] = rhs_values[current]
                # 处理前驱节点（无向图）
                for neighbor in G_succ[current]:
                    new_rhs = g_values[current] + weight_func(neighbor, current, G_succ[current][neighbor])
                    if new_rhs < rhs_values[neighbor]:
                        rhs_values[neighbor] = new_rhs
                        parent_map[neighbor] = current
                        update_node(neighbor)
            else:
                # 回溯更新
                g_values[current] = float('inf')
                update_node(current)
                for neighbor in G_succ[current]:
                    if parent_map.get(neighbor) == current:
                        rhs_values[neighbor] = min(
                            [g_values[pred] + weight_func(neighbor, pred, G_succ[neighbor][pred])
                             for pred in G_succ[neighbor]]
                        )
                        update_node(neighbor)

        # --------------- 路径重构 ---------------
        path = []
        current_node = start
        explored = {}

        # 从起点向目标回溯
        while current_node != target:
            path.append(current_node)
            explored[current_node] = parent_map.get(current_node)

            # 安全检查
            if current_node not in parent_map:
                raise nx.NetworkXNoPath(f"路径中断于节点 {current_node}")

            current_node = parent_map[current_node]

        path.append(target)
        explored[target] = None

        return path, rhs_values[start], explored

    def DStarLite_optimized(self, graph, start, target):
        """
        优化版D* Lite算法
        改进点：
        1. 引入地标点加速启发式计算
        2. 优化优先队列键值计算
        3. 动态调整启发式权重
        4. 改进换层节点处理逻辑
        """
        # --------------- 初始化 ---------------
        location = nx.get_node_attributes(graph, 'location')
        pos = nx.get_node_attributes(graph, 'pos')
        weight_func = _weight_function(graph, 'weight')
        G_succ = graph._adj

        # 初始化地标数据（使用预计算的地标距离）
        landmarks = graph.graph.get('landmarks', {})

        # 节点状态容器
        nodes = {
            n: {
                'g': float('inf'),
                'rhs': float('inf'),
                'parent': None,
                'state': 'NEW'
            } for n in graph.nodes
        }

        # --------------- 核心改进 1：混合启发式函数 ---------------
        def heuristic(u, v):
            """混合启发式：结合地标和曼哈顿距离"""
            base_h = abs(location[v][0]-location[u][0]) + abs(location[v][1]-location[u][1])

            # 地标加速（如果可用）
            if landmarks:
                lower_bounds = [abs(landmarks[L][v] - landmarks[L][u]) for L in landmarks]
                return max(base_h, *lower_bounds)
            return base_h

        # --------------- 核心改进 2：动态权重 ---------------
        beta = 1.2  # 初始权重系数
        dynamic_weight = lambda h: beta * h  # 动态调整函数

        # --------------- 核心改进 3：优化优先队列键 ---------------
        def calculate_key(node):
            """优化键值计算：优先考虑rhs值"""
            k1 = min(nodes[node]['g'], nodes[node]['rhs']) + dynamic_weight(heuristic(node, start))
            k2 = min(nodes[node]['g'], nodes[node]['rhs'])
            return (k1, k2)

        # # --------------- 核心改进 4：换层优化 ---------------
        # def is_valid_transition(current, neighbor):
        #     """优化换层节点验证逻辑"""
        #     current_layer = pos[current][2]
        #     neighbor_layer = pos[neighbor][2]
        #
        #     # 同层移动
        #     if current_layer == neighbor_layer:
        #         return True
        #
        #     # 跨层移动必须通过换层节点
        #     return current in self.cross_nodes[current_layer] and \
        #         neighbor in self.cross_nodes[neighbor_layer]

        # --------------- 算法初始化 ---------------
        open_queue = []
        nodes[target]['rhs'] = 0
        heapq.heappush(open_queue, (calculate_key(target), target))

        # --------------- 主循环优化 ---------------
        while open_queue:
            current_key, current = heapq.heappop(open_queue)

            # 提前终止条件
            if current == start and nodes[start]['g'] == nodes[start]['rhs']:
                break

            if current_key > calculate_key(current):
                continue

            # 状态更新优化
            if nodes[current]['g'] > nodes[current]['rhs']:
                nodes[current]['g'] = nodes[current]['rhs']
                for neighbor in G_succ[current]:
                    # # 有效性检查
                    # if not is_valid_transition(current, neighbor):
                    #     continue

                    new_rhs = nodes[current]['g'] + weight_func(current, neighbor, G_succ[current][neighbor])
                    if new_rhs < nodes[neighbor]['rhs']:
                        nodes[neighbor]['rhs'] = new_rhs
                        nodes[neighbor]['parent'] = current
                        heapq.heappush(open_queue, (calculate_key(neighbor), neighbor))
            else:
                nodes[current]['g'] = float('inf')
                for neighbor in G_succ[current]:
                    if nodes[neighbor]['parent'] == current:
                        nodes[neighbor]['rhs'] = min(
                            [nodes[pred]['g'] + weight_func(neighbor, pred, G_succ[neighbor][pred])
                             for pred in G_succ[neighbor]]
                        )
                    heapq.heappush(open_queue, (calculate_key(neighbor), neighbor))

        # --------------- 路径重构优化 ---------------
        path = []
        current_node = start
        explored = {}

        while current_node != target:
            path.append(current_node)
            explored[current_node] = nodes[current_node]['parent']

            # 安全处理
            if current_node not in nodes or nodes[current_node]['parent'] is None:
                raise nx.NetworkXNoPath(f"路径中断于节点 {current_node}")

            current_node = nodes[current_node]['parent']

        path.append(target)
        return path, nodes[start]['rhs'], explored

    # ... 其他代码保持不变 ...

    def DStarLite(self, graph, start, target):
        """
        D* Lite算法实现（静态环境优化版）
        适用于加权无向图，利用二维位置信息优化启发式函数
        :param graph: NetworkX加权无向图
        :param start: 起始节点ID
        :param target: 目标节点ID
        :return: (路径列表, 路径成本, 探索节点字典)
        """
        # --------------- 初始化数据结构 ---------------
        location = nx.get_node_attributes(graph, 'location')  # 节点坐标字典
        weight_func = _weight_function(graph, 'weight')       # 权重计算函数
        G_succ = graph._adj                                   # 邻接表

        # 初始化节点状态字典
        g_values = {node: float('inf') for node in graph.nodes}  # 实际成本估计
        rhs_values = {node: float('inf') for node in graph.nodes} # 右侧启发值
        parent_map = {}                                         # 父节点回溯字典

        # 优先队列（按优先级排序）
        open_queue = []

        # --------------- 工具函数定义 ---------------
        def heuristic(u, v):
            """改进的欧几里得距离启发式（兼容二维位置）"""
            dx = location[u][0] - location[v][0]
            dy = location[u][1] - location[v][1]
            return math.hypot(dx, dy) * 0.98  # 保证启发式可采纳性

        def calculate_key(node):
            """优先队列键值计算（决定节点处理顺序）"""
            return (
                min(g_values[node], rhs_values[node]) + heuristic(node, start),
                min(g_values[node], rhs_values[node])
            )

        def update_node(node):
            """更新节点状态并维护优先队列"""
            if g_values[node] != rhs_values[node]:
                # 插入或更新队列
                heapq.heappush(open_queue, (calculate_key(node), node))
            else:
                # 从队列移除（如果存在）
                if (calculate_key(node), node) in open_queue:
                    open_queue.remove((calculate_key(node), node))
                    heapq.heapify(open_queue)

        # --------------- 算法初始化 ---------------
        # 设置目标节点状态
        rhs_values[target] = 0
        heapq.heappush(open_queue, (calculate_key(target), target))
        parent_map[target] = None

        # --------------- 主计算循环 ---------------
        while open_queue:
            current_key, current = heapq.heappop(open_queue)

            # 终止条件：到达起点且路径一致
            if current == start and g_values[start] == rhs_values[start]:
                break

            # 跳过过时节点（已更新更优路径）
            if current_key > calculate_key(current):
                continue

            # 节点状态转换
            if g_values[current] > rhs_values[current]:
                # 更新实际成本
                g_values[current] = rhs_values[current]
                # 处理前驱节点（无向图）
                for neighbor in G_succ[current]:
                    new_rhs = g_values[current] + weight_func(neighbor, current, G_succ[current][neighbor])
                    if new_rhs < rhs_values[neighbor]:
                        rhs_values[neighbor] = new_rhs
                        parent_map[neighbor] = current
                        update_node(neighbor)
            else:
                # 回溯更新
                g_values[current] = float('inf')
                update_node(current)
                for neighbor in G_succ[current]:
                    if parent_map.get(neighbor) == current:
                        rhs_values[neighbor] = min(
                            [g_values[pred] + weight_func(neighbor, pred, G_succ[neighbor][pred])
                             for pred in G_succ[neighbor]]
                        )
                        update_node(neighbor)

        # --------------- 路径重构 ---------------
        path = []
        current_node = start
        explored = {}

        # 从起点向目标回溯
        while current_node != target:
            path.append(current_node)
            explored[current_node] = parent_map.get(current_node)

            # 安全检查
            if current_node not in parent_map:
                raise nx.NetworkXNoPath(f"路径中断于节点 {current_node}")

            current_node = parent_map[current_node]

        path.append(target)
        explored[target] = None

        return path, rhs_values[start], explored

    def DStar(self, graph, start, goal):
        """
        D* Lite 算法实现 (适用于动态环境路径规划)
        :param graph: NetworkX加权无向图
        :param start: 起始节点
        :param goal:  目标节点
        :return: (路径列表, 路径成本, 探索节点字典)
        """
        # --------------- 初始化数据结构 ---------------
        location = nx.get_node_attributes(graph, 'location')
        pos = nx.get_node_attributes(graph, 'pos')
        weight_func = _weight_function(graph, 'weight')
        G_succ = graph._adj

        # 节点状态字典
        nodes = {n: self.DStarNode() for n in graph.nodes}

        # 优先队列 (按k值排序)
        open_queue = []
        # 自定义键值计算
        def queue_key(node):
            k1 = min(nodes[node].g, nodes[node].rhs) + heuristic(node, goal)
            k2 = min(nodes[node].g, nodes[node].rhs)
            return (k1, k2)  # 按主次排序条件
        def heuristic(u, v):
            """欧几里得距离启发式"""
            dx = location[u][0] - location[v][0]
            dy = location[u][1] - location[v][1]
            return math.hypot(dx, dy)
        # 优先队列排序函数
        def queue_key(node):
            return (min(nodes[node].g, nodes[node].rhs) +
                    heuristic(node, goal),
                    min(nodes[node].g, nodes[node].rhs))

        # --------------- 工具函数 ---------------
        def update_node(u):
            """更新节点状态"""
            if nodes[u].g != nodes[u].rhs:
                if u in open_queue:
                    heapq.heappush(open_queue, (queue_key(u), u))
                else:
                    heapq.heappush(open_queue, (queue_key(u), u))
                    nodes[u].state = 'OPEN'
            else:
                if u in open_queue:
                    open_queue.remove((queue_key(u), u))
                nodes[u].state = 'CLOSED'

        def compute_shortest_path():
            """主计算循环"""
            while open_queue and \
                    (queue_key(goal)[0] < queue_key(open_queue[0][1])[0] or
                     nodes[goal].rhs != nodes[goal].g):

                _, u = heapq.heappop(open_queue)
                nodes[u].state = 'CLOSED'

                if nodes[u].g > nodes[u].rhs:
                    nodes[u].g = nodes[u].rhs
                    for pred in G_succ[u]:  # 处理前驱节点（无向图）
                        if pred == u: continue
                        new_rhs = nodes[u].g + weight_func(pred, u, G_succ[pred][u])
                        if new_rhs < nodes[pred].rhs:
                            nodes[pred].rhs = new_rhs
                            nodes[pred].parent = u
                            update_node(pred)
                else:
                    nodes[u].g = float('inf')
                    for pred in G_succ[u]:
                        if pred == u: continue
                        if nodes[pred].parent == u:
                            nodes[pred].rhs = float('inf')
                            update_node(pred)
                    update_node(u)

        # --------------- 初始化算法 ---------------
        # 设置目标节点
        nodes[goal].rhs = 0
        heapq.heappush(open_queue, (queue_key(goal), goal))
        nodes[goal].state = 'OPEN'

        # 首次计算最短路径
        compute_shortest_path()

        # 检查路径是否存在
        if nodes[start].rhs == float('inf'):
            raise nx.NetworkXNoPath(f"No path from {start} to {goal}")

        # --------------- 路径重构 ---------------
        path = []
        current = start
        explored = {}
        while current != goal:
            path.append(current)
            explored[current] = nodes[current].parent
            min_cost = float('inf')
            next_node = None

            # 动态环境下选择最优邻居
            for neighbor in G_succ[current]:
                cost = nodes[neighbor].g + weight_func(current, neighbor, G_succ[current][neighbor])
                if cost < min_cost:
                    min_cost = cost
                    next_node = neighbor

            if next_node is None:
                break
            current = next_node

        path.append(goal)
        explored[goal] = None

        return path, nodes[start].rhs, explored
    def JPS(self, graph, source, target):
        """
        Jump Point Search 算法实现
        :param graph: 图结构
        :param source: 起点
        :param target: 终点
        :return: (路径列表, 路径成本, 探索过的节点字典)
        """
        # ------------ 初始化数据结构 ------------
        weight = 'weight'
        location = nx.get_node_attributes(graph, 'location')
        pos = nx.get_node_attributes(graph, 'pos')
        G_succ = graph._adj
        weight_func = _weight_function(graph, 'weight')       # 权重计算函数
        # 自定义优先级队列
        # ------------ 核心数据结构 ------------
        open_heap = []                  # 优先队列 (f_cost, node)
        heappush(open_heap, (0, source))
        came_from = {}                  # 路径回溯字典
        g_cost = {source: 0}            # 实际成本
        closed_set = set()              # 已探索节点
        # 启发式函数（曼哈顿距离）
        heuristic = lambda u, v: abs(location[v][0]-location[u][0]) + abs(location[v][1]-location[u][1])
        # ------------ 方向向量定义 ------------
        # 方向定义（四方向：东、南、西、北）
        directions = {
            'E': (1, 0),   # 东 (x+1)
            'S': (0, 1),   # 南 (y+1)
            'W': (-1, 0),  # 西 (x-1)
            'N': (0, -1)   # 北 (y-1)
        }
        # ------------ 工具函数 ------------
        def get_direction(current, neighbor):
            """根据坐标差判断移动方向"""
            dx = location[neighbor][0] - location[current][0]
            dy = location[neighbor][1] - location[current][1]
            if dx > 0:   return 'E'
            elif dx < 0: return 'W'
            elif dy > 0: return 'S'
            elif dy < 0: return 'N'
            else:        return None  # 同一位置（可能跨层）
        def has_forced_neighbor(current, dir):
            """
            判断当前方向是否存在强制邻居
            :param current: 当前节点
            :param dir: 当前移动方向 ('E','S','W','N')
            :return: 是否存在需要转向的强制邻居
            """
            # 获取当前节点坐标
            x, y = location[current]
            # 根据方向计算检查区域
            if dir == 'E':
                check_nodes = [(x, y+1), (x, y-1)]  # 北、南方向
            elif dir == 'W':
                check_nodes = [(x, y+1), (x, y-1)]
            elif dir == 'N':
                check_nodes = [(x+1, y), (x-1, y)]  # 东、西方向
            elif dir == 'S':
                check_nodes = [(x+1, y), (x-1, y)]

            # 在拓扑地图中检查实际存在的节点
            for (nx, ny) in check_nodes:
                for node in graph.nodes:
                    epsilon = 0.01  # 容差
                    # if location[node] == (nx, ny) and node in G_succ[current]:
                    if abs(location[node][0] - nx) < epsilon and abs(location[node][1] - ny) < epsilon and node in G_succ[current]:
                        print(f"强制邻居：{current} -> {node}")
                        return True
                    print(f"无强制邻居：{current} -> {node}")
            return False
        def jump(current, direction):
            """
            沿指定方向跳跃搜索
            :param current: 当前节点
            :param direction: 移动方向 ('E','S','W','N')
            :return: 下一个跳跃点或None
            """
            next_node = None
            # 沿方向查找直接邻居
            for neighbor in G_succ[current]:
                if get_direction(current, neighbor) == direction:
                    print(get_direction(current, neighbor), direction)
                    next_node = neighbor
                    break
                if not next_node:
                    return None  # 该方向无直接连接
                # 检查换层节点（需特殊处理）
                if pos[current][2] != pos[next_node][2]:
                    return next_node  # 换层节点视为跳跃点
                # 检查强制邻居条件
                if has_forced_neighbor(current, direction):
                    return next_node
                # 继续跳跃
                return jump(next_node, direction)
        # ------------ 主循环 ------------
        while open_heap:
            print(f"open_heap: {open_heap}")
            current_f, current = heappop(open_heap)
            if current == target:
                # 重构路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, g_cost[target], came_from
            closed_set.add(current)
            print(f"closed_set: {closed_set}")
            # 遍历所有可能方向
            for dir in directions:
                jump_point = jump(current, dir)
                print(f"jump_point: {jump_point}")
                if not jump_point or jump_point in closed_set:
                    continue
                # 计算新成本（使用实际边权重）
                new_g = g_cost[current] + weight_func(current, jump_point, G_succ[current][jump_point])
                if jump_point not in g_cost or new_g < g_cost[jump_point]:
                    came_from[jump_point] = current
                    g_cost[jump_point] = new_g
                    f_cost = new_g + heuristic(jump_point, target)
                    heappush(open_heap, (f_cost, jump_point))


        raise nx.NetworkXNoPath(f"No path between {source} and {target}")

    '''双向A*算法：返回 path,cost,explored'''
    def bidirectional_Astar(self, graph, source, target):
        if source not in graph or target not in graph:
            graph = self.TopoGraph
        if source == target:
            return [source], 0.0, {source: None}
        weight = 'weight'
        # 初始化数据结构
        push = heappush
        pop = heappop
        weight_func = _weight_function(graph, weight)
        G_succ = graph._adj
        location = nx.get_node_attributes(graph, 'location')
        heuristic = lambda u, v:  abs(location[v][0] - location[u][0]) +  abs(location[v][1] - location[u][1])
        pos = nx.get_node_attributes(graph, 'pos')
        source_layer = pos[source][2]
        target_layer = pos[target][2]
        # 记录提升机节点
        #是否换层
        cross_flag = source_layer != target_layer
        #起始层的提升机节点集
        source_cross_nodes_set = set(self.cross_nodes[source_layer])
        #目标层的提升机节点集
        target_cross_nodes_set = set(self.cross_nodes[target_layer]) if cross_flag else set()
        # 前向搜索（从起点开始）
        forward_queue = []
        forward_enqueued = {}
        forward_explored = {}
        c = count()
        push(forward_queue, (0, next(c), source, 0, None))
        forward_enqueued[source] = (0, heuristic(source, target))

        # 反向搜索（从终点开始）
        backward_queue = []
        backward_enqueued = {}
        backward_explored = {}
        push(backward_queue, (0, next(c), target, 0, None))
        backward_enqueued[target] = (0, heuristic(target, source))

        meeting_node = None
        min_path_length = float('inf')

        while forward_queue and backward_queue:
            # 交替扩展两个方向的节点
            for direction in ['forward', 'backward']:
                if not (forward_queue and backward_queue):
                    continue

                # 获取当前方向的队列
                if direction == 'forward':
                    queue = forward_queue
                    enqueued = forward_enqueued
                    explored = forward_explored
                    target_set = backward_enqueued
                    other_explored = backward_explored
                else:
                    queue = backward_queue
                    enqueued = backward_enqueued
                    explored = backward_explored
                    target_set = forward_enqueued
                    other_explored = forward_explored

                # 弹出当前最小代价节点
                _, __, curnode, dist, parent = pop(queue)

                if curnode in explored:
                    continue
                explored[curnode] = parent

                # 检查是否相遇
                if curnode in other_explored:
                    path_length = dist + backward_enqueued.get(curnode, (0,0))[0]
                    if path_length < min_path_length:
                        meeting_node = curnode
                        min_path_length = path_length

                # 扩展邻居节点
                for neighbor, edge in G_succ[curnode].items():
                    if cross_flag == True: # 换层标志
                        if curnode in source_cross_nodes_set: # 当前节点是起始层提升机节点
                            if pos[neighbor][2] == source_layer:  #邻居节点层数不是换层节点，跳过
                                continue
                        # 换层后也不能经过提升机节点
                        elif neighbor in target_cross_nodes_set:  # #当前节点不是起始层提升机节点，且邻居节点是目标层提升机节点，跳过
                            continue
                    #不换层
                    elif neighbor in source_cross_nodes_set:  #跳过起始层提升机节点
                        continue
                    cost = weight_func(curnode, neighbor, edge)
                    if cost is None:
                        continue
                    new_dist = dist + cost

                    # 处理换层逻辑（根据原有逻辑调整）
                    source_layer = pos[curnode][2]
                    neighbor_layer = pos[neighbor][2]
                    if source_layer != neighbor_layer and curnode not in self.cross_nodes[source_layer]:
                        continue

                    if neighbor in enqueued:
                        qcost, h = enqueued[neighbor]
                        if qcost <= new_dist:
                            continue
                    else:
                        h = heuristic(neighbor,target if direction == 'forward' else source)

                    enqueued[neighbor] = (new_dist, h)
                    push(queue, (new_dist + h, next(c), neighbor, new_dist, curnode))

                # 检查是否找到最优路径
                if meeting_node is not None:
                    # 重构路径
                    forward_path = []
                    node = meeting_node
                    while node is not None:
                        forward_path.append(node)
                        node = forward_explored[node]
                    forward_path.reverse()

                    backward_path = []
                    node = meeting_node
                    while node is not None:
                        node = backward_explored[node]
                        if node is not None:
                            backward_path.append(node)
                    full_path = forward_path + backward_path
                    total_cost = min_path_length
                    explored_nodes = {**forward_explored, **backward_explored}
                    return full_path, round(nx.path_weight(graph,full_path,'weight'), 2), explored_nodes
        raise nx.NetworkXNoPath(f"No path between {source} and {target}")

    ## 返回路径，路径长度，探索过的节点
    def A_star(self,graph, source, target):
        if source not in graph or target not in graph:
            graph = self.TopoGraph
        elif source == target:
            return 0, 0, 0
        weight = 'weight'
        location = nx.get_node_attributes(graph, 'location')
        pos = nx.get_node_attributes(graph, 'pos')
        source_layer = pos[source][2]
        target_layer = pos[target][2]
        #是否换层
        cross_flag = source_layer != target_layer
        #起始层的提升机节点集
        source_cross_nodes_set = set(self.cross_nodes[source_layer])
        #目标层的提升机节点集
        target_cross_nodes_set = set(self.cross_nodes[target_layer])

        heuristic = lambda u, v: abs(location[v][0] - location[u][0]) + abs(location[v][1] - location[u][1])

        push = heappush    # push function for the heap堆的Push函数
        pop = heappop     # pop function for the heap堆的Pop函数
        weight_function = _weight_function(graph, weight)  # weight function for the graph
        G_succ = graph._adj  # 用于存储图中每个节点的邻接信息{1: {2: {'weight': 1.5}}, 2: {1: {'weight': 1.5}, 3: {'weight': 1.5}}, 3: {2: {'weight': 1.5}}}
        c = count()  # 计数器，用于生成唯一的ID
        queue = [(0, next(c), source, 0 ,None)]  # 队列，元素为元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
        enqueued = {}  # 记录节点是否已经入队，记录到达节点的距离  和 节点到目标节点启发式评估值
        explored = {}  # 记录节点是否已经探索过
        while queue:
            # 弹出队列中代价最小的元素
            # 元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
            _, __, curnode, dist, parent = pop(queue)
            if curnode == target:
                path = [curnode]
                node = parent
                while node is not None:
                    path.append(node)
                    node = explored[node]
                path.reverse()
                return path, round(dist,2) , explored
                # return path, round(nx.path_weight(graph, path, weight),2) , explored
            if curnode in explored:         # 已经探索过？
                if explored[curnode] is None:  # 父节点是None,说明是源节点，跳过
                    continue
                # 之前的距离和本次的距离比较谁优
                qcost, h = enqueued[curnode]
                if qcost <= dist:  # 之前的距离更优，跳过
                    continue
            explored[curnode] = parent  # 标记为已经探索过
            # 遍历当前节点的邻居节点
            for neighbor, datas in G_succ[curnode].items():
                if cross_flag == True: # 换层标志
                    if curnode in source_cross_nodes_set: # 当前节点是起始层提升机节点
                        if pos[neighbor][2] == source_layer:  #邻居节点层数不是换层节点，跳过
                            # print("换层节点跳过",neighbor)
                            continue
                        # 换层后也不能经过提升机节点
                elif neighbor in target_cross_nodes_set:  # #当前节点不是起始层提升机节点，且邻居节点是目标层提升机节点，跳过
                    # print("换层后也不能经过提升机节点",neighbor)
                    continue
                #不换层
                elif neighbor in source_cross_nodes_set:  #跳过起始层提升机节点
                    continue

                # 计算从当前节点到邻居节点的距离
                cost = weight_function(curnode, neighbor, datas)
                if cost is None:
                    continue
                ncost = dist + cost     # 到邻居节点的成本
                if neighbor in enqueued:
                    qcost, h = enqueued[neighbor]
                    if qcost <= ncost:  # 之前的距离更优，跳过
                        continue
                else:               # 当前距离更优，更新队列
                    h = heuristic(neighbor, target)
                enqueued[neighbor] = (ncost, h)
                push(queue, (ncost + h, next(c), neighbor, ncost, curnode))
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

    '''测试权重1.2、转弯成本，绝对值ATL'''
    def abs_ATL_star(self,graph, source, target,landmarks=None):
        # beta = 1.3 #1.2  # h(n)预估函数的权重
        if source not in graph or target not in graph:
            graph = self.TopoGraph
        elif source == target:
            return 0, 0, 0
        landmarks = graph.graph['landmarks']
        weight = 'weight'
        location = nx.get_node_attributes(graph, 'location')
        pos = nx.get_node_attributes(graph, 'pos')
        source_layer = pos[source][2]
        target_layer = pos[target][2]
        #是否换层
        cross_flag = source_layer != target_layer
        #起始层的提升机节点集
        # source_cross_nodes_set = set(self.cross_nodes[source_layer])
        # #目标层的提升机节点集
        # target_cross_nodes_set = set(self.cross_nodes[target_layer]) if cross_flag else set()

        push = heappush    # push function for the heap堆的Push函数
        pop = heappop     # pop function for the heap堆的Pop函数
        weight_function = _weight_function(graph, weight)  # weight function for the graph
        G_succ = graph._adj  # 用于存储图中每个节点的邻接信息{1: {2: {'weight': 1.5}}, 2: {1: {'weight': 1.5}, 3: {'weight': 1.5}}, 3: {2: {'weight': 1.5}}}
        c = count()  # 计数器，用于生成唯一的ID
        queue = [(0, next(c), source, 0 ,None)]  # 队列，元素为元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
        enqueued = {}  # 记录节点是否已经入队，记录到达节点的距离  和 节点到目标节点启发式评估值
        explored = {}  # 记录节点是否已经探索过
        def ATL_heuristic( neighbor, target,pos, location,landmarks):
            heuristic = abs(location[target][0] - location[neighbor][0]) +  abs(location[target][1] - location[neighbor][1])# + abs(pos[target][2] - pos[neighbor][2])
            # ATL距离启发式函数, A为起点，target为终点，landmarks为地标点集合，location为节点坐标集合
            if landmarks is not None:
                lower_bounds = [abs(landmarks[L][target] - landmarks[L][neighbor]) for L in landmarks]
                heuristic = max(heuristic, *lower_bounds)
            return heuristic
        while queue:
            # 弹出队列中代价最小的元素
            # 元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
            _, __, curnode, dist, parent = pop(queue)
            if curnode == target:
                path = [curnode]
                node = parent
                while node is not None:
                    path.append(node)
                    node = explored[node]
                path.reverse()
                return path, round(nx.path_weight(graph, path, weight),2) , explored
            if curnode in explored:         # 已经探索过？
                if explored[curnode] is None:  # 父节点是None,说明是源节点，跳过
                    continue
                # 之前的距离和本次的距离比较谁优
                qcost, h = enqueued[curnode]
                if qcost <= dist:  # 之前的距离更优，跳过
                    continue
            explored[curnode] = parent  # 标记为已经探索过
            # 遍历当前节点的邻居节点
            for neighbor, datas in G_succ[curnode].items():
                # if cross_flag == True: # 换层标志
                #     if curnode in source_cross_nodes_set: # 当前节点是起始层提升机节点
                #         if pos[neighbor][2] == source_layer:  #邻居节点层数不是换层节点，跳过
                #             continue
                #     # 换层后也不能经过提升机节点
                #     elif neighbor in target_cross_nodes_set:  # #当前节点不是起始层提升机节点，且邻居节点是目标层提升机节点，跳过
                #         continue
                #     #不换层
                # elif neighbor in source_cross_nodes_set:  #跳过起始层提升机节点
                #     continue
                # 计算从当前节点到邻居节点的距离
                cost = weight_function(curnode, neighbor, datas)
                if cost is None:
                    continue
                # ncost = dist + cost     # 到邻居节点的成本
                turn_cost = self.Turn_Cost(location,parent,neighbor,target)  # 引入拐点成本
                ncost = dist + cost + turn_cost    # 到邻居节点的成本
                if neighbor in enqueued:
                    qcost, h = enqueued[neighbor]
                    if qcost <= ncost:  # 之前的距离更优，跳过
                        continue
                else:               # 当前距离更优，更新队列
                    h = ATL_heuristic(neighbor, target,pos,location, landmarks)
                D = ATL_heuristic(source, curnode,pos,location, landmarks) # 引入拐点成本
                if   D == 0:
                    D = 1
                beta = math.e**(h/D) # 动态权重计算  从2-1进行调整
                h = h * beta
                enqueued[neighbor] = (ncost, h)
                push(queue, ( ncost + h, next(c), neighbor, ncost, curnode))
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

    '''测试权重1.2、转弯成本ATL'''
    def ATL_star(self,graph, source, target,landmarks=None):
        weight = 'weight'
        alpha = 1  # g(n)实际成本函数的权重
        beta = 1.2 #1.2  # h(n)预估函数的权重
        if source not in graph or target not in graph:
            graph = self.TopoGraph
        elif source == target:
            return 0, 0, 0
        location = nx.get_node_attributes(graph, 'location')
        pos = nx.get_node_attributes(graph, 'pos')
        source_layer = pos[source][2]
        target_layer = pos[target][2]
        #是否换层
        cross_flag = source_layer != target_layer
        #起始层的提升机节点集
        source_cross_nodes_set = set(self.cross_nodes[source_layer])
        #目标层的提升机节点集
        target_cross_nodes_set = set(self.cross_nodes[target_layer]) if cross_flag else set()

        # cost = nx.path_weight(graph, path, weight)
        push = heappush    # push function for the heap堆的Push函数
        pop = heappop     # pop function for the heap堆的Pop函数
        weight_function = _weight_function(graph, weight)  # weight function for the graph
        G_succ = graph._adj  # 用于存储图中每个节点的邻接信息{1: {2: {'weight': 1.5}}, 2: {1: {'weight': 1.5}, 3: {'weight': 1.5}}, 3: {2: {'weight': 1.5}}}
        c = count()  # 计数器，用于生成唯一的ID
        queue = [(0, next(c), source, 0 ,None)]  # 队列，元素为元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
        enqueued = {}  # 记录节点是否已经入队，记录到达节点的距离  和 节点到目标节点启发式评估值
        explored = {}  # 记录节点是否已经探索过
        def ATL_heuristic(neighbor, target,  location,landmarks = None):
            heuristic = abs(location[target][0] - location[neighbor][0]) +  abs(location[target][1] - location[neighbor][1])
            # ATL距离启发式函数, A为起点，target为终点，landmarks为地标点集合，location为节点坐标集合
            for L in landmarks:
                # lower_bound = abs(self.first_landmarks[L][target] - self.first_landmarks[L][neighbor])
                lower_bound = landmarks[L][target] - landmarks[L][neighbor]
                if lower_bound - heuristic > 0.001 :
                    # print("A:", A, "target:", Z ,"lower_bound:", lower_bound, "h:", heuristic)
                    heuristic = lower_bound
            # print("manhattan_distance:",manhattan_distance,"heuristic:",heuristic)
            return heuristic
        while queue:
            # 弹出队列中代价最小的元素
            # 元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
            _, __, curnode, dist, parent = pop(queue)
            if curnode == target:
                path = [curnode]
                node = parent
                while node is not None:
                    path.append(node)
                    node = explored[node]
                path.reverse()
                return path, round(nx.path_weight(graph, path, weight),2) , explored
            if curnode in explored:         # 已经探索过？
                if explored[curnode] is None:  # 父节点是None,说明是源节点，跳过
                    continue
                # 之前的距离和本次的距离比较谁优
                qcost, h = enqueued[curnode]
                if qcost <= dist:  # 之前的距离更优，跳过
                    continue
            explored[curnode] = parent  # 标记为已经探索过
            # 遍历当前节点的邻居节点
            for neighbor, datas in G_succ[curnode].items():
                # if cross_flag == True: # 换层标志
                #     if curnode in source_cross_nodes_set: # 当前节点是起始层提升机节点
                #         if pos[neighbor][2] == source_layer:  #邻居节点层数不是换层节点，跳过
                #             continue
                #     # 换层后也不能经过提升机节点
                #     elif neighbor in target_cross_nodes_set:  # #当前节点不是起始层提升机节点，且邻居节点是目标层提升机节点，跳过
                #         continue
                #     #不换层
                # elif neighbor in source_cross_nodes_set:  #跳过起始层提升机节点
                #     continue
                # 计算从当前节点到邻居节点的距离
                cost = weight_function(curnode, neighbor, datas)
                if cost is None:
                    continue
                # ncost = dist + cost     # 到邻居节点的成本
                turn_cost = self.Turn_Cost(location,parent,neighbor,target)  # 引入拐点成本
                ncost = dist + cost + turn_cost    # 到邻居节点的成本
                if neighbor in enqueued:
                    qcost, h = enqueued[neighbor]
                    if qcost <= ncost:  # 之前的距离更优，跳过
                        continue
                else:               # 当前距离更优，更新队列
                    h = ATL_heuristic(neighbor, target,location, landmarks)
                #方向成本
                h = h * beta # 权重
                enqueued[neighbor] = (ncost, h)
                push(queue, (alpha * ncost + h, next(c), neighbor, ncost, curnode))
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")


    '''引入权重1.2，拐点成本'''
    ## 返回路径，路径长度，探索过的节点
    def improve_A_star(self, graph, source, target):
        try:
            weight = 'weight'
            # alpha = 1  # g(n)实际函数的权重
            # beta = 1.2  # h(n)预估函数的权重
            if source not in graph or target not in graph:
                graph = self.TopoGraph
            elif source == target:
                return 0, 0, 0
            location = nx.get_node_attributes(graph, 'location')
            pos = nx.get_node_attributes(graph, 'pos')
            source_layer = pos[source][2]
            target_layer = pos[target][2]
            #是否换层
            cross_flag = source_layer != target_layer
            #起始层的提升机节点集
            source_cross_nodes_set = set(self.cross_nodes[source_layer])
            #目标层的提升机节点集
            target_cross_nodes_set = set(self.cross_nodes[target_layer]) if cross_flag else set()
            # 以画布中的绝对位置作为启发式预估函数参数
            heuristic = lambda u, v: abs(location[v][0] - location[u][0]) + abs(location[v][1] - location[u][1])

            push = heappush    # push function for the heap堆的Push函数
            pop = heappop     # pop function for the heap堆的Pop函数
            weight_function = _weight_function(graph, weight)  # weight function for the graph
            G_succ = graph._adj  # 用于存储图中每个节点的邻接信息{1: {2: {'weight': 1.5}}, 2: {1: {'weight': 1.5}, 3: {'weight': 1.5}}, 3: {2: {'weight': 1.5}}}
            c = count()  # 计数器，用于生成唯一的ID
            queue = [(0, next(c), source, 0 ,None)]  # 队列，元素为元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
            enqueued = {}  # 记录节点是否已经入队，记录到达节点的距离  和 节点到目标节点启发式评估值
            explored = {}  # 记录节点是否已经探索过,记录父节点
            while queue:
            #     # 弹出队列中代价最小的元素
            #     # 元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
                _, _, current_node, dist_current_node,parent = pop(queue)
                if current_node == target:     # 1.找到目标节点
                    path = [current_node]
                    node = parent
                    while node is not None:
                        path.append(node)           # 反向构建路径
                        node = explored[node]       # 回溯父节点
                    path.reverse()                  # 反转路径
                    return path, round(nx.path_weight(graph, path, weight),2), explored  # 返回路径，路径长度，探索过的节点
                if current_node in explored:        # 2.已经探索过，跳过
                    if explored[current_node] is None:  # 已经探索过，但父节点为None，说明是源节点，跳过
                        continue
                    qcost, h = enqueued[current_node]   # 取出到当前节点的距离和启发式评估值
                    if qcost < dist_current_node:       # 已经探索过，且距离更短所以方案更优，跳过本次探索
                        continue
                explored[current_node] = parent         # 记录父节点
                for neighbor, w in G_succ[current_node].items():   # 3.遍历当前节点的邻居节点
                    # if cross_flag == True: # 换层标志
                    #     if current_node in source_cross_nodes_set: # 当前节点是起始层提升机节点
                    #         if pos[neighbor][2] == source_layer:  #邻居节点层数不是换层节点，跳过
                    #             continue
                    #     # 换层后也不能经过提升机节点
                    #     elif neighbor in target_cross_nodes_set:  # #当前节点不是起始层提升机节点，且邻居节点是目标层提升机节点，跳过
                    #         continue
                    # #不换层
                    # elif neighbor in source_cross_nodes_set:  #跳过起始层提升机节点
                    #     continue
                    cost = weight_function(current_node, neighbor, w)  # 计算当前节点到邻居节点的距离
                    # print(f"current_node:",current_node,"neighbor:",neighbor,"w:",w,"cost:",cost)
                    #current_node: 1469 neighbor: 1468 w: {'weight': 2.02} cost: 2.02
                    if cost is None:#不可达的节点，跳过
                        continue
                    turn_cost = self.Turn_Cost(location,parent,neighbor,target)  # 引入拐点成本
                    ncost =  dist_current_node + cost + turn_cost      # 计算起点到邻居节点的总代价：起点到当前节点的距离 + 当前节点到邻居节点的距离
                    # ncost = dist_current_node + cost  # 计算起点到邻居节点的总代价：起点到当前节点的距离 + 当前节点到邻居节点的距离
                    if neighbor  in enqueued:         # 4.判断邻居节点是否已经入队
                        qcost, h = enqueued[neighbor]  # 取出邻居节点的距离和启发式评估值
                        if qcost <= ncost:             # 邻居节点已经入队，且距离更短，跳过本次邻居节点的探索
                            continue
                    else:# 5.邻居节点没有入队
                        h = heuristic(neighbor, target)  # 计算邻居节点到目标节点的启发式评估值
                    D = heuristic(source, current_node)  # 计算当前节点到源节点的启发式评估值
                    if   D == 0:
                        D = 1
                    beta = math.e**(h/D) # 动态权重计算  从2-1进行调整
                    h = h * beta
                    enqueued[neighbor] = (ncost, h)      # 记录起点到邻居节点的距离ncost和邻居节点到目标节点的启发式评估值h
                    push(queue, (ncost + h, next(c), neighbor, ncost, current_node))  # 6.入队，并更新队列
                    # f = ncost + h 为A*算法的评估值，用于判断节点的优先级，使得算法更加贪婪，更加关注距离短的节点
            raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))  # 6.找不到路径，抛出异常
        except Exception as e:
            print(f"改进A* - Exception: {e}")


    def Dijkstra(self,graph, source, target):
        # 调用networkx的dijkstra算法
        if source not in graph or target not in graph:
            graph = self.TopoGraph
        elif source == target:
            return 0, 0, 0
        weight = 'weight'
        cutoff = None
        pred = None
        paths ={source: [source]}          #保存路径，用于存储从源节点到每个其他节点的路径列表。字典的键是节点标签，值是从源节点到该节点的路径（一个节点序列）。
        pos = nx.get_node_attributes(graph, 'pos')
        source_layer = pos[source][2]
        target_layer = pos[target][2]
        #是否换层
        cross_flag = source_layer != target_layer
        #起始层的提升机节点集
        source_cross_nodes_set = set(self.cross_nodes[source_layer])
        #目标层的提升机节点集
        target_cross_nodes_set = set(self.cross_nodes[target_layer]) if cross_flag else set()

        #G_succ图 G 的邻接表，存储了每个节点的后继节点。{1: {2: {'weight': 1.5}}, 2: {1: {'weight': 1.5}, 3: {'weight': 1.5}}......}
        G_succ = graph._adj  # For speed-up (and works for both directed and undirected graphs)
        weight_function = _weight_function(graph, weight)  # weight function for the graph
        push = heappush         #push 和 pop：分别是用于插入和移除优先队列元素的函数
        pop = heappop
        dist = {}               # 【闭合列表】用于存储从源节点到其他节点的最终距离。【确认了最短路径】
        seen = {}               # 记录每个节点的当前最短路径长度。【暂时的，动态的】
        # fringe is heapq with 3-tuples (distance,c,node)
        # use the count c to avoid comparing nodes (may not be able to)
        c = count()
        fringe = []             # 优先队列（最小堆）【开放列表】，存储节点及其对应的路径成本，通过 (距离, 计数, 节点) 的三元组进行管理，以避免节点比较。
        # for source in sources:
        seen[source] = 0
        push(fringe, (0, next(c), source))
        while fringe:
            (d, _, current_node) = pop(fringe)      # 当 fringe 不为空时，弹出最小距离的节点 (d, _, v)。
            if current_node in dist:
                continue  # already searched this node.
            dist[current_node] = d
            if current_node == target:
                return paths[current_node], round(dist[current_node],2), list(dist.keys())            # 找到目标节点，返回路径长度和路径
                # break
            current_layer = pos[current_node][2]
            for neighbor, e in G_succ[current_node].items():      # 遍历当前节点的邻居节点
                # if cross_flag == True: # 换层标志
                #     if current_node in source_cross_nodes_set: # 当前节点是起始层提升机节点
                #         if pos[neighbor][2] == source_layer:  #邻居节点层数不是换层节点，跳过
                #             continue
                #     # 换层后也不能经过提升机节点
                #     elif neighbor in target_cross_nodes_set:  # #当前节点不是起始层提升机节点，且邻居节点是目标层提升机节点，跳过
                #         continue
                #     #不换层
                # elif neighbor in source_cross_nodes_set:  #跳过起始层提升机节点
                #     continue
                cost = weight_function(current_node, neighbor, e)          # 计算当前节点到邻居节点的距离
                if cost is None:                # 无法到达的节点，跳过
                    continue
                vu_dist = dist[current_node] + cost        # 到邻居节点的总代价：起点到当前节点的距离 + 当前节点到邻居节点的距离
                if cutoff is not None:          # 距离超过限制阈值，则跳过本次邻居节点的探索
                    if vu_dist > cutoff:
                        continue
                if neighbor in dist:                   # 已经探索过，且距离更短，跳过本次邻居节点的探索
                    u_dist = dist[neighbor]
                    if vu_dist < u_dist:
                        #如果新路径长度更短，则说明存在矛盾路径，可能是由于图中存在负权重边或环（即在最短路径算法中不应出现的情形）。
                        raise ValueError("Contradictory paths found:", "negative weights?")
                    #当前路径和之前发现的路径长度是否相同
                    elif pred is not None and vu_dist == u_dist:
                        pred[neighbor].append(current_node)       #记录多条路径可能导致同样的最短距离。这在某些情况下是有用的，如寻找多条路径的场景。
                elif neighbor not in seen or vu_dist < seen[neighbor]:    # 邻居节点没有入队，或距离更短，则入队
                    seen[neighbor] = vu_dist
                    push(fringe, (vu_dist, next(c), neighbor))
                    if paths is not None:
                        paths[neighbor] = paths[current_node] + [neighbor]
                    if pred is not None:
                        pred[neighbor] = [current_node]
                elif vu_dist == seen[neighbor]:
                    if pred is not None:
                        pred[neighbor].append(current_node)
        raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))  # 6.找不到路径，抛出异常
        # The optional predecessor and path dictionaries can be accessed
        # by the caller via the pred and paths objects passed as arguments.

    """==============功能函数================="""
    #计算优化率
    def cal_optimity(self,algorithm_results):
        algorithms = algorithm_results.keys()
        optimity = {}
        # #计算每个算法各个指标的平均值
        for algorithm in algorithms:
            optimity[algorithm] = {}
            for metric in algorithm_results[algorithm].keys():
                val = round(sum(algorithm_results[algorithm][metric])/len(algorithm_results[algorithm][metric]),3)
                optimity[algorithm][metric] = val
                print(f"算法：{algorithm}，指标：{metric}，平均值：{val}")
        #计算ATL_star的优化率
        algorithms_to_compare = [ 'D*Lite', '双向A*', '改进A*','双向A*']
        metrics = ['take_time', 'cal_path_time', 'explored']
        for metric in metrics:
            for compare_algo in algorithms_to_compare:
                base_algo = '本文方法'
                optim_rate = (optimity[base_algo][metric] - optimity[compare_algo][metric]) / optimity[compare_algo][metric]
                print(f"{base_algo}——{compare_algo}——{metric}-优化率：{round(optim_rate, 3)}")


    # 转向成本1.3，目标节点转向成本0.5
    def Turn_Cost(self,location,parent,next_node,target):
        turn_cost = 1.3#1.5  # 普通节点转向代价  权重在1的时候，拐点数量还是较多
        target_turn_cost = 0.5  # 目标节点转向代价
        if parent is None or next_node is None or target is None:   #起始父节点是None，说明是源节点，跳过
            # print("参数错误")
            return 0
        if parent == next_node:     # 当前current的 邻居节点是父节点
            # print("父节点 = 下一邻居节点相同")
            return 0
        parent_pos = location[parent]
        # current_pos = location[current_node]
        next_pos = location[next_node]
        # 计算父节点、下一节点在 x 和 y 轴上的偏差
        delta_x = abs(parent_pos[0] - next_pos[0])  # x轴偏差
        delta_y = abs(parent_pos[1] - next_pos[1])  # y轴偏差
        # print(f"parent_pos[0]={parent_pos[0]}, parent_pos[1]={parent_pos[1]}, next_pos[0]={next_pos[0]}, next_pos[1]={next_pos[1]},delta_x={delta_x},delta_y={delta_y}")
        # 设定一个阈值，假设阈值为某个常量，例如 1.0
        threshold = 0.1
        # 判断是否在 x 和 y 轴上都超过阈值,出现拐点
        if delta_x > threshold and delta_y > threshold:
            if next_node == target:      # 目标节点，转向代价
                return target_turn_cost
            else:                        # 普通节点，转向代价
                return turn_cost
        return 0

    '''累计转向次数'''
    def Turn_Count(self,graph, path):
        # 设定一个阈值，假设阈值为某个常量，例如 1.0
        threshold = 1.0
        turn_count = 0
        location = nx.get_node_attributes(graph, 'location')
        # cost = nx.path_weight(graph, path, 'weight')
        # 遍历路径列表，从第一个点开始，直到倒数第三个点
        for i in range(1, len(path) - 1):
            parent = path[i - 1]      # 前一个节点
            # current_node = path[i]    # 当前节点
            next_node = path[i + 1]   # 下一个节点
            parent_pos = location[parent]
            # current_pos = location[current_node]
            next_pos = location[next_node]
            delta_x = abs(parent_pos[0] - next_pos[0])  # x轴偏差
            delta_y = abs(parent_pos[1] - next_pos[1])  # y轴偏差
            # 判断是否在 x 和 y 轴上都超过阈值
            if delta_x > threshold and delta_y > threshold:
                turn_count += 1  # 如果转向，计数加
        return turn_count

    '''判断next_node与current_node是否存在转向'''
    def is_turn(self,location,parent,next_node):
        if parent is None or next_node is None:
            return False
        # 设定一个阈值，假设阈值为某个常量，例如 1.0
        threshold = 1.0
        parent_pos = location[parent]
        next_pos = location[next_node]
        delta_x = abs(parent_pos[0] - next_pos[0])  # x轴偏差
        delta_y = abs(parent_pos[1] - next_pos[1])  # y轴偏差
        # 判断是否在 x 和 y 轴上都超过阈值
        if delta_x > threshold and delta_y > threshold:
            return True
        return False

    def save_image(self,name,folder="pics"):
        # 获取当前文件的目录
        current_directory = os.path.dirname(os.path.abspath(__file__))
        # print(f"当前文件目录：{current_directory}")                #D:\Application_Data\MyPythonProject\Program
        # 获取当前文件的父目录
        parent_directory = os.path.dirname(current_directory)     #D:\Application_Data\MyPythonProject
        # 拼接出完整的文件夹路径
        folder_path = os.path.join(parent_directory, folder)
        # 检查pics文件夹是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            # print(f"保存图像{start}-{end}-{algorithm}-{heristic_name}.png，文件夹{folder_path}不存在，创建文件夹！")
            os.makedirs(folder_path)
        filename = f"{name}.png"
        file_path = os.path.join(folder_path, filename)
        # 生成文件名
        # file_path = "pics/" + filename
        # ax.set_title(f"{name} ")
        self.figure.savefig(file_path, dpi=300)
        print(f"保存图片成功！{file_path}")

        # 计算路径时间成本
    def cal_path_time(self, graph, path):
        time_cost = 0      #初始化路径时间
        turning_time = 4   #单次转向时间
        ele_time = 0       #初始化提升机换层时间
        '''计算四向车运行时间'''
        def cal_time(sub_path):
            path_time = 0
            Vmax = 1.2; Acc = 0.3; Dcc = 0.3;   # 四向车最大速度，加速度，减速度
            t_acc = round(Vmax / Acc, 3)   #加速时间
            t_dcc = round(Vmax / Dcc, 3)   #减速时间
            S_acc = round(Vmax**2 / (2*Acc), 3)   #从0加速到最大速度所需距离
            S_dcc = round(Vmax**2 / (2*Dcc), 3)    #从最大速度减速到0所需距离
            S0 = S_dcc + S_acc   #临界距离
            length =  round(nx.path_weight(graph, sub_path, 'weight'),3)
            if length >= S0:      # 路径长度超过了从0加速到最大速度所需距离+从最大速度减速到0所需距离
                constant_time = round((length - S_acc - S_dcc)/Vmax,3)   # 计算匀速时间
                path_time += constant_time + t_acc + t_dcc   # 路径时间 = 匀速时间 + 加速时间 + 减速时间
                # print(f"四向车达到Vmax,sub_path={sub_path},length={length},constant_time={constant_time},path_time={path_time}")
            else: #匀加速到非最大速度后，立即做匀减速运动
                t = round(math.sqrt(2*length*(1/Acc + 1/Dcc)),3)
                path_time += t
                # print(f"四向车非最大速度，sub_path={sub_path},length={length},t={t},path_time={path_time}")
            return path_time

        '''计算垂直段运行时间：'''
        def calculate_vertical_time(dz):
            Acc_lift = 0.15;                              #提升机加速度
            Dec_lift = 0.15;                              #提升机减速度
            Max_speed_lift = 1.4;                         #提升机最大速度
            t_acc = round(Max_speed_lift / Acc_lift,3)     # 加速度阶段时间
            t_dec = round(Max_speed_lift / Dec_lift,3)     # 减速阶段时间
            s_acc = round(0.5 * Acc_lift * t_acc**2 ,3 )     # 加速度阶段位移
            s_dec = round(0.5 * Dec_lift * t_dec**2,3)       # 减速阶段位移
            total_acc_dec_distance = s_acc + s_dec  # 临界加速减速距离
            if dz <= total_acc_dec_distance:
                # math.sqrt(2*height_diff*(1/Acc + 1/Dcc))
                return round((2*dz*(1/Acc_lift + 1/Dec_lift))**0.5,3)
            else:
                t_cruise = round((dz - total_acc_dec_distance) / Max_speed_lift,3)
                return  t_acc + t_dec + t_cruise

        '''获取路径中拐点、提升机换层节点'''
        def get_special_nodes(graph,path):
            turn_nodes = []      # 转向节点
            elevator_nodes = []  # 提升机换层节点
            location = nx.get_node_attributes(graph, 'location')
            threshold = 1.0
            # 采用滑动窗口进行路径特征分析
            for i in range(1, len(path) - 1):
                parent = path[i - 1]      # 前一个节点
                current_node = path[i]    # 当前节点
                next_node = path[i + 1]   # 下一个节点
                parent_pos = location[parent]
                next_pos = location[next_node]
                delta_x = abs(parent_pos[0] - next_pos[0])  # x轴偏差
                delta_y = abs(parent_pos[1] - next_pos[1])  # y轴偏差
                # 判断是否在 x 和 y 轴上都超过阈值
                if delta_x > threshold and delta_y > threshold:
                    turn_nodes.append(current_node)
                # 判断是否是提升机换层节点
                elif current_node in self.cross_nodes_list :
                    # print("提升机换层节点：",current_node)
                    elevator_nodes.append(current_node)
            return turn_nodes, elevator_nodes

        #二维列表，存储划分后的路径
        result = []
        #引入转向节点列表,提升机换层节点列表
        turn_nodes, elevator_nodes = get_special_nodes(graph,path)
        critical_nodes = sorted(set(turn_nodes + elevator_nodes),   # 使用集合（set）来去重
                                key=lambda x: path.index(x))    # 通过 path.index(x) 方法，以确保节点按它们在原路径中出现的顺序排列。
        start_index = 0
        last_is_ele_node = False  # 记录前一个节点是否是提升机节点

        if len(elevator_nodes) == 0:  # 路径中没有提升机换层节点
            #遍历分割点，剥离提升机换层路段
            for point in critical_nodes:
                # 找到分割点在原列表中的索引
                index = path.index(point,start_index)   #从start_index开始查找
                # 将从上次开始索引到当前分割点的子列表添加到结果列表中
                sub_path = path[start_index:index+1]       #加1，以包含 index 所指向的元素。
                result.append(sub_path)
                # 计算子路径时间
                time_cost += cal_time(sub_path)
                # # 更新开始索引
                start_index = index
            if start_index < len(path):  # 处理最后一段路径.（即 start_index 仍然指向路径中的某个位置）
                sub_path = path[start_index:]
                result.append(sub_path)
                time_cost += cal_time(sub_path)
        else:
            #todo: 分析提升机的耗时,考虑是否只有一个提升机，增加节点对应的提升机信息
            print("PATH:",path," 提升机换层节点列表：",elevator_nodes)
            try:
                ele_length = round(nx.path_weight(graph,elevator_nodes, 'weight'),3)
            except:
                print("提升机ele_length计算错误,",elevator_nodes)
            # ele_time = len(elevator_nodes) * 8
            time_cost += ele_time
            # print("提升机换层时间：",ele_time)
            #遍历分割点，剥离提升机换层路段
            for point in critical_nodes:
                if point in elevator_nodes :
                    if last_is_ele_node == False :     #本次是提升机节点，且前一个节点不是提升机节点
                        last_is_ele_node = True        #更新提升机节点
                    else:   # True 本次是提升机节点，且前一个节点也是提升机节点,说明是提升机路段，跳过
                        start_index += 1
                        continue
                else:
                    last_is_ele_node = False  # 记录前一个节点不是提升机节点

                # 找到分割点在原列表中的索引
                index = path.index(point,start_index)   #从start_index开始查找
                # 将从上次开始索引到当前分割点的子列表添加到结果列表中
                sub_path = path[start_index:index+1]       #加1，以包含 index 所指向的元素。
                result.append(sub_path)
                # 计算子路径时间
                time_cost += cal_time(sub_path)
                # # 更新开始索引
                start_index = index
            if start_index < len(path):  # 处理最后一段路径.（即 start_index 仍然指向路径中的某个位置）
                sub_path = path[start_index:]
                result.append(sub_path)
                time_cost += cal_time(sub_path)

        turn_count = len(turn_nodes)  # 计算转向次数
        turn_time = turn_count * turning_time  # 计算转向时间
        time_cost += turn_time  # 路径时间 = 路径时间 + 转向时间
        # print(f"分割点列表：{result}", " 换层节点：" ,elevator_nodes," 转向次数：",turn_count,"，转向时间：",turn_time," ele_time：",ele_time,"，路径时间：",time_cost)
        return time_cost

    '''出图方法'''
    def plot_results2(self, algorithm_results):
        # 提取算法名称和任务数
        algorithms = list(algorithm_results.keys())
        num_tasks = len(algorithm_results['本文方法']['cost'])  # 假设所有算法的任务数相同

        # 初始化画布和子图
        fig= plt.figure(figsize=(15, 10))

        # ------------------------- 1. 路径成本 -------------------------
        ax1 = fig.add_subplot(221)
        for algo in algorithms:
            costs = algorithm_results[algo]['cost']
            ax1.plot(range(1, num_tasks+1), costs,
                     marker='o',
                     label=algo)#.replace('_star', '*').replace('本文方法', '本文方法 ').replace("双向A*",'双向A*').replace('improve_A*','改进A*'))
        # 设置y轴的最大值以避免与图例冲突
        ax1.set_ylim(0, max(max(algorithm_results[algo]['cost'] for algo in algorithms)) * 1.2)
        ax1.set_title('路径成本对比', fontsize=13, fontweight='bold')
        ax1.set_xlabel('任务编号', fontsize=13)
        ax1.set_xticks(range(1, num_tasks+1))  # 强制显示所有刻度
        ax1.set_ylabel('路径成本 (m)', fontsize=13)
        ax1.grid(True)
        # ax1.legend(fontsize=8, loc='upper right')
        ax1.legend(fontsize=13, loc='upper center', ncol=len(algorithms))
        # ------------------------- 2. 路径耗时 -------------------------
        ax2 = fig.add_subplot(222)
        for algo in algorithms:
            turn_counts = algorithm_results[algo]['cal_path_time']
            ax2.plot(range(1, num_tasks+1), turn_counts,
                     marker='o',
                     label=algo )#.replace('_star', '*').replace('本文方法', '本文方法 ').replace("双向A*",'双向A*').replace('improve_A*','改进A*'))
        # 设置y轴的最大值以避免与图例冲突
        ax2.set_ylim(0, max(max(algorithm_results[algo]['cal_path_time'] for algo in algorithms)) * 1.2)
        ax2.set_title('路径耗时对比', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(1, num_tasks+1))  # 强制显示所有刻度
        ax2.set_xlabel('任务编号', fontsize=13)
        ax2.set_ylabel('路径耗时 (s)', fontsize=13)
        ax2.grid(True)
        # ax2.legend(fontsize=8, loc='upper right')
        ax2.legend(fontsize=13, loc='upper center', ncol=len(algorithms))
        # ------------------------- 3. 探索节点数 -------------------------
        ax3 = fig.add_subplot(223)
        for algo in algorithms:
            explored = algorithm_results[algo]['explored']
            ax3.plot(range(1, num_tasks+1), explored,
                     marker='o',
                     label=algo )#.replace('_star', '*').replace('本文方法', '本文方法 ').replace("双向A*",'双向A*').replace('improve_A*','改进A*'))
        # 设置y轴的最大值以避免与图例冲突
        ax3.set_ylim(0, max(max(algorithm_results[algo]['explored'] for algo in algorithms)) * 1.2)
        ax3.set_title('探索节点数对比', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(1, num_tasks+1))  # 强制显示所有刻度
        ax3.set_xlabel('任务编号', fontsize=13)
        ax3.set_ylabel('探索节点数 (个)', fontsize=13)
        ax3.grid(True)
        # ax3.legend(fontsize=8, loc='upper right')
        ax3.legend(fontsize=13, loc='upper center', ncol=len(algorithms))

        # ------------------------- 4. 搜索耗时 -------------------------
        ax4 = fig.add_subplot(224)
        for algo in algorithms:
            take_time = algorithm_results[algo]['take_time']
            ax4.plot(range(1, num_tasks+1), take_time,
                     marker='o',
                     label=algo)    #.replace('_star', '*').replace('本文方法', '本文方法 ').replace("双向A*",'双向A*').replace('improve_A*','改进A*'))
        ax4.set_ylim(0, max(max(algorithm_results[algo]['take_time'] for algo in algorithms)) * 1.2)
        ax4.set_title('搜索耗时对比', fontsize=13, fontweight='bold')
        ax4.set_xticks(range(1, num_tasks+1))  # 强制显示所有刻度
        ax4.set_xlabel('任务编号', fontsize=13)
        ax4.set_ylabel('耗时 (ms)', fontsize=13)
        ax4.grid(True)
        # ax4.legend(fontsize=10, loc='upper right')
        ax4.legend(fontsize=13, loc='upper center', ncol=len(algorithms))
        # 调整布局并保存
        plt.tight_layout()
        plt.show()
        fig.clf()  # 清空画布避免重叠


def main():
    # Example usage of the A* algorithm
    model = Model()
    combined_graph, floor1, floor2, floor3 = model.combined_graph, model.floor1, model.floor2, model.floor3
    planing =  Path_Planning(model)
    # print(planing.DStarLite_optimized2(combined_graph,1,1500))
    algorithm_results = planing.test_All_Nodes(combined_graph)
    planing.cal_optimity(algorithm_results)
    planing.plot_results2(algorithm_results)

if __name__ == '__main__':
    main()
