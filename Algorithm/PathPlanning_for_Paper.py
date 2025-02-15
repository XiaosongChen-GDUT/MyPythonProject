# This file contains the implementation of the A* algorithm for path planning
from networkx.algorithms.shortest_paths.weighted import _weight_function, _dijkstra
from Program.DataModel import Model
from heapq import heappop, heappush
from itertools import count  # count object for the heap堆的计数对象
from itertools import chain
import networkx as nx
import random
import math
import time


lanmarks_1 = [381, 369, 357,857,845,833,1324,1300,1629,1661]  # 关键十字路口地标点
lanmarks_2 = [642,674,1116,1148]    #1楼提升机接驳点
lanmarks_3 = [357,381,1300,1324]    #较分散的十字路口地标点
lanmarks_4 = [1,50,1708,1748]  # 地图的四个角落地标点

first_landmarks = [1,50,1708,1748]               # 1楼地标点
second_landmarks = [1749,1795,3528,3578]         # 2楼地标点
third_landmarks = [3579,3602,4443,4466]          # 3楼地标点
fist_connect_point = [642,1116,674,1148]        #1楼提升机接驳点
second_connect_point = [2374,2844,2406,2876]    #2楼接驳点
third_connect_point = [3899,4135]               #3楼接驳点
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
        for index in self.second_landmarks_index:
            self.second_landmarks[index] = {}
                # 计算当前节点到地标的距离
            distance = nx.shortest_path_length(self.second_floor,source=index,weight='weight')   #返回的是字典
            self.second_landmarks[index] = distance
        for index in self.third_landmarks_index:
            self.third_landmarks[index] = {}
                # 计算当前节点到地标的距离
            distance = nx.shortest_path_length(self.third_floor,source=index,weight='weight')   #返回的是字典
            self.third_landmarks[index] = distance
        #初始化换层节点
        self.cross_nodes = {1:fist_connect_point,2:second_connect_point, 3:third_connect_point}     #dict_values([[642, 1116, 674, 1148], [2374, 2844, 2406, 2876], [3899, 4135]])
        self.cross_nodes_list = list(chain(*self.cross_nodes.values()))     # [642, 1116, 674, 1148, 2374, 2844, 2406, 2876, 3899, 4135]
        # print(self.cross_nodes_list)


    '''take_time, path, explored, cost, turn_count'''
    def run(self, graph, source, target, algorithm_index=None, heuristic_index=None):
        try:
            weight='weight'
            if algorithm_index == 0:# Dijkstra算法
                return self.Dijkstra(graph, source, target, weight)
            elif algorithm_index == 1:# A*算法
                return self.A_star(graph, source, target, heuristic_index, weight)
            elif algorithm_index == 2:# ATL_star
                return self.ATL_star(graph, source, target, heuristic_index, lanmarks_4,weight)
            elif algorithm_index == 3:#改进的A*算法
                return self.improve_A_star(graph, source, target, heuristic_index, weight)
            elif algorithm_index == 4:# 方向A*
                # return self.direction_Astar(graph, source, target, heuristic_index, weight)
                return self.weight_ATL_star(graph, source, target, heuristic_index, lanmarks_4,weight)
            else:
                raise ValueError("class-> Path_Planning -> run : algorithm must be Dijkstra or A*!")
        except ValueError as ve:
            print(f"ValueError: {ve}")

    '''遍历完节点的实验'''
    def test_All_Nodes(self,graph,source,target,heuristic_index=2,weight='weight'):
        V = 1.2 #平均速度
        t = 4   #换向时间
        nodes = list(graph.nodes)
        # source = random.choice(nodes)  # 随机选择源节点
        source = 51 # 随机选择源节点
        nodes.remove(source)  # 去掉源节点
        # 随机选择200个目标节点
        # targets = random.sample(nodes, 10)
        sources = [51,348,925,1110,1620]
        targets = [26,323,50,1399,1727,1748]
        algorithm_results = {}  # 存储每个算法的结果
        # 初始化累计变量
        # 定义算法名称和对应的函数
        algorithms = {
            "Dijkstra": self.Dijkstra,
            "A_star": self.A_star,
            "improve_A_star": self.improve_A_star,
            # "ATL_star": self.ATL_star,
            "weight_ATL_star": self.weight_ATL_star
        }
        # for algorithm in algorithms:
        #     algorithm_results[algorithm] = {
        #         'take_time': 0,      # 累计耗时
        #         'explored': 0,       # 累计探索节点数
        #         'cost': 0,           # 累计成本
        #         'turn_count': 0      # 累计转向次数
        #     }
        for algorithm in algorithms:
            algorithm_results[algorithm] = {
                'take_time': [],      # 累计耗时
                'explored': [],       # 累计探索节点数
                'cost': [],           # 累计成本
                'turn_count': []      # 累计转向次数
            }
        for source in sources:
            for target in targets:
                # 遍历每个算法
                for algo_name, algo_func in algorithms.items():
                    # print(f"正在测试算法：{algo_name}，源节点：{source}，目标节点：{target}")
                    time_start = time.time()
                # 调用算法函数
                    if algo_name == "Dijkstra":
                        path, cost, explored = self.Dijkstra(graph, source, target, weight)
                    elif algo_name == "A_star":
                        path, cost, explored = self.A_star(graph, source, target, heuristic_index, weight)
                    elif algo_name == "improve_A_star":
                        path, cost, explored = self.improve_A_star(graph, source, target, heuristic_index, weight)
                    # elif algo_name == "ATL_star":
                    #     path, cost, explored = self.ATL_star(graph, source, target, heuristic_index, lanmarks_4, weight)
                    elif algo_name == "weight_ATL_star":
                        path, cost, explored = self.weight_ATL_star(graph, source, target, heuristic_index, lanmarks_4, weight)
                    else:
                        raise ValueError("class-> Path_Planning -> test_All_Nodes : algorithm must be Dijkstra or A*!")

                    take_time = round((time.time() - time_start) * 1000, 3)  # 耗时
                    turn_count = self.Turn_Count(graph, path)  # 转向次数

                    # 更新结果
                    # algorithm_results[algo_name]['take_time'] += take_time
                    # algorithm_results[algo_name]['explored'] += len(explored)
                    # algorithm_results[algo_name]['cost'] += cost
                    # algorithm_results[algo_name]['turn_count'] += turn_count# 更新结果
                    algorithm_results[algo_name]['take_time'].append(take_time)
                    algorithm_results[algo_name]['explored'].append(len(explored))
                    algorithm_results[algo_name]['cost'].append(cost)
                    algorithm_results[algo_name]['turn_count'].append(turn_count)

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
        algorithm_results["A_star"] = {
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
        # canvas.show_path_with_color(graph, path,"green",name='A_star')  # 显示路径
        # canvas.save_image(source, target, "A_star", "曼哈顿距离")
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
        # algorithm_results["improve_A_star"] = {
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
        # canvas.show_path_with_color(graph, path,color='orange',name='improve_A_star')  # 显示路径
        # canvas.save_image(source, target, "improve_A_star", "曼哈顿距离")
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

    ## 返回路径，路径长度，探索过的节点
    def A_star(self,graph, source, target, heuristic_index=None, weight='weight'):
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

        # 以画布中的绝对位置作为启发式预估参数
        if heuristic_index == 0:#欧几里得距离作为启发式函数:两点间的最短直线距离
            heuristic = lambda u, v: math.hypot(location[v][0] - location[u][0], location[v][1] - location[u][1])
            # heuristic = lambda u, v: (location[v][0] - location[u][0])**2 + (location[v][1] - location[u][1])**2
        elif heuristic_index == 1:#曼哈顿距离作为启发式函数:横纵坐标绝对值之和
            heuristic = lambda u, v:  abs(location[v][0] - location[u][0]) +  abs(location[v][1] - location[u][1])
        elif heuristic_index == 2:#切比雪夫距离作为启发式函数：各座标数值差的最大值。适用于走斜线
            heuristic = lambda u, v: max((location[v][0] - location[u][0]), abs(location[v][1] - location[u][1]))
        elif heuristic_index == 3:#平方欧几里得作为启发式函数：两点间的最短直线距离的平方
            heuristic = lambda u, v: (location[v][0] - location[u][0])**2 + (location[v][1] - location[u][1])**2
        else:
            heuristic = 0   # The default heuristic is h=0 - same as Dijkstra's algorithm

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
            current_layer = pos[curnode][2]
            # 遍历当前节点的邻居节点
            for neighbor, datas in G_succ[curnode].items():
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

    '''测试动态权重'''
    def weight_ATL_star(self,graph, source, target, heuristic_index=None,landmarks=None, weight='weight'):
        # alpha = 1  # g(n)实际成本函数的权重
        beta = 1.4 #1.2  # h(n)预估函数的权重
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

        push = heappush    # push function for the heap堆的Push函数
        pop = heappop     # pop function for the heap堆的Pop函数
        weight_function = _weight_function(graph, weight)  # weight function for the graph
        G_succ = graph._adj  # 用于存储图中每个节点的邻接信息{1: {2: {'weight': 1.5}}, 2: {1: {'weight': 1.5}, 3: {'weight': 1.5}}, 3: {2: {'weight': 1.5}}}
        c = count()  # 计数器，用于生成唯一的ID
        queue = [(0, next(c), source, 0 ,None)]  # 队列，元素为元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
        enqueued = {}  # 记录节点是否已经入队，记录到达节点的距离  和 节点到目标节点启发式评估值
        explored = {}  # 记录节点是否已经探索过
        # 动态权重计算.起点到目标节点的预估距离
        # D = self.ATL_heuristic(source, target,heuristic_index ,location, landmarks)
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
                    h = self.ATL_heuristic(neighbor, target,heuristic_index ,location, landmarks)
                # beta = 1 + h/D # 动态权重计算  从2-1进行调整
                h = h * beta # 动态权重计算
                enqueued[neighbor] = (ncost, h)
                push(queue, ( ncost + h, next(c), neighbor, ncost, curnode))
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

    def ATL_star(self,graph, source, target, heuristic_index=None,landmarks=None, weight='weight'):
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
                    h = self.ATL_heuristic(neighbor, target,heuristic_index ,location, landmarks)
                #方向成本
                h = h * beta # 权重
                enqueued[neighbor] = (ncost, h)
                push(queue, (alpha * ncost + h, next(c), neighbor, ncost, curnode))
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

    '''差分启发式函数'''
    def ATL_heuristic(self, neighbor, target, heuristic_index, location,landmarks = None):
        # # 以画布中的绝对位置作为启发式预估参数
        # if heuristic_index == 0:#欧几里得距离作为启发式函数:两点间的最短直线距离
        #     heuristic = math.hypot(location[target][0] - location[neighbor][0], location[target][1] - location[neighbor][1])
        # elif heuristic_index == 1:#曼哈顿距离作为启发式函数:横纵坐标绝对值之和
        #     heuristic = abs(location[target][0] - location[neighbor][0]) +  abs(location[target][1] - location[neighbor][1])
        # elif heuristic_index == 2:#切比雪夫距离作为启发式函数：各座标数值差的最大值。适用于走斜线
        #     heuristic = max((location[target][0] - location[neighbor][0]), abs(location[target][1] - location[neighbor][1]))
        # elif heuristic_index == 3:#平方欧几里得作为启发式函数：两点间的最短直线距离的平方
        #     heuristic = (location[target][0] - location[neighbor][0])**2 + (location[target][1] - location[neighbor][1])**2
        # else:
        #     heuristic = 0   # The default heuristic is h=0 - same as Dijkstra's algorithm
        # if landmarks is None:
        #     return heuristic
        # heuristic = 0
        heuristic = abs(location[target][0] - location[neighbor][0]) +  abs(location[target][1] - location[neighbor][1])
        # ATL距离启发式函数, A为起点，target为终点，landmarks为地标点集合，location为节点坐标集合
        for L in landmarks:
            # lower_bound = abs(self.first_landmarks[L][target] - self.first_landmarks[L][neighbor])
            lower_bound = self.first_landmarks[L][target] - self.first_landmarks[L][neighbor]
            if lower_bound - heuristic > 0.001 :
                # print("A:", A, "target:", Z ,"lower_bound:", lower_bound, "h:", heuristic)
                heuristic = lower_bound
        # print("manhattan_distance:",manhattan_distance,"heuristic:",heuristic)
        return heuristic


    '''引入权重1.5，拐点成本'''
    ## 返回路径，路径长度，探索过的节点
    def improve_A_star(self, graph, source, target, heuristic_index, weight='weight'):
        try:
            alpha = 1  # g(n)实际函数的权重
            beta = 1.5  # h(n)预估函数的权重
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
            if heuristic_index == 0:#欧几里得距离作为启发式函数:两点间的最短直线距离
                heuristic = lambda u, v: math.hypot(location[v][0] - location[u][0], location[v][1] - location[u][1])
            elif heuristic_index == 1:#曼哈顿距离作为启发式函数:横纵坐标绝对值之和
                heuristic = lambda u, v: abs(location[v][0] - location[u][0]) + abs(location[v][1] - location[u][1])
            elif heuristic_index == 2:#切比雪夫距离作为启发式函数：各座标数值差的最大值。适用于走斜线
                heuristic = lambda u, v: max((location[v][0] - location[u][0]), abs(location[v][1] - location[u][1]))
            elif heuristic_index == 3:#平方欧几里得作为启发式函数：两点间的最短直线距离的平方
                heuristic = lambda u, v: (location[v][0] - location[u][0])**2 + (location[v][1] - location[u][1])**2
            else:
                heuristic = None
            D = heuristic(source, target)
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
                    if cross_flag == True: # 换层标志
                        if current_node in source_cross_nodes_set: # 当前节点是起始层提升机节点
                            if pos[neighbor][2] == source_layer:  #邻居节点层数不是换层节点，跳过
                                continue
                        # 换层后也不能经过提升机节点
                        elif neighbor in target_cross_nodes_set:  # #当前节点不是起始层提升机节点，且邻居节点是目标层提升机节点，跳过
                            continue
                    #不换层
                    elif neighbor in source_cross_nodes_set:  #跳过起始层提升机节点
                        continue
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
                    h =  beta * h
                    enqueued[neighbor] = (ncost, h)      # 记录起点到邻居节点的距离ncost和邻居节点到目标节点的启发式评估值h
                    push(queue, (alpha * ncost + h, next(c), neighbor, ncost, current_node))  # 6.入队，并更新队列
                    # f = ncost + h 为A*算法的评估值，用于判断节点的优先级，使得算法更加贪婪，更加关注距离短的节点
            raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))  # 6.找不到路径，抛出异常
        except Exception as e:
            print(f"improve_A_star - Exception: {e}")

    # ##方向A*算法 返回路径，路径长度，探索过的节点
    # def direction_Astar(self, graph, source, target, heuristic_index, weight='weight'):
    #     try:
    #         alpha = 1  # g(n)实际函数的权重
    #         beta = 1.5  # h(n)预估函数的权重
    #         location = nx.get_node_attributes(graph, 'location')
    #         # 以画布中的绝对位置作为启发式预估函数参数
    #         if heuristic_index == 0:#欧几里得距离作为启发式函数:两点间的最短直线距离
    #             heuristic = lambda u, v: math.hypot(location[v][0] - location[u][0], location[v][1] - location[u][1])
    #         elif heuristic_index == 1:#曼哈顿距离作为启发式函数:横纵坐标绝对值之和
    #             heuristic = lambda u, v: abs(location[v][0] - location[u][0]) + abs(location[v][1] - location[u][1])
    #         elif heuristic_index == 2:#切比雪夫距离作为启发式函数：各座标数值差的最大值。适用于走斜线
    #             heuristic = lambda u, v: max((location[v][0] - location[u][0]), abs(location[v][1] - location[u][1]))
    #         elif heuristic_index == 3:#平方欧几里得作为启发式函数：两点间的最短直线距离的平方
    #             heuristic = lambda u, v: (location[v][0] - location[u][0])**2 + (location[v][1] - location[u][1])**2
    #         else:
    #             heuristic = None
    #
    #         push = heappush    # push function for the heap堆的Push函数
    #         pop = heappop     # pop function for the heap堆的Pop函数
    #         weight_function = _weight_function(graph, weight)  # weight function for the graph
    #         G_succ = graph._adj  # 用于存储图中每个节点的邻接信息{1: {2: {'weight': 1.5}}, 2: {1: {'weight': 1.5}, 3: {'weight': 1.5}}, 3: {2: {'weight': 1.5}}}
    #         c = count()  # 计数器，用于生成唯一的ID
    #         queue = [(0, next(c), source, 0 ,None)]  # 队列，元素为元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
    #         enqueued = {}  # 记录节点是否已经入队，记录到达节点的距离  和 节点到目标节点启发式评估值
    #         explored = {}  # 记录节点是否已经探索过,记录父节点
    #         while queue:
    #             #     # 弹出队列中代价最小的元素
    #             #     # 元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
    #             _, _, current_node, dist_current_node,parent = pop(queue)
    #             if current_node == target:     # 1.找到目标节点
    #                 path = [current_node]
    #                 node = parent
    #                 while node is not None:
    #                     path.append(node)           # 反向构建路径
    #                     node = explored[node]       # 回溯父节点
    #                 path.reverse()                  # 反转路径
    #                 return path, nx.path_weight(graph, path, weight), explored  # 返回路径，路径长度，探索过的节点
    #             if current_node in explored:        # 2.已经探索过，跳过
    #                 if explored[current_node] is None:  # 已经探索过，但父节点为None，说明是源节点，跳过
    #                     continue
    #                 qcost, h = enqueued[current_node]   # 取出到当前节点的距离和启发式评估值
    #                 if qcost < dist_current_node:       # 已经探索过，且距离更短所以方案更优，跳过本次探索
    #                     continue
    #             explored[current_node] = parent         # 记录父节点
    #             for neighbor, w in G_succ[current_node].items():   # 3.遍历当前节点的邻居节点
    #                 cost = weight_function(current_node, neighbor, w)  # 计算当前节点到邻居节点的距离
    #                 # print(f"current_node:",current_node,"neighbor:",neighbor,"w:",w,"cost:",cost)
    #                 #current_node: 1469 neighbor: 1468 w: {'weight': 2.02} cost: 2.02
    #                 if cost is None:#不可达的节点，跳过
    #                     continue
    #                 # 引入拐点成本
    #                 turn_cost = self.Turn_Cost(location,parent,neighbor,target)
    #                 #方向成本
    #                 direction_cost = self.Direction_Cost(location,source,parent,current_node,neighbor,target)
    #                 ncost =  dist_current_node + cost + turn_cost + direction_cost      # 计算起点到邻居节点的总代价：起点到当前节点的距离 + 当前节点到邻居节点的距离
    #                 # ncost = dist_current_node + cost  # 计算起点到邻居节点的总代价：起点到当前节点的距离 + 当前节点到邻居节点的距离
    #                 if neighbor  in enqueued:         # 4.判断邻居节点是否已经入队
    #                     qcost, h = enqueued[neighbor]  # 取出邻居节点的距离和启发式评估值
    #                     if qcost <= ncost:             # 邻居节点已经入队，且距离更短，跳过本次邻居节点的探索
    #                         continue
    #                 else:# 5.邻居节点没有入队
    #                     h = heuristic(neighbor, target)  # 计算邻居节点到目标节点的启发式评估值
    #                 enqueued[neighbor] = (ncost, h)      # 记录起点到邻居节点的距离ncost和邻居节点到目标节点的启发式评估值h
    #                 push(queue, (alpha * ncost + beta * h, next(c), neighbor, ncost, current_node))  # 6.入队，并更新队列
    #                 # f = ncost + h 为A*算法的评估值，用于判断节点的优先级，使得算法更加贪婪，更加关注距离短的节点
    #         raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))  # 6.找不到路径，抛出异常
    #     except Exception as e:
    #         print(f"improve_A_star - Exception: {e}")

    ## 返回路径，路径长度，探索过的节点

    def Dijkstra(self,graph, source, target,  weight='weight'):
        # 调用networkx的dijkstra算法
        if source not in graph or target not in graph:
            graph = self.TopoGraph
        elif source == target:
            return 0, 0, 0
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
                if cross_flag == True: # 换层标志
                    if current_node in source_cross_nodes_set: # 当前节点是起始层提升机节点
                        if pos[neighbor][2] == source_layer:  #邻居节点层数不是换层节点，跳过
                            continue
                    # 换层后也不能经过提升机节点
                    elif neighbor in target_cross_nodes_set:  # #当前节点不是起始层提升机节点，且邻居节点是目标层提升机节点，跳过
                        continue
                    #不换层
                elif neighbor in source_cross_nodes_set:  #跳过起始层提升机节点
                    continue
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


    # 判断是否出现垂直
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
            # (x1, y1) = (round(current_pos[0] - parent_pos[0], 2), round(current_pos[1] - parent_pos[1], 2))  # 当前节点到父节点的向量
            # (x2, y2) = (round(next_pos[0] - current_pos[0], 2), round(next_pos[1] - current_pos[1], 2))  # 当前节点到下一个节点的向量
            # result = int(x1 * y2 - x2 * y1)  # 计算叉乘，判断是否在同一直线上
            # if result != 0:  # 判断是否在同一直线上,不等于0，说明不在同一直线上，发生转向
            #     turn_count += 1
            #     print(f"发生转向: x1={x1},y1={y1},x2={x2},y2={y2},result={result}")
            # 计算在 x 和 y 轴上的偏差
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

    # '''返回停滞节点列表：拐点、提升机换层节点'''
    # def get_stop_nodes(self,graph,path):
    #     stop_nodes = []
    #     location = nx.get_node_attributes(graph, 'location')
    #     threshold = 1.0
    #     # 采用滑动窗口进行路径特征分析
    #     for i in range(1, len(path) - 1):
    #         parent = path[i - 1]      # 前一个节点
    #         current_node = path[i]    # 当前节点
    #         next_node = path[i + 1]   # 下一个节点
    #         parent_pos = location[parent]
    #         next_pos = location[next_node]
    #         delta_x = abs(parent_pos[0] - next_pos[0])  # x轴偏差
    #         delta_y = abs(parent_pos[1] - next_pos[1])  # y轴偏差
    #         # 判断是否在 x 和 y 轴上都超过阈值
    #         if delta_x > threshold and delta_y > threshold:
    #             stop_nodes.append(current_node)
    #         # 判断是否是提升机换层节点
    #         elif current_node in self.cross_nodes_list :
    #             print("提升机换层节点：",current_node)
    #             stop_nodes.append(current_node)
    #     return stop_nodes



    def Direction_Cost(self,location,source,parent,current,next_node,target):
        # direction_cost = 1.5
        if parent is None :return 0
        parent_to_next = (location[next_node][0] - location[parent][0], location[next_node][1] - location[parent][1])
        source_to_target = (location[target][0] - location[source][0], location[target][1] - location[source][1])
        current_to_target = (location[target][0] - location[current][0], location[target][1] - location[current][1])
        current_to_next = (location[next_node][0] - location[current][0], location[next_node][1] - location[current][1])
        #计算current_to_next 和 current_to_target 的点积，夹角小，余弦值大；夹角大，余弦值小，0是垂直分界点
        dot_product = current_to_next[0] * current_to_target[0] + current_to_next[1] * current_to_target[1]
        # dot_product = current_to_target[0] * parent_to_next[0] + current_to_target[1] * parent_to_next[1]   #parent指向next向量 与 当前点指向终点的向量，节点数略少，但耗时较高
        # dot_product = source_to_target[0] * parent_to_next[0] + source_to_target[1] * parent_to_next[1]   #parent指向next向量 与 起点点指向终点的向量，节点数增多，耗时高
        # dot_product = current_to_next[0] * source_to_target[0] + current_to_next[1] * source_to_target[1] #不可行
        if dot_product > 0:
            return 0    #dot_product
        else:   #<=0
            return  -dot_product #direction_cost 值越大，反而拓展的节点数会更多！
        # return -dot_product


    # 分析路径，计算路径长度和转向次数
    def Analyze_Path(self,graph, source, target, algorithm_index=None, heuristic_index=None):
        # turn_count = 0  # 初始化转向计数
        start_time = time.time()
        path, cost, explored = self.run(graph, source, target, algorithm_index, heuristic_index)
        take_time = round( (time.time() - start_time)*1000,2)  # 计算运行时间
        turn_count = self.Turn_Count(graph, path)  # 计算转向次数
        return take_time, path, explored, cost, turn_count



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
            # print("PATH:",path," 提升机换层节点列表：",elevator_nodes)
            ele_length = round(nx.path_weight(graph,elevator_nodes, 'weight'),3)
            ele_time = calculate_vertical_time(ele_length)
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


def main():
    # Example usage of the A* algorithm
    model = Model()
    combined_graph, floor1, floor2, floor3 = model.combined_graph, model.floor1, model.floor2, model.floor3
    planing =  Path_Planning(model)
    # path, cost ,explored = planing.improve_A_star(floor1, 1110, 3019,1 )
    # nx_shortest_path = nx.shortest_path(combined_graph,source=1110,target=4436,weight='weight')
    # print("nx_shortest_path 最短路径：",nx_shortest_path)
    path, cost ,explored = planing.A_star(floor1, 636, 684,1 )
    print("A*算法最短路径：",path,"，A*算法路径长度：",cost
          ," 转向次数：",planing.Turn_Count(combined_graph,path))
    # print("转向节点列表：",planing.get_special_nodes(combined_graph,path))

    # nx.shortest_path(combined_graph, source=1110, target=3019, weight='weight')
    # my_cost = nx.path_weight(floor1,a_star_path,'weight')

    planing.cal_path_time(combined_graph,path)

if __name__ == '__main__':
    main()
