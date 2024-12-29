# This file contains the implementation of the A* algorithm for path planning
from networkx.algorithms.shortest_paths.weighted import _weight_function, _dijkstra
import networkx as nx
from Program.DataModel import Model
from heapq import heappop, heappush
from itertools import count  # count object for the heap堆的计数对象
import math
import time


lanmarks_1 = [381, 369, 357,857,845,833,1324,1300,1629,1661]  # 关键十字路口地标点
lanmarks_2 = [642,674,1116,1148]    #1楼提升机接驳点
lanmarks_3 = [357,381,1300,1324]    #较分散的十字路口地标点
lanmarks_4 = [1,50,1708,1748]  # 地图的四个角落地标点
class Path_Planning:
    def __init__(self,canvasList):
        self.canvasList = canvasList
        self.first_canvas = canvasList[0]
        self.second_canvas = canvasList[1]
        self.third_canvas = canvasList[2]
        #地图
        self.first_floor = canvasList[0].floor
        self.second_floor = canvasList[1].floor
        self.third_floor = canvasList[2].floor

        self.first_landmarks_index = lanmarks_1+lanmarks_2 + lanmarks_3 + lanmarks_4 # 组合所有地标点的索引
        # first_location = nx.get_node_attributes(self.first_floor, 'location')
        # 地标点的坐标
        # self.first_landmarks_location = {node: first_location[node] for node in self.first_landmarks_index}
        self.first_landmarks = {}  # 初始化字典
        # # 计算每个地标点到每个节点的距离
        for index in self.first_landmarks_index:
            self.first_landmarks[index] = {}
                # 计算当前节点到地标的距离
                # distance = math.hypot(loc[0] - self.first_landmarks_location[index][0], loc[1] - self.first_landmarks_location[index][1])
                # distance = abs(loc[0] - self.first_landmarks_location[index][0])+ abs(loc[1] - self.first_landmarks_location[index][1])
            distance = nx.shortest_path_length(self.first_floor,source=index,weight='weight')   #返回的是字典
                # 将结果存储在字典中，使用 (node, id) 作为键来唯一标识
                # self.first_landmarks[index][node] = distance
            self.first_landmarks[index] = distance


    def run(self, graph, source, target, algorithm_index=None, heuristic_index=None):
        try:
            weight='weight'
            if algorithm_index == 0:# Dijkstra算法
                return self.Dijkstra(graph, source, target, weight)
            elif algorithm_index == 1:# A*算法
                return self.A_star(graph, source, target, heuristic_index, weight)
            elif algorithm_index == 2:# ATL_star
                return self.ATL_star(graph, source, target, heuristic_index, lanmarks_3,weight)
            elif algorithm_index == 3:#改进的A*算法
                return self.improve_A_star(graph, source, target, heuristic_index, weight)
            elif algorithm_index == 4:# 方向A*
                return self.direction_Astar(graph, source, target, heuristic_index, weight)
            else:
                raise ValueError("class-> Path_Planning -> run : algorithm must be Dijkstra or A*!")
        except ValueError as ve:
            print(f"ValueError: {ve}")

    """对比测试不同的地标点的数量、位置"""
    def test_ATL_star(self, graph, source, target, heuristic_index=2, weight='weight'):
        canvas =  graph.graph['canvas']
        targets = [26,50,323,1399,1727,1748]  # 测试目标点
        algorithm_results = {}  # 存储每个算法的结果

        start_time = time.time()
        path, cost, explored=self.ATL_star(graph, source, target, heuristic_index, lanmarks_1,weight)
        take_time = time.time() - start_time
        turn_count = self.Turn_Count(graph,path)
        # 存储结果
        algorithm_results["lanmarks_1"] = {
            'take_time': take_time,
            'explored': len(explored),
            'cost': cost,
            'turn_count': turn_count
        }
        canvas.show_visited_process(graph, explored)    #显示探索过的节点
        canvas.show_path_with_color(graph, path,color='pink',name='lanmarks_1')  # 显示路径
        canvas.save_image(source, target, "ATL_star", "lanmarks_1")
        canvas.reset_canvas()  # 重置画布

        start_time = time.time()
        path, cost, explored=self.ATL_star(graph, source, target, heuristic_index, lanmarks_2,weight)
        take_time = time.time() - start_time
        turn_count = self.Turn_Count(graph,path)
        # 存储结果
        algorithm_results["lanmarks_2"] = {
            'take_time': take_time,
            'explored': len(explored),
            'cost': cost,
            'turn_count': turn_count
        }
        canvas.show_visited_process(graph, explored)    #显示探索过的节点
        canvas.show_path_with_color(graph, path,color='brown',name='lanmarks_2')  # 显示路径
        canvas.save_image(source, target, "ATL_star", "lanmarks_2")
        canvas.reset_canvas()  # 重置画布

        start_time = time.time()
        path, cost ,explored=self.ATL_star(graph, source, target, heuristic_index, lanmarks_3,weight)
        take_time = time.time() - start_time
        turn_count = self.Turn_Count(graph,path)
        # 存储结果
        algorithm_results["lanmarks_3"] = {
            'take_time': take_time,
            'explored': len(explored),
            'cost': cost,
            'turn_count': turn_count
        }
        canvas.show_visited_process(graph, explored)    #显示探索过的节点
        canvas.show_path_with_color(graph, path,color='purple',name='lanmarks_3')  # 显示路径
        canvas.save_image(source, target, "ATL_star", "lanmarks_3")
        canvas.reset_canvas()  # 重置画布

        start_time = time.time()
        path, cost ,explored=self.ATL_star(graph, source, target, heuristic_index, lanmarks_4,weight)
        take_time = time.time() - start_time
        turn_count = self.Turn_Count(graph,path)
        # 存储结果
        algorithm_results["lanmarks_4"] = {
            'take_time': take_time,
            'explored': len(explored),
            'cost': cost,
            'turn_count': turn_count
        }
        canvas.show_visited_process(graph, explored)    #显示探索过的节点
        canvas.show_path_with_color(graph, path,color='black',name='lanmarks_4')  # 显示路径
        canvas.save_image(source, target, "ATL_star", "lanmarks_4")
        canvas.reset_canvas()  # 重置画布

        start_time = time.time()
        path, cost, explored=self.A_star(graph, source, target, heuristic_index,weight)
        take_time = time.time() - start_time
        turn_count = self.Turn_Count(graph,path)
        # 存储结果
        algorithm_results["A_star"] = {
            'take_time': take_time,
            'explored': len(explored),
            'cost': cost,
            'turn_count': turn_count
        }
        canvas.show_visited_process(graph, explored)    #显示探索过的节点
        canvas.show_path_with_color(graph, path,"green",name='A_star')  # 显示路径
        canvas.save_image(source, target, "A_star", "曼哈顿距离")
        canvas.reset_canvas()  # 重置画布

        start_time = time.time()
        path, cost, explored=self.improve_A_star(graph, source, target, heuristic_index,weight)
        take_time = time.time() - start_time
        turn_count = self.Turn_Count(graph,path)
        # 存储结果
        algorithm_results["improve_A_star"] = {
            'take_time': take_time,
            'explored': len(explored),
            'cost': cost,
            'turn_count': turn_count
        }
        canvas.show_visited_process(graph, explored)    #显示探索过的节点
        canvas.show_path_with_color(graph, path,color='orange',name='improve_A_star')  # 显示路径
        canvas.save_image(source, target, "improve_A_star", "曼哈顿距离")
        canvas.reset_canvas()  # 重置画布

        start_time = time.time()
        path, cost, explored=self.Dijkstra(graph, source, target,weight)
        take_time = time.time() - start_time
        turn_count = self.Turn_Count(graph,path)
        # 存储结果
        algorithm_results["Dijkstra"] = {
            'take_time': take_time,
            'explored': len(explored),
            'cost': cost,
            'turn_count': turn_count
        }
        canvas.show_visited_process(graph, explored)    #显示探索过的节点
        canvas.show_path_with_color(graph, path,color='blue',name='Dijkstra算法')  # 显示路径
        canvas.save_image(source, target, "dijkstra算法", "")
        canvas.reset_canvas()  # 重置画布
        return algorithm_results



    ## 返回路径，路径长度，探索过的节点
    def A_star(self,graph, source, target, heuristic_index=None, weight='weight'):

        location = nx.get_node_attributes(graph, 'location')
        # 以画布中的绝对位置作为启发式预估参数
        if heuristic_index == 0:#欧几里得距离作为启发式函数:两点间的最短直线距离
            heuristic = lambda u, v: math.hypot(location[v][0] - location[u][0], location[v][1] - location[u][1])
        elif heuristic_index == 1:#曼哈顿距离作为启发式函数:横纵坐标绝对值之和
            heuristic = lambda u, v:  abs(location[v][0] - location[u][0]) +  abs(location[v][1] - location[u][1])
        elif heuristic_index == 2:#切比雪夫距离作为启发式函数：各座标数值差的最大值。适用于走斜线
            heuristic = lambda u, v: max((location[v][0] - location[u][0]), abs(location[v][1] - location[u][1]))
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
                return path, nx.path_weight(graph, path, weight) , explored
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

    def ATL_star(self,graph, source, target, heuristic_index=None,landmarks=None, weight='weight'):
        location = nx.get_node_attributes(graph, 'location')
        alpha = 1  # g(n)实际成本函数的权重
        beta = 1.2  # h(n)预估函数的权重
        # path = nx.astar_path(graph, source=source, target=target, heuristic=heuristic, weight=weight)
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
                return path, nx.path_weight(graph, path, weight) , explored
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
                # direction_cost = self.Direction_Cost(location,source,parent,curnode,neighbor,target)
                # h = h + direction_cost  #加上方向成本后，遍历的节点会未加方向成本更多
                enqueued[neighbor] = (ncost, h)
                # print("A:", curnode, "Z:", neighbor,"COST:", cost, "lower_bound:", h)
                push(queue, (alpha * ncost + beta * h, next(c), neighbor, ncost, curnode))
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

    '''差分启发式函数'''
    def ATL_heuristic(self, A, Z, heuristic_index, location,landmarks = None):
        # # 以画布中的绝对位置作为启发式预估参数
        if heuristic_index == 0:#欧几里得距离作为启发式函数:两点间的最短直线距离
            heuristic = math.hypot(location[Z][0] - location[A][0], location[Z][1] - location[A][1])
        elif heuristic_index == 1:#曼哈顿距离作为启发式函数:横纵坐标绝对值之和
            heuristic = abs(location[Z][0] - location[A][0]) +  abs(location[Z][1] - location[A][1])
        elif heuristic_index == 2:#切比雪夫距离作为启发式函数：各座标数值差的最大值。适用于走斜线
            heuristic = max((location[Z][0] - location[A][0]), abs(location[Z][1] - location[A][1]))
        else:
            heuristic = 0   # The default heuristic is h=0 - same as Dijkstra's algorithm
        # ATL距离启发式函数, A为起点，Z为终点，landmarks为地标点集合，location为节点坐标集合
        for L in landmarks:
            # lower_bound = abs(self.first_landmarks[L][Z] - self.first_landmarks[L][A])
            lower_bound = self.first_landmarks[L][Z] - self.first_landmarks[L][A]
            if lower_bound - heuristic > 0.001 :
                # print("A:", A, "Z:", Z ,"lower_bound:", lower_bound, "h:", heuristic)
                heuristic = lower_bound
        return heuristic



    ## 返回路径，路径长度，探索过的节点
    def improve_A_star(self, graph, source, target, heuristic_index, weight='weight'):
        try:
            alpha = 1  # g(n)实际函数的权重
            beta = 1.5  # h(n)预估函数的权重
            location = nx.get_node_attributes(graph, 'location')
            # 以画布中的绝对位置作为启发式预估函数参数
            if heuristic_index == 0:#欧几里得距离作为启发式函数:两点间的最短直线距离
                heuristic = lambda u, v: math.hypot(location[v][0] - location[u][0], location[v][1] - location[u][1])
            elif heuristic_index == 1:#曼哈顿距离作为启发式函数:横纵坐标绝对值之和
                heuristic = lambda u, v: abs(location[v][0] - location[u][0]) + abs(location[v][1] - location[u][1])
            elif heuristic_index == 2:#切比雪夫距离作为启发式函数：各座标数值差的最大值。适用于走斜线
                heuristic = lambda u, v: max((location[v][0] - location[u][0]), abs(location[v][1] - location[u][1]))
            else:
                heuristic = None

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
                    return path, nx.path_weight(graph, path, weight), explored  # 返回路径，路径长度，探索过的节点
                if current_node in explored:        # 2.已经探索过，跳过
                    if explored[current_node] is None:  # 已经探索过，但父节点为None，说明是源节点，跳过
                        continue
                    qcost, h = enqueued[current_node]   # 取出到当前节点的距离和启发式评估值
                    if qcost < dist_current_node:       # 已经探索过，且距离更短所以方案更优，跳过本次探索
                        continue
                explored[current_node] = parent         # 记录父节点
                for neighbor, w in G_succ[current_node].items():   # 3.遍历当前节点的邻居节点
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
                    enqueued[neighbor] = (ncost, h)      # 记录起点到邻居节点的距离ncost和邻居节点到目标节点的启发式评估值h
                    push(queue, (alpha * ncost + beta * h, next(c), neighbor, ncost, current_node))  # 6.入队，并更新队列
                    # f = ncost + h 为A*算法的评估值，用于判断节点的优先级，使得算法更加贪婪，更加关注距离短的节点
            raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))  # 6.找不到路径，抛出异常
        except Exception as e:
            print(f"improve_A_star - Exception: {e}")

    ## 返回路径，路径长度，探索过的节点
    def direction_Astar(self, graph, source, target, heuristic_index, weight='weight'):
        try:
            alpha = 1  # g(n)实际函数的权重
            beta = 1  # h(n)预估函数的权重
            location = nx.get_node_attributes(graph, 'location')
            # 以画布中的绝对位置作为启发式预估函数参数
            if heuristic_index == 0:#欧几里得距离作为启发式函数:两点间的最短直线距离
                heuristic = lambda u, v: math.hypot(location[v][0] - location[u][0], location[v][1] - location[u][1])
            elif heuristic_index == 1:#曼哈顿距离作为启发式函数:横纵坐标绝对值之和
                heuristic = lambda u, v: abs(location[v][0] - location[u][0]) + abs(location[v][1] - location[u][1])
            elif heuristic_index == 2:#切比雪夫距离作为启发式函数：各座标数值差的最大值。适用于走斜线
                heuristic = lambda u, v: max((location[v][0] - location[u][0]), abs(location[v][1] - location[u][1]))
            else:
                heuristic = None

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
                    return path, nx.path_weight(graph, path, weight), explored  # 返回路径，路径长度，探索过的节点
                if current_node in explored:        # 2.已经探索过，跳过
                    if explored[current_node] is None:  # 已经探索过，但父节点为None，说明是源节点，跳过
                        continue
                    qcost, h = enqueued[current_node]   # 取出到当前节点的距离和启发式评估值
                    if qcost < dist_current_node:       # 已经探索过，且距离更短所以方案更优，跳过本次探索
                        continue
                explored[current_node] = parent         # 记录父节点
                for neighbor, w in G_succ[current_node].items():   # 3.遍历当前节点的邻居节点
                    cost = weight_function(current_node, neighbor, w)  # 计算当前节点到邻居节点的距离
                    # print(f"current_node:",current_node,"neighbor:",neighbor,"w:",w,"cost:",cost)
                    #current_node: 1469 neighbor: 1468 w: {'weight': 2.02} cost: 2.02
                    if cost is None:#不可达的节点，跳过
                        continue
                    # 引入拐点成本
                    turn_cost = self.Turn_Cost(location,parent,neighbor,target)
                    #方向成本
                    direction_cost = self.Direction_Cost(location,source,parent,current_node,neighbor,target)
                    ncost =  dist_current_node + cost + turn_cost + direction_cost      # 计算起点到邻居节点的总代价：起点到当前节点的距离 + 当前节点到邻居节点的距离
                    # ncost = dist_current_node + cost  # 计算起点到邻居节点的总代价：起点到当前节点的距离 + 当前节点到邻居节点的距离
                    if neighbor  in enqueued:         # 4.判断邻居节点是否已经入队
                        qcost, h = enqueued[neighbor]  # 取出邻居节点的距离和启发式评估值
                        if qcost <= ncost:             # 邻居节点已经入队，且距离更短，跳过本次邻居节点的探索
                            continue
                    else:# 5.邻居节点没有入队
                        h = heuristic(neighbor, target)  # 计算邻居节点到目标节点的启发式评估值
                    enqueued[neighbor] = (ncost, h)      # 记录起点到邻居节点的距离ncost和邻居节点到目标节点的启发式评估值h
                    push(queue, (alpha * ncost + beta * h, next(c), neighbor, ncost, current_node))  # 6.入队，并更新队列
                    # f = ncost + h 为A*算法的评估值，用于判断节点的优先级，使得算法更加贪婪，更加关注距离短的节点
            raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))  # 6.找不到路径，抛出异常
        except Exception as e:
            print(f"improve_A_star - Exception: {e}")

    ## 返回路径，路径长度，探索过的节点
    def Dijkstra(self,graph, source, target,  weight='weight'):
        # 调用networkx的dijkstra算法
        # path = nx.dijkstra_path(graph, source=source, target=target, weight=weight)
        # cost = nx.path_weight(graph, path, weight)
        cutoff = None
        pred = None
        paths ={source: [source]}          #保存路径，用于存储从源节点到每个其他节点的路径列表。字典的键是节点标签，值是从源节点到该节点的路径（一个节点序列）。

        if source not in graph:
            raise nx.NodeNotFound(f"Node {source} not found in graph")
        if target not in graph:
            raise nx.NodeNotFound(f"Node {target} not found in graph")
        if source == target:
            return 0, 0, 0
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
            (d, _, v) = pop(fringe)      # 当 fringe 不为空时，弹出最小距离的节点 (d, _, v)。
            if v in dist:
                continue  # already searched this node.
            dist[v] = d
            if v == target:
                return paths[v], dist[v], dist.keys()            # 找到目标节点，返回路径长度和路径
                # break
            for u, e in G_succ[v].items():      # 遍历当前节点的邻居节点
                cost = weight_function(v, u, e)          # 计算当前节点到邻居节点的距离
                if cost is None:                # 无法到达的节点，跳过
                    continue
                vu_dist = dist[v] + cost        # 到邻居节点的总代价：起点到当前节点的距离 + 当前节点到邻居节点的距离
                if cutoff is not None:          # 距离超过限制阈值，则跳过本次邻居节点的探索
                    if vu_dist > cutoff:
                        continue
                if u in dist:                   # 已经探索过，且距离更短，跳过本次邻居节点的探索
                    u_dist = dist[u]
                    if vu_dist < u_dist:
                        #如果新路径长度更短，则说明存在矛盾路径，可能是由于图中存在负权重边或环（即在最短路径算法中不应出现的情形）。
                        raise ValueError("Contradictory paths found:", "negative weights?")
                    #当前路径和之前发现的路径长度是否相同
                    elif pred is not None and vu_dist == u_dist:
                        pred[u].append(v)       #记录多条路径可能导致同样的最短距离。这在某些情况下是有用的，如寻找多条路径的场景。
                elif u not in seen or vu_dist < seen[u]:    # 邻居节点没有入队，或距离更短，则入队
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    if paths is not None:
                        paths[u] = paths[v] + [u]
                    if pred is not None:
                        pred[u] = [v]
                elif vu_dist == seen[u]:
                    if pred is not None:
                        pred[u].append(v)
        raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))  # 6.找不到路径，抛出异常
        # The optional predecessor and path dictionaries can be accessed
        # by the caller via the pred and paths objects passed as arguments.


    # 判断是否出现垂直
    def Turn_Cost(self,location,parent,next_node,target):
        turn_cost = 1.5  # 普通节点转向代价  权重在1的时候，拐点数量还是较多
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
        turn_count = 0  # 初始化转向计数
        start_time = time.time()
        path, cost, explored = self.run(graph, source, target, algorithm_index, heuristic_index)
        take_time = time.time() - start_time
        if cost == 0 or path is None:
            return take_time, path, cost, turn_count  # 路径不存在，返回0
        turn_count = self.Turn_Count(graph, path)  # 计算转向次数

        return take_time, path, explored, cost, turn_count

    def Turn_Count(self,graph, path):
        turn_count = 0
        location = nx.get_node_attributes(graph, 'location')
        # cost = nx.path_weight(graph, path, 'weight')
        # 遍历路径列表，从第一个点开始，直到倒数第三个点
        for i in range(1, len(path) - 1):
            parent = path[i - 1]      # 前一个节点
            current_node = path[i]    # 当前节点
            next_node = path[i + 1]   # 下一个节点
            parent_pos = location[parent]
            current_pos = location[current_node]
            next_pos = location[next_node]
            # 计算在 x 和 y 轴上的偏差
            delta_x = abs(parent_pos[0] - next_pos[0])  # x轴偏差
            delta_y = abs(parent_pos[1] - next_pos[1])  # y轴偏差

            # 设定一个阈值，假设阈值为某个常量，例如 1.0
            threshold = 1.0

            # 判断是否在 x 和 y 轴上都超过阈值
            if delta_x > threshold and delta_y > threshold:
                # print(f"发生转向: parent_pos={parent_pos}, next_pos={next_pos},delta_x={delta_x},delta_y={delta_y}")
                turn_count += 1  # 如果转向，计数加
        return turn_count


def main():
    # Example usage of the A* algorithm
    model = Model()
    combined_graph, floor1, floor2, floor3 = model.combined_graph, model.floor1, model.floor2, model.floor3
    planing =  Path_Planning()
    path, cost = planing.A_star(floor1, 1, 180, weight='weight')
    print("A*算法最短路径：",path,"，A*算法路径长度：",cost ,"，A*算法path_weight：",nx.path_weight(floor1,path,'weight'))
    dijkstra_path, dijkstra_cost = planing.Dijkstra()
    print("dijkstra算法最短路径：",dijkstra_path,"，dijkstra算法路径长度：",dijkstra_cost)
    # path = nx.shortest_path(floor1,source=1,target=10,weight='weight',method='dijkstra')
    # cost = nx.path_weight(floor1,path,'weight')
    # print("dijkstra最短路径：",path,"，dijkstra路径长度：",cost)
    # a_star_path = nx.astar_path(floor1,source=1,target=10,weight='weight')
    # my_cost = nx.path_weight(floor1,a_star_path,'weight')
    # print("a_star_path路径长度：",my_cost)


if __name__ == '__main__':
    main()
