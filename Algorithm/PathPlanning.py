# This file contains the implementation of the A* algorithm for path planning
from networkx.algorithms.shortest_paths.weighted import _weight_function
import networkx as nx
from Program.DataModel import Model
from heapq import heappop, heappush
from itertools import count  # count object for the heap堆的计数对象
import math
import time



class Path_Planning:
    def __init__(self):
        pass


    def run(self, graph, source, target, algorithm_index=None, heuristic_index=None):
        try:
            weight='weight'
            # if graph is None:
            #     raise ValueError("class-> Path_Planning -> run : graph cannot be None!")
            # if source is None:
            #     raise ValueError("class-> Path_Planning -> run : source cannot be None!")
            # if target is None:
            #     raise ValueError("class-> Path_Planning -> run : target cannot be None!")
            if algorithm_index == 0:# Dijkstra算法
                return self.Dijkstra(graph, source, target, weight)
            elif algorithm_index == 1:# A*算法
                return self.A_star(graph, source, target, heuristic_index, weight)
            elif algorithm_index == 2:# 双向Dijkstra
                return self.BiDirectional_Dijkstra(graph, source, target, weight)
            elif algorithm_index == 3:# 贝尔曼-福特
                return self.Bellman_Ford(graph, source, target, weight)
            elif algorithm_index == 4:#改进的A*算法
                return self.improve_A_star(graph, source, target, heuristic_index, weight)
            else:
                raise ValueError("class-> Path_Planning -> run : algorithm must be Dijkstra or A*!")
        except ValueError as ve:
            print(f"ValueError: {ve}")

    def A_star(self,graph, source, target, heuristic_index=None, weight='weight'):
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        location = nx.get_node_attributes(graph, 'location')
        # 以画布中的绝对位置作为启发式预估参数
        if heuristic_index == 0:#欧几里得距离作为启发式函数:两点间的最短直线距离
            heuristic = lambda u, v: math.hypot(location[v][0] - location[u][0], location[v][1] - location[u][1])
        elif heuristic_index == 1:#曼哈顿距离作为启发式函数:横纵坐标绝对值之和
            heuristic = lambda u, v: abs(location[v][0] - location[u][0]) + abs(location[v][1] - location[u][1])
        elif heuristic_index == 2:#切比雪夫距离作为启发式函数：各座标数值差的最大值。适用于走斜线
            heuristic = lambda u, v: max((location[v][0] - location[u][0]), abs(location[v][1] - location[u][1]))
        else:
            heuristic = None
        path = nx.astar_path(graph, source=source, target=target, heuristic=heuristic, weight=weight)
        cost = nx.path_weight(graph, path, weight)
        return path, cost

    def improve_A_star(self, graph, source, target, heuristic_index, weight='weight'):
        try:
            k = 1.5  # h(n)预估函数的权重
            p = 0.00001  # h(n)的衡量单位
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
            explored = {}  # 记录节点是否已经探索过
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
                    return path, nx.path_weight(graph, path, weight)
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
                        h = k * h * (1 + p)                 # 为了区别相同的f(n)，引入一个权重p

                        # h = heuristic(neighbor, target)  # 计算邻居节点到目标节点的启发式评估值
                    enqueued[neighbor] = (ncost, h)      # 记录起点到邻居节点的距离ncost和邻居节点到目标节点的启发式评估值h
                    push(queue, (ncost + h, next(c), neighbor, ncost, current_node))  # 6.入队，并更新队列
                    # y = ncost + h 为A*算法的评估值，用于判断节点的优先级，使得算法更加贪婪，更加关注距离短的节点
            raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))  # 6.找不到路径，抛出异常
        except Exception as e:
            print(f"improve_A_star - Exception: {e}")


    def Dijkstra(self,graph, source, target,  weight='weight'):
        # 调用networkx的dijkstra算法
        path = nx.dijkstra_path(graph, source=source, target=target, weight=weight)
        cost = nx.path_weight(graph, path, weight)
        return path, cost

    def BiDirectional_Dijkstra(self,graph, source, target, weight='weight'):
        # 调用networkx的bidirectional_dijkstra算法
        finaldist, finalpath = nx.bidirectional_dijkstra(graph, source=source, target=target, weight=weight)
        return finalpath, finaldist
    def Bellman_Ford(self,graph, source, target,weight='weight'):
        # 调用networkx的bellman_ford算法
        path = nx.bellman_ford_path(graph, source=source,target=target ,weight=weight)
        cost = nx.path_weight(graph, path, weight)
        return path, cost

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
            if next_node == target:      # 目标节点，转向代价为0.5
                return target_turn_cost
            else:                        # 普通节点，转向代价为1
                return turn_cost
        return 0


    # 分析路径，计算路径长度和转向次数
    def Analyze_Path(self,graph, source, target, algorithm_index=None, heuristic_index=None):
        turn_count = 0  # 初始化转向计数
        start_time = time.time()
        path, cost = self.run(graph, source, target, algorithm_index, heuristic_index)
        take_time = time.time() - start_time
        if cost == 0 or path is None:
            return take_time, path, cost, turn_count  # 路径不存在，返回0
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

        return take_time, path, cost, turn_count


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
