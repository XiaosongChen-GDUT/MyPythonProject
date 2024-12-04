# This file contains the implementation of the A* algorithm for path planning
from networkx.algorithms.shortest_paths.weighted import _weight_function
import networkx as nx
from Program.DataModel import Model
from heapq import heappop, heappush
from itertools import count  # count object for the heap堆的计数对象
import math



class Path_Planning:
    def __init__(self):
        pass

    # def __init__(self, graph, source, target, heuristic=None, weight='weight'):
    #     self.graph = graph
    #     self.source = source
    #     self.target = target
    #     self.heuristic = heuristic  # A*算法的启发式函数
    #     self.weight = weight
    #     self.cost = 0  # 记录最短路径的总代价
    #     if source not in graph or target not in graph:
    #         msg = f"Either source {source} or target {target} is not in G"
    #         raise nx.NodeNotFound(msg)
    #     if heuristic is None:# if no heuristic is provided, use the Euclidean distance as the heuristic
    #         self.heuristic = lambda u, v: math.hypot(self.graph.nodes[v]['pos'][0] - self.graph.nodes[u]['pos'][0], self.graph.nodes[v]['pos'][1] - self.graph.nodes[u]['pos'][1])
    #     self.push = heappush    # push function for the heap堆的Push函数
    #     self.pop = heappop     # pop function for the heap堆的Pop函数
    #     self.weight_function = _weight_function(graph, weight)  # weight function for the graph
    #     self.G_succ = self.graph._adj  # 用于存储图中每个节点的邻接信息{1: {2: {'weight': 1.5}}, 2: {1: {'weight': 1.5}, 3: {'weight': 1.5}}, 3: {2: {'weight': 1.5}}}
    #     self.c = count()  # 计数器，用于生成唯一的ID
    #     self.queue = [(0, next(self.c), self.source, 0 ,None)]  # 队列，元素为元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
    #     self.enqueued = {}  # 记录节点是否已经入队，记录到达节点的距离  和 节点到目标节点启发式评估值
    #     self.explored = {}  # 记录节点是否已经探索过

    def A_star(self,graph, source, target, heuristic=None, weight='weight'):
        self.graph = graph
        self.source = source
        self.target = target
        self.heuristic = heuristic  # A*算法的启发式函数
        self.weight = weight
        if source not in graph or target not in graph:
            msg = f"Either source {source} or target {target} is not in G"
            raise nx.NodeNotFound(msg)
        if heuristic is None:# if no heuristic is provided, use the Euclidean distance as the heuristic
            self.heuristic = lambda u, v: math.hypot(self.graph.nodes[v]['pos'][0] - self.graph.nodes[u]['pos'][0], self.graph.nodes[v]['pos'][1] - self.graph.nodes[u]['pos'][1])
        self.push = heappush    # push function for the heap堆的Push函数
        self.pop = heappop     # pop function for the heap堆的Pop函数
        self.weight_function = _weight_function(graph, weight)  # weight function for the graph
        self.G_succ = self.graph._adj  # 用于存储图中每个节点的邻接信息{1: {2: {'weight': 1.5}}, 2: {1: {'weight': 1.5}, 3: {'weight': 1.5}}, 3: {2: {'weight': 1.5}}}
        self.c = count()  # 计数器，用于生成唯一的ID
        self.queue = [(0, next(self.c), self.source, 0 ,None)]  # 队列，元素为元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
        self.enqueued = {}  # 记录节点是否已经入队，记录到达节点的距离  和 节点到目标节点启发式评估值
        self.explored = {}  # 记录节点是否已经探索过
        while self.queue:
            # 弹出队列中代价最小的元素
            # 元组(节点的总代价，ID，当前节点，到当前节点成本，父节点)
            _, _, current_node, dist_current_node,parent = self.pop(self.queue)
            if current_node == self.target:     # 1.找到目标节点
                path = [current_node]
                node = parent
                while node is not None:
                    path.append(node)           # 反向构建路径
                    node = self.explored[node]  # 回溯父节点
                path.reverse()                  # 反转路径
                return path, dist_current_node
            if current_node in self.explored:  # 2.已经探索过，跳过
                if self.explored[current_node] is None:  # 已经探索过，但父节点为None，说明是源节点，跳过
                    continue
                qcost, h = self.enqueued[current_node]   # 取出到当前节点的距离和启发式评估值
                if qcost <= dist_current_node:  # 已经探索过，且距离更短所以方案更优，跳过本次探索
                    continue
            self.explored[current_node] = parent  # 记录父节点
            for neighbor, w in self.G_succ[current_node].items():   # 3.遍历当前节点的邻居节点
                print("current_node:",current_node,"neighbor:",neighbor,"weight:",w)
                cost = self.weight_function(current_node, neighbor, w)  # 计算当前节点到邻居节点的距离
                if cost is None:
                    continue
                ncost = dist_current_node + cost  # 计算起点到邻居节点的总代价：起点到当前节点的距离 + 当前节点到邻居节点的距离
                if neighbor  in self.enqueued:  # 4.判断邻居节点是否已经入队
                    qcost, h = self.enqueued[neighbor]  # 取出邻居节点的距离和启发式评估值
                    if qcost <= ncost:  # 邻居节点已经入队，且距离更短，跳过本次邻居节点的探索
                        continue
                else:
                    h = self.heuristic(neighbor, self.target)  # 计算邻居节点到目标节点的启发式评估值
                self.enqueued[neighbor] = (ncost, h)  # 记录起点到邻居节点的距离ncost和邻居节点到目标节点的启发式评估值h
                self.push(self.queue, (ncost + h, next(self.c), neighbor, ncost, current_node))  # 5.入队，并更新队列
                # y = ncost + h 为A*算法的评估值，用于判断节点的优先级，使得算法更加贪婪，更加关注距离短的节点
        raise nx.NetworkXNoPath("No path between %s and %s." % (self.source, self.target))  # 6.找不到路径，抛出异常

    def dijkstra(self):
        # 调用networkx的dijkstra算法
        path = nx.dijkstra_path(self.graph, source=self.source, target=self.target, weight=self.weight)
        cost = nx.path_weight(self.graph, path, self.weight)
        return path, cost


def main():
    # Example usage of the A* algorithm
    model = Model()
    combined_graph, floor1, floor2, floor3 = model.combined_graph, model.floor1, model.floor2, model.floor3
    planing =  Path_Planning()
    path, cost = planing.A_star(floor1, 1, 180, weight='weight')
    print("A*算法最短路径：",path,"，A*算法路径长度：",cost ,"，A*算法path_weight：",nx.path_weight(floor1,path,'weight'))
    dijkstra_path, dijkstra_cost = planing.dijkstra()
    print("dijkstra算法最短路径：",dijkstra_path,"，dijkstra算法路径长度：",dijkstra_cost)
    # path = nx.shortest_path(floor1,source=1,target=10,weight='weight',method='dijkstra')
    # cost = nx.path_weight(floor1,path,'weight')
    # print("dijkstra最短路径：",path,"，dijkstra路径长度：",cost)
    # a_star_path = nx.astar_path(floor1,source=1,target=10,weight='weight')
    # my_cost = nx.path_weight(floor1,a_star_path,'weight')
    # print("a_star_path路径长度：",my_cost)


if __name__ == '__main__':
    main()
