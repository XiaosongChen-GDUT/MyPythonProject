import random

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq
import math
from heapq import heappush, heappop
from itertools import count
from collections import defaultdict
from Program.DataModel import Model
# 假设这些是您的全局变量，来自原始代码
heights = [3.15, 2.55, 1.5]  # 各楼层高度
cross_nodes = {1: [642, 674, 1116, 1148], 2: [2374, 2406, 2844, 2876], 3: [3899, 4135]}  # 换层节点

"""
    # 1. 多车路径规划算法类,Grok生成的代码，为了实现等待的功能，暂未实现
    """
class MultiVehiclePathPlanner:
    def __init__(self, graph, vehicles, tasks):
        """
        初始化路径规划器。

        参数:
            graph (nx.Graph): 网络图，包含节点和边的权重信息。
            vehicles (list): 车辆列表，每个车辆包含 {'id': int, 'pos': int, 'status': str}。
            tasks (list): 任务列表，每个任务包含 {'id': int, 'source': int, 'target': int, 'priority': int, 'time_window': tuple, 'agv_id': int}。
        """
        self.graph = graph  # 图结构，用于路径计算
        self.vehicles = vehicles  # 车辆信息
        self.tasks = tasks  # 任务信息
        self.edge_occupancy = defaultdict(list)  # 边占用记录，格式: {edge: [(start_time, end_time, vehicle_id), ...]}
        self.node_occupancy = defaultdict(list)  # 节点占用记录，格式: {node: [(start_time, end_time, vehicle_id), ...]}
        self.paths = {}  # 路径结果，格式: {vehicle_id: {'path': [nodes], 'times': [times]}}
        self.MAX_WAIT_TIME = 5.0  # 最大等待时间阈值，若等待时间超过此值，可能选择绕行

        # 初始化车辆的初始位置占用
        for vehicle in vehicles:
            task = next((t for t in tasks if t['agv_id'] == vehicle['id']), None)  # 查找车辆对应的任务
            start_time = task['time_window'][0] if task else 0  # 若有任务，取任务开始时间，否则为0
            self.node_occupancy[vehicle['pos']].append((0, start_time, vehicle['id']))  # 记录从0到任务开始时间的占用
            print(f"车辆 {vehicle['id']} 在节点 {vehicle['pos']} 的初始占用直到 {start_time}")

    def improve_A_star(self, source, target, vehicle_id, task_priority, time_window, start_time=0):
        """
        改进的A*算法，规划单辆车的路径，考虑冲突和时间窗。

        参数:
            source (int): 起点节点
            target (int): 目标节点
            vehicle_id (int): 车辆ID
            task_priority (int): 任务优先级
            time_window (tuple): 时间窗 (start, end)
            start_time (float): 路径开始时间，默认为0

        返回:
            tuple: (路径列表, 路径总成本, 时间列表) 或 None（无路径）
        """
        # 检查起点和目标是否在图中
        if source not in self.graph or target not in self.graph:
            return None
        # 如果起点和目标相同，返回单节点路径
        if source == target:
            return [source], 0, [start_time]

        # 获取节点位置属性并定义启发式函数
        location = nx.get_node_attributes(self.graph, 'location')
        heuristic = lambda u, v: abs(location[v][0] - location[u][0]) + abs(location[v][1] - location[u][1])
        # 初始化优先级队列，格式: (f_score, 计数器, 节点, g_score, 父节点, 当前时间)
        queue = [(0, next(count()), source, 0, None, start_time)]
        enqueued = {}  # 记录已入队的节点和成本
        explored = {}  # 记录已探索的节点及其父节点和时间

        while queue:
            # 从队列中取出f值最小的节点
            _, _, current_node, dist_current_node, parent, current_time = heappop(queue)
            if current_node == target:  # 到达目标，构建路径
                path = [current_node]
                times = [current_time]
                node = parent
                while node is not None:
                    path.append(node)
                    times.append(explored[node][1])
                    node = explored[node][0]
                path.reverse()
                times.reverse()
                return path, round(nx.path_weight(self.graph, path, 'weight'), 2), times

            if current_node in explored:  # 已探索过的节点，跳过
                continue

            explored[current_node] = (parent, current_time)  # 记录当前节点的父节点和到达时间
            # 遍历当前节点的邻居
            for neighbor, w in self.graph.adj[current_node].items():
                cost = w['weight']  # 边权重
                turn_cost = self.turn_cost(location, parent, current_node, neighbor)  # 计算转弯成本
                ncost = dist_current_node + cost + turn_cost  # 当前路径总成本
                arrival_time = current_time + cost  # 到达邻居的时间

                edge = tuple(sorted([current_node, neighbor]))  # 无向边，排序以统一表示
                # 检查冲突
                conflict = self.check_conflict(edge, current_node, neighbor, arrival_time - cost, arrival_time, vehicle_id, task_priority)
                if conflict:
                    conflict_type, other_vehicle_id, other_priority = conflict
                    wait_time, detour_node = self.resolve_conflict(conflict_type, current_node, neighbor, vehicle_id, task_priority, other_vehicle_id, other_priority, arrival_time)
                    if wait_time > 0:  # 需要等待
                        print(f"车辆 {vehicle_id}: 在节点 {current_node} 等待 {wait_time} 单位时间，因 {conflict_type} 冲突与车辆 {other_vehicle_id}")
                        arrival_time += wait_time  # 更新到达时间
                    elif detour_node:  # 需要绕行
                        print(f"车辆 {vehicle_id}: 绕行至节点 {detour_node}，因 {conflict_type} 冲突与车辆 {other_vehicle_id}")
                        detour_result = self.improve_A_star(detour_node, target, vehicle_id, task_priority, time_window, arrival_time)
                        if detour_result:
                            detour_path, detour_cost, detour_times = detour_result
                            path_to_detour = [current_node, detour_node]
                            times_to_detour = [current_time, arrival_time]
                            return path_to_detour[:-1] + detour_path, ncost + detour_cost, times_to_detour[:-1] + detour_times
                        continue  # 绕行失败，跳过此邻居
                    else:  # 无绕行方案，默认等待
                        print(f"车辆 {vehicle_id}: 无绕行方案，等待 {wait_time} 单位时间，因 {conflict_type} 冲突")
                        arrival_time += wait_time

                # 计算启发式值并加入队列
                h = heuristic(neighbor, target)
                D = heuristic(source, current_node) or 1
                beta = math.e ** (h / D)  # 动态调整启发式权重
                h *= beta

                if neighbor in enqueued and enqueued[neighbor][0] <= ncost:  # 若已有更优路径，跳过
                    continue

                enqueued[neighbor] = (ncost, h)
                heappush(queue, (ncost + h, next(count()), neighbor, ncost, current_node, arrival_time))

        return None  # 无路径可达

    def turn_cost(self, location, parent, current, neighbor):
        """
        计算转弯成本，若路径方向相反则增加成本。

        参数:
            location (dict): 节点位置字典
            parent (int): 父节点
            current (int): 当前节点
            neighbor (int): 邻居节点

        返回:
            float: 转弯成本（1.0 或 0）
        """
        if parent is None:  # 无父节点，无转弯
            return 0
        dx1 = location[current][0] - location[parent][0]  # 前段x方向
        dy1 = location[current][1] - location[parent][1]  # 前段y方向
        dx2 = location[neighbor][0] - location[current][0]  # 后段x方向
        dy2 = location[neighbor][1] - location[current][1]  # 后段y方向
        if dx1 * dx2 + dy1 * dy2 < 0:  # 方向相反（点积<0），增加转弯成本
            return 1.0
        return 0

    def check_conflict(self, edge, current_node, neighbor, start_time, end_time, vehicle_id, priority):
        """
        检查路径冲突类型，包括边冲突、节点冲突和安全距离。

        参数:
            edge (tuple): 检查的边 (node1, node2)
            current_node (int): 当前节点
            neighbor (int): 邻居节点
            start_time (float): 进入边的时间
            end_time (float): 离开边的时间
            vehicle_id (int): 当前车辆ID
            priority (int): 当前任务优先级

        返回:
            tuple: (冲突类型, 其他车辆ID, 其他优先级) 或 None
        """
        # 检查边冲突
        for occ_start, occ_end, occ_vehicle_id in self.edge_occupancy[edge]:
            if occ_vehicle_id != vehicle_id and not (end_time <= occ_start or start_time >= occ_end):  # 时间重叠
                other_priority = next((t['priority'] for t in self.tasks if t['agv_id'] == occ_vehicle_id), 0)
                print(f"车辆 {vehicle_id}: 在边 {edge} 上与车辆 {occ_vehicle_id} 发生潜在冲突，时间 [{start_time}, {end_time}] vs [{occ_start}, {occ_end}]")
                # 检查相向冲突
                for path_data in self.paths.values():
                    path = path_data['path']
                    times = path_data['times']
                    for i in range(len(path) - 1):
                        if path[i] == neighbor and path[i + 1] == current_node and times[i] <= end_time and times[i + 1] >= start_time:
                            print(f"车辆 {vehicle_id}: 与车辆 {occ_vehicle_id} 发生相向冲突")
                            return ('opposite', occ_vehicle_id, other_priority)
                        # 检查追尾冲突
                        if path[i] == current_node and path[i + 1] == neighbor and times[i] <= end_time and times[i + 1] >= start_time:
                            print(f"车辆 {vehicle_id}: 与车辆 {occ_vehicle_id} 发生追尾冲突")
                            return ('rear_end', occ_vehicle_id, other_priority)

        # 检查节点冲突
        for node in [current_node, neighbor]:
            for occ_start, occ_end, occ_vehicle_id in self.node_occupancy[node]:
                if occ_vehicle_id != vehicle_id and not (end_time <= occ_start or start_time >= occ_end):
                    other_priority = next((t['priority'] for t in self.tasks if t['agv_id'] == occ_vehicle_id), 0)
                    print(f"车辆 {vehicle_id}: 在节点 {node} 与车辆 {occ_vehicle_id} 发生节点冲突，时间 [{start_time}, {end_time}] vs [{occ_start}, {occ_end}]")
                    return ('node', occ_vehicle_id, other_priority)

        # 检查安全距离
        for vid, path_data in self.paths.items():
            path = path_data['path']
            times = path_data['times']
            for i in range(len(path) - 1):
                if times[i] <= end_time and times[i + 1] >= start_time:  # 时间重叠
                    dist = nx.shortest_path_length(self.graph, current_node, path[i])  # 计算两车距离
                    if dist < 2:  # 安全距离小于2个节点
                        other_priority = next((t['priority'] for t in self.tasks if t['agv_id'] == vid), 0)
                        print(f"车辆 {vehicle_id}: 在边 {edge} 与车辆 {vid} 违反安全距离，距离 {dist}")
                        return ('safety_violation', vid, other_priority)
        return None  # 无冲突

    def get_wait_time(self, node, arrival_time, other_vehicle_id):
        """
        计算等待时间，直到节点或边空闲。
        参数:
            node (int): 当前节点
            arrival_time (float): 到达时间
            other_vehicle_id (int): 冲突车辆ID
        返回:
            float: 等待时间
        """
        max_end_time = 0  # 记录最晚的占用结束时间
        for start, end, vid in self.node_occupancy[node]:
            if vid != other_vehicle_id and arrival_time < end:  # 考虑未结束的占用
                max_end_time = max(max_end_time, end)
        for edge in self.graph.edges(node):
            for start, end, vid in self.edge_occupancy[tuple(sorted(edge))]:
                if vid != other_vehicle_id and arrival_time < end:
                    max_end_time = max(max_end_time, end)
        wait_time = max(0, max_end_time - arrival_time)  # 计算等待时间
        print(f"车辆 {other_vehicle_id}: 在节点 {node} 的等待时间: {wait_time} (最晚结束时间={max_end_time}, 到达时间={arrival_time})")
        return wait_time

    def resolve_conflict(self, conflict_type, current_node, neighbor, vehicle_id, priority, other_vehicle_id, other_priority, arrival_time):
        """
        解决路径冲突，优先选择等待。

        参数:
            conflict_type (str): 冲突类型 ('opposite', 'rear_end', 'node', 'safety_violation')
            current_node (int): 当前节点
            neighbor (int): 邻居节点
            vehicle_id (int): 当前车辆ID
            priority (int): 当前任务优先级
            other_vehicle_id (int): 冲突车辆ID
            other_priority (int): 冲突车辆优先级
            arrival_time (float): 到达邻居的时间

        返回:
            tuple: (等待时间, 绕行节点) 或 (0, None)
        """
        wait_time = self.get_wait_time(current_node, arrival_time, other_vehicle_id)  # 计算等待时间
        detour_cost = None
        detour_node = None

        # 寻找绕行节点
        for detour in self.graph.neighbors(current_node):
            if detour != neighbor and not self.check_conflict(tuple(sorted([current_node, detour])), current_node, detour, arrival_time, arrival_time + 1, vehicle_id, priority):
                detour_cost = nx.shortest_path_length(self.graph, detour, neighbor, weight='weight')
                detour_node = detour
                break

        # 根据优先级决定处理方式
        if priority < other_priority:  # 低优先级车辆
            if wait_time <= self.MAX_WAIT_TIME and (detour_cost is None or wait_time < detour_cost):
                return wait_time, None  # 等待成本更低
            elif detour_node:
                return 0, detour_node  # 绕行
            else:
                return wait_time, None  # 无绕行，默认等待
        elif priority > other_priority:  # 高优先级车辆，直接通行
            return 0, None
        else:  # 优先级相等，后车等待
            if wait_time <= self.MAX_WAIT_TIME and (detour_cost is None or wait_time < detour_cost):
                return wait_time, None
            elif detour_node:
                return 0, detour_node
            else:
                return wait_time, None

    def plan_paths(self):
        """
        为所有车辆规划路径，按优先级排序。
        """
        # 创建任务队列，按优先级排序（负数使高优先级在前）
        task_queue = [(-task['priority'], task['id'], task) for task in self.tasks]
        heapq.heapify(task_queue)

        while task_queue:
            _, task_id, task = heapq.heappop(task_queue)  # 取出优先级最高的任务
            source, target = task['source'], task['target']
            time_window = task['time_window']
            agv_id = task.get('agv_id')

            # 选择车辆，优先使用指定车辆，否则选最近的空闲车辆
            vehicle = next((v for v in self.vehicles if v['id'] == agv_id and v['status'] == 'idle'), None)
            if not vehicle:
                print(f"任务 {task_id}: 无可用车辆")
                continue

            print(f"规划车辆 {vehicle['id']} 的任务 {task_id}: {vehicle['pos']} -> {source} -> {target}")
            start_time = max(time_window[0], 0)  # 取时间窗开始时间
            # 规划从当前位置到任务起点的路径
            result = self.improve_A_star(vehicle['pos'], source, vehicle['id'], task['priority'], time_window, start_time)
            if not result:
                print(f"任务 {task_id}: 从 {vehicle['pos']} 到 {source} 无路径")
                continue
            path_to_source, _, times_to_source = result

            # 规划从任务起点到目标的路径
            result = self.improve_A_star(source, target, vehicle['id'], task['priority'], time_window, times_to_source[-1])
            if not result:
                print(f"任务 {task_id}: 从 {source} 到 {target} 无路径")
                continue
            path_to_target, cost, times_to_target = result

            # 合并路径和时间
            full_path = path_to_source[:-1] + path_to_target
            full_times = times_to_source[:-1] + times_to_target
            self.paths[vehicle['id']] = {'path': full_path, 'times': full_times}
            print(f"车辆 {vehicle['id']}: 路径规划完成 - {full_path}, 时间: {full_times}")

            # 更新边和节点占用
            for i in range(len(full_path) - 1):
                edge = tuple(sorted([full_path[i], full_path[i + 1]]))
                self.edge_occupancy[edge].append((full_times[i], full_times[i + 1], vehicle['id']))
                self.node_occupancy[full_path[i]].append((full_times[i], full_times[i + 1], vehicle['id']))  # 记录时间区间
            self.node_occupancy[full_path[-1]].append((full_times[-1], full_times[-1], vehicle['id']))

            # 更新车辆状态
            vehicle['pos'] = target
            vehicle['status'] = 'busy'


# 2. 三维静态显示类
class PathVisualizer:
    def __init__(self, graph, planner):
        """
        初始化路径可视化器。

        :param graph: NetworkX图
        :param planner: MultiVehiclePathPlanner实例
        """
        self.graph = graph
        self.planner = planner
        self.location = nx.get_node_attributes(graph, 'location')  # 二维位置

    def plot_paths(self):
        """
        绘制AGV路径的三维静态图，z轴为时间。
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制地图节点（二维位置，z=0）
        # 获取原始坐标
        original_xs = [self.location[node][0] for node in self.graph.nodes]
        original_ys = [self.location[node][1] for node in self.graph.nodes]
        # 计算Y轴翻转参数
        max_y = max(original_ys)  # 获取原始坐标系的最大Y值
        # 坐标转换：X保持不变，Y进行镜像翻转
        xs = original_xs
        ys = [max_y - y for y in original_ys]  # 关键转换步骤
        # 然后重新计算 xs 和 ys
        zs = [0] * len(xs)  # 初始时间为0
        # xs = [self.location[node][0] for node in self.graph.nodes]
        # ys = [self.location[node][1] for node in self.graph.nodes]
        ax.scatter(xs, ys, zs, c='gray', s=10, alpha=0.15, label='Nodes')

        # 设置x轴和y轴的显示范围，确保不会超出边界
        ax.set_xlim(min(xs) - 2, max(xs) + 2)  # 稍微扩展边界以便更好显示
        ax.set_ylim(min(ys) - 2, max(ys) + 2)  # 稍微扩展边界以便更好显示


        # 绘制每辆AGV的路径
        colors = ['r', 'b', 'g', 'y', 'm', 'c']  # 为每辆AGV分配不同颜色
        for i, (vid, path_data) in enumerate(self.planner.paths.items()):
            path = path_data['path']
            times = path_data['times']

            # 提取路径的x, y, z（时间）坐标
            path_xs = [self.location[node][0] for node in path]
            path_ys = [self.location[node][1] for node in path]
            path_zs = times  # z轴为时间

            # 绘制路径线条
            ax.plot(path_xs, path_ys, path_zs, c=colors[i % len(colors)], label=f'四向穿梭车ID：{vid}', linewidth=5)
            # 标记路径点
            ax.scatter(path_xs, path_ys, path_zs, c=colors[i % len(colors)], s=40,alpha=0.6)
        def on_zoom(event):
            # 只有在3D轴的情况下才进行调整
            if event.inaxes == ax:
                # 获取当前视图的x轴和y轴的范围
                xlim = event.inaxes.get_xlim()
                ylim = event.inaxes.get_ylim()

                # 根据当前视图的范围调整边界
                # 这里可以根据缩放方向（event.button）来决定是放大还是缩小
                if event.button == 'up':
                    # 放大时缩小边界
                    ax.set_xlim(xlim[0] - 5, xlim[1] + 5)
                    ax.set_ylim(ylim[0] - 5, ylim[1] + 5)
                elif event.button == 'down':
                    # 缩小时扩大边界
                    ax.set_xlim(xlim[0] + 5, xlim[1] - 5)
                    ax.set_ylim(ylim[0] + 5, ylim[1] - 5)
                # 重新绘制图像
                plt.draw()
        fig.canvas.mpl_connect('scroll_event', on_zoom)  # 鼠标移动事件处理

        # 设置轴标签
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Time')
        ax.set_title(' Paths with Time Axis')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    # 创建示例图（假设已有）
    model = Model()
    G = model.combined_graph

    # 初始化车辆和任务
    vehicles = [{'id': 1, 'pos': 1620, 'status': 'idle'},
                {'id': 2, 'pos': 925, 'status': 'idle'},
                {'id': 3, 'pos': 348, 'status': 'idle'},
                {'id': 4, 'pos': 1174, 'status': 'idle'}]
    tasks = [
        {'id': 1, 'source': 1620, 'target': 323, 'priority': 2, 'time_window': (0, 1000), 'agv_id': 1},
        {'id': 2, 'source': 925, 'target': 1727, 'priority': 1, 'time_window': (0, 1000), 'agv_id': 2},
        {'id': 3, 'source': 1174, 'target': 1399, 'priority': 2, 'time_window': (0, 1000), 'agv_id': 3},
        {'id': 4, 'source': 348, 'target': 1727, 'priority': 3, 'time_window': (0, 1000), 'agv_id': 4}
    ]

    # 运行规划
    planner = MultiVehiclePathPlanner(G, vehicles, tasks)
    planner.plan_paths()

    # 输出最终路径和时间
    print("\nFinal Paths and Times:")
    for vid, data in planner.paths.items():
        print(f"Vehicle {vid}: Path = {data['path']}, Times = {data['times']}")
    # 显示结果
    viz = PathVisualizer(G, planner)
    viz.plot_paths()
