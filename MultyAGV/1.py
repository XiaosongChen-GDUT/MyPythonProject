import networkx as nx
import heapq
from heapq import heappush, heappop
from itertools import count
from collections import defaultdict
import random
import math

class MultiVehiclePathPlanner:
    def __init__(self, graph, vehicles, tasks, cross_nodes):
        """
        初始化多车路径规划器。

        :param graph: NetworkX加权无向图
        :param vehicles: 车辆列表，格式 [{'id': int, 'pos': node_id, 'status': 'idle'}, ...]
        :param tasks: 任务列表，格式 [{'id': int, 'source': node, 'target': node, 'priority': int, 'time_window': (start, end), 'agv_id': int}, ...]
        :param cross_nodes: 提升机节点字典，格式 {layer: [node_ids]}
        """
        self.graph = graph
        self.vehicles = vehicles
        self.tasks = tasks
        self.cross_nodes = cross_nodes
        self.edge_occupancy = defaultdict(list)  # 边占用时间表 {edge: [(start_time, end_time, vehicle_id), ...]}
        self.node_occupancy = defaultdict(list)  # 节点占用时间表 {node: [(start_time, end_time, vehicle_id), ...]}
        self.paths = {}  # {vehicle_id: {'path': [nodes], 'times': [times]}}

    def improve_A_star(self, source, target, vehicle_id, task_priority, time_window, start_time=0):
        """
        改进A*算法，考虑时间窗、优先级和冲突。

        :param source: 起点节点
        :param target: 终点节点
        :param vehicle_id: 当前车辆ID
        :param task_priority: 任务优先级（1到4，越大越优先）
        :param time_window: 时间窗 (start, end)
        :param start_time: 路径规划的起始时间
        :return: (path, total_cost, time_schedule) 或 None
        """
        if source not in self.graph or target not in self.graph:
            return None
        if source == target:
            return [source], 0, [start_time]

        location = nx.get_node_attributes(self.graph, 'location')
        pos = nx.get_node_attributes(self.graph, 'pos')
        source_layer = pos[source][2]
        target_layer = pos[target][2]
        cross_flag = source_layer != target_layer
        source_cross_nodes_set = set(self.cross_nodes[source_layer])
        target_cross_nodes_set = set(self.cross_nodes[target_layer]) if cross_flag else set()

        heuristic = lambda u, v: abs(location[v][0] - location[u][0]) + abs(location[v][1] - location[u][1])
        queue = [(0, next(count()), source, 0, None, start_time)]  # (f_cost, id, node, g_cost, parent, current_time)
        enqueued = {}
        explored = {}

        while queue:
            _, _, current_node, dist_current_node, parent, current_time = heappop(queue)
            if current_node == target:
                path = [current_node]
                times = [current_time]
                node = parent
                while node is not None:
                    path.append(node)
                    times.append(explored[node][1])
                    node = explored[node][0]
                path.reverse()
                times.reverse()
                if times[-1] > time_window[1]:
                    print(f"车辆 {vehicle_id}: 最终到达时间 {times[-1]} 超出时间窗 {time_window}")
                    return None
                return path, round(nx.path_weight(self.graph, path, 'weight'), 2), times

            if current_node in explored:
                continue

            explored[current_node] = (parent, current_time)
            for neighbor, w in self.graph.adj[current_node].items():
                if cross_flag:
                    if current_node in source_cross_nodes_set and pos[neighbor][2] == source_layer:
                        continue
                    elif neighbor in target_cross_nodes_set:
                        continue
                elif neighbor in source_cross_nodes_set:
                    continue

                cost = w['weight']
                turn_cost = self.turn_cost(location, parent, current_node, neighbor)
                ncost = dist_current_node + cost + turn_cost
                arrival_time = current_time + cost

                if arrival_time > time_window[1]:
                    print(f"车辆 {vehicle_id}: 到达节点 {neighbor} 的时间 {arrival_time} 超出时间窗 {time_window}")
                    continue

                edge = tuple(sorted([current_node, neighbor]))
                conflict = self.check_conflict(edge, current_node, neighbor, current_time, arrival_time, vehicle_id, task_priority)
                if conflict:
                    conflict_type, other_vehicle_id, other_priority = conflict
                    print(f"当前车辆 {vehicle_id}: 与车辆 {other_vehicle_id} 发生冲突，类型 {conflict_type}")
                    wait_time, detour_node = self.resolve_conflict(conflict_type, current_node, neighbor, vehicle_id, task_priority, other_vehicle_id, other_priority, arrival_time)
                    if wait_time > 0:  # 等待
                        arrival_time += wait_time
                        if arrival_time > time_window[1]:
                            print(f"车辆 {vehicle_id}: 等待后到达时间 {arrival_time} 超出时间窗 {time_window}")
                            continue
                    elif detour_node:  # 绕行
                        detour_result = self.improve_A_star(detour_node, target, vehicle_id, task_priority, time_window, arrival_time)
                        if detour_result:
                            detour_path, detour_cost, detour_times = detour_result
                            path_to_detour = [current_node, detour_node]
                            times_to_detour = [current_time, arrival_time]
                            return path_to_detour[:-1] + detour_path, ncost + detour_cost, times_to_detour[:-1] + detour_times
                        continue

                h = heuristic(neighbor, target)
                D = heuristic(source, current_node) or 1
                beta = math.e ** (h / D)
                h *= beta

                if neighbor in enqueued and enqueued[neighbor][0] <= ncost:
                    continue

                enqueued[neighbor] = (ncost, h)
                heappush(queue, (ncost + h, next(count()), neighbor, ncost, current_node, arrival_time))

        return None

    def turn_cost(self, location, parent, current, neighbor):
        """
        计算拐点成本。

        :param location: 节点位置字典
        :param parent: 父节点
        :param current: 当前节点
        :param neighbor: 邻居节点
        :return: 拐点成本（1.2 或 0）
        """
        if parent is None:
            return 0
        dx1 = location[current][0] - location[parent][0]
        dy1 = location[current][1] - location[parent][1]
        dx2 = location[neighbor][0] - location[current][0]
        dy2 = location[neighbor][1] - location[current][1]
        if dx1 * dx2 + dy1 * dy2 < 0:
            return 1.2
        return 0

    def check_conflict(self, edge, current_node, neighbor, start_time, end_time, vehicle_id, priority):
        """
        检查路径冲突类型。

        :param edge: 当前车辆尝试使用的边，格式为 tuple(sorted([current_node, neighbor]))
        :param current_node: 当前节点
        :param neighbor: 邻居节点（目标节点）
        :param start_time: 进入边的时间
        :param end_time: 离开边的时间
        :param vehicle_id: 当前车辆 ID
        :param priority: 当前任务优先级
        :return: (conflict_type, other_vehicle_id, other_priority) 或 None
        """
        # 检查时间窗
        task = next((t for t in self.tasks if t['agv_id'] == vehicle_id), None)
        if task:
            time_window = task['time_window']
            if start_time < time_window[0] or end_time > time_window[1]:
                print(f"车辆 {vehicle_id}: 时间窗冲突，当前时间 [{start_time}, {end_time}] 超出时间窗 {time_window}")
                return ('time_window', vehicle_id, priority)

        # 检查边冲突
        for occ_start, occ_end, occ_vehicle_id in self.edge_occupancy[edge]:
            if occ_vehicle_id == vehicle_id:
                continue
            if not (end_time <= occ_start or start_time >= occ_end):
                other_priority = next((t['priority'] for t in self.tasks if t['agv_id'] == occ_vehicle_id), 0)
                other_path_data = self.paths.get(occ_vehicle_id)
                if not other_path_data:
                    continue
                other_path = other_path_data['path']
                other_times = other_path_data['times']

                # 相向冲突
                for i in range(len(other_path) - 1):
                    if (other_path[i] == neighbor and other_path[i + 1] == current_node and
                            other_times[i] <= end_time and other_times[i + 1] >= start_time):
                        print(f"车辆 {vehicle_id}: 与车辆 {occ_vehicle_id} 在边 {edge} 发生相向冲突")
                        return ('opposite', occ_vehicle_id, other_priority)

                # 追尾冲突
                for i in range(len(other_path) - 1):
                    if (other_path[i] == current_node and other_path[i + 1] == neighbor and
                            other_times[i] <= end_time and other_times[i + 1] >= start_time):
                        print(f"车辆 {vehicle_id}: 与车辆 {occ_vehicle_id} 在边 {edge} 发生追尾冲突")
                        return ('rear_end', occ_vehicle_id, other_priority)

        # 检查节点冲突
        for node in (current_node, neighbor):
            for occ_start, occ_end, occ_vehicle_id in self.node_occupancy[node]:
                if occ_vehicle_id == vehicle_id:
                    continue
                if not (end_time <= occ_start or start_time >= occ_end):
                    other_priority = next((t['priority'] for t in self.tasks if t['agv_id'] == occ_vehicle_id), 0)
                    print(f"车辆 {vehicle_id}: 与车辆 {occ_vehicle_id} 在节点 {node} 发生节点冲突")
                    return ('node', occ_vehicle_id, other_priority)

        return None

    def resolve_conflict(self, conflict_type, current_node, neighbor, vehicle_id, priority, other_vehicle_id, other_priority, arrival_time):
        """
        根据冲突类型和优先级处理冲突。

        :param conflict_type: 冲突类型 ('opposite', 'node', 'rear_end', 'time_window')
        :param current_node: 当前节点
        :param neighbor: 邻居节点
        :param vehicle_id: 当前车辆 ID
        :param priority: 当前任务优先级
        :param other_vehicle_id: 冲突的另一辆车 ID
        :param other_priority: 另一辆车的任务优先级
        :param arrival_time: 到达邻居节点的时间
        :return: (wait_time, detour_node) - 等待时间或绕行节点
        """
        if conflict_type == 'time_window':
            print(f"Vehicle {vehicle_id}: Time window conflict, cannot proceed")
            return float('inf'), None

        detour_cost = None
        detour_node = None
        wait_time = self.get_wait_time(current_node, arrival_time, other_vehicle_id)

        # 寻找可行绕行节点
        for detour in self.graph.neighbors(current_node):
            if detour != neighbor:
                cost = self.graph[current_node][detour]['weight']
                if not self.check_conflict(tuple(sorted([current_node, detour])), current_node, detour, arrival_time, arrival_time + cost, vehicle_id, priority):
                    detour_cost = nx.shortest_path_length(self.graph, detour, neighbor, weight='weight')
                    detour_node = detour
                    break

        # 根据优先级处理冲突
        if priority < other_priority:
            if detour_cost is None or wait_time < detour_cost:
                print(f"Vehicle {vehicle_id}: Priority {priority} < {other_priority}, waiting {wait_time} units")
                return wait_time, None
            elif detour_node:
                print(f"Vehicle {vehicle_id}: Priority {priority} < {other_priority}, detouring to {detour_node}")
                return 0, detour_node
            else:
                print(f"Vehicle {vehicle_id}: Priority {priority} < {other_priority}, forced to wait {wait_time}")
                return wait_time, None
        elif priority > other_priority:
            print(f"Vehicle {vehicle_id}: Priority {priority} > {other_priority}, proceeding")
            return 0, None
        else:
            if random.choice([True, False]):
                print(f"Vehicle {vehicle_id}: Priority equal, proceeding")
                return 0, None
            else:
                print(f"Vehicle {vehicle_id}: Priority equal, waiting {wait_time} units")
                return wait_time, None

    def get_wait_time(self, node, arrival_time, other_vehicle_id):
        """
        计算等待时间直到节点空闲。

        :param node: 当前节点
        :param arrival_time: 到达时间
        :param other_vehicle_id: 冲突的另一辆车 ID
        :return: 等待时间
        """
        max_end_time = 0
        for start, end, vid in self.node_occupancy[node]:
            if vid != other_vehicle_id and arrival_time < end:
                max_end_time = max(max_end_time, end)
        for edge in self.graph.edges(node):
            for start, end, vid in self.edge_occupancy[tuple(sorted(edge))]:
                if vid != other_vehicle_id and arrival_time < end:
                    max_end_time = max(max_end_time, end)
        wait_time = max(0, max_end_time - arrival_time)
        if wait_time == 0 and max_end_time > 0:
            wait_time = 1  # 至少等待 1 个时间单位
        return wait_time
        return max(0, max_end_time - arrival_time)

    def plan_paths(self):
        """
        为所有车辆规划路径，考虑优先级和冲突。
        """
        task_queue = [(-task['priority'], task['id'], task) for task in self.tasks]
        heapq.heapify(task_queue)

        last_vehicle_end_time = 0  # 记录上一辆车的结束时间
        while task_queue:
            _, task_id, task = heapq.heappop(task_queue)
            source, target = task['source'], task['target']
            time_window = task['time_window']
            agv_id = task.get('agv_id')
            vehicle = next((v for v in self.vehicles if v['id'] == agv_id and v['status'] == 'idle'), None) or \
                      min((v for v in self.vehicles if v['status'] == 'idle'),
                          key=lambda v: nx.shortest_path_length(self.graph, v['pos'], source, weight='weight'), default=None)
            if not vehicle:
                print(f"No available vehicle for task {task_id}")
                continue

            # 确保车辆在上一辆车之后出发
            start_time = max(time_window[0], last_vehicle_end_time)
            result = self.improve_A_star(vehicle['pos'], source, vehicle['id'], task['priority'], time_window, start_time)
            if not result:
                print(f"No path from {vehicle['pos']} to {source} for task {task_id}")
                continue
            path_to_source, _, times_to_source = result

            result = self.improve_A_star(source, target, vehicle['id'], task['priority'], time_window, times_to_source[-1])
            if not result:
                print(f"No path from {source} to {target} for task {task_id}")
                continue
            path_to_target, cost, times_to_target = result

            full_path = path_to_source[:-1] + path_to_target
            full_times = times_to_source[:-1] + times_to_target
            self.paths[vehicle['id']] = {'path': full_path, 'times': full_times}

            # 更新占用信息
            for i in range(len(full_path) - 1):
                edge = tuple(sorted([full_path[i], full_path[i + 1]]))
                self.edge_occupancy[edge].append((full_times[i], full_times[i + 1], vehicle['id']))
                self.node_occupancy[full_path[i]].append((full_times[i], full_times[i], vehicle['id']))
            self.node_occupancy[full_path[-1]].append((full_times[-1], full_times[-1], vehicle['id']))

            vehicle['pos'] = target
            vehicle['status'] = 'busy'
            last_vehicle_end_time = full_times[-1]  # 更新结束时间

    def plan_paths(self):
        """
        为所有车辆规划路径，考虑优先级和冲突。
        """
        task_queue = [(-task['priority'], task['id'], task) for task in self.tasks]  # 负数使高优先级在前
        heapq.heapify(task_queue)

        while task_queue:
            _, task_id, task = heapq.heappop(task_queue)
            source, target = task['source'], task['target']
            time_window = task['time_window']
            agv_id = task.get('agv_id')
            vehicle = next((v for v in self.vehicles if v['id'] == agv_id and v['status'] == 'idle'), None) or \
                      min((v for v in self.vehicles if v['status'] == 'idle'),
                          key=lambda v: nx.shortest_path_length(self.graph, v['pos'], source, weight='weight'), default=None)
            if not vehicle:
                print(f"No available vehicle for task {task_id}")
                continue

            start_time = max(time_window[0], 0)
            result = self.improve_A_star(vehicle['pos'], source, vehicle['id'], task['priority'], time_window, start_time)
            if not result:
                print(f"No path from {vehicle['pos']} to {source} for task {task_id}")
                continue
            path_to_source, _, times_to_source = result

            result = self.improve_A_star(source, target, vehicle['id'], task['priority'], time_window, times_to_source[-1])
            if not result:
                print(f"No path from {source} to {target} for task {task_id}")
                continue
            path_to_target, cost, times_to_target = result

            full_path = path_to_source[:-1] + path_to_target
            full_times = times_to_source[:-1] + times_to_target
            self.paths[vehicle['id']] = {'path': full_path, 'times': full_times}

            # 更新占用信息
            for i in range(len(full_path) - 1):
                edge = tuple(sorted([full_path[i], full_path[i + 1]]))
                self.edge_occupancy[edge].append((full_times[i], full_times[i + 1], vehicle['id']))
                self.node_occupancy[full_path[i]].append((full_times[i], full_times[i], vehicle['id']))
            self.node_occupancy[full_path[-1]].append((full_times[-1], full_times[-1], vehicle['id']))

            vehicle['pos'] = target
            vehicle['status'] = 'busy'

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PathVisualizer:
    def __init__(self, graph, planner):
        """
        初始化路径可视化器。

        :param graph: NetworkX图
        :param planner: MultiVehiclePathPlanner实例
        """
        self.graph = graph
        self.planner = planner
        self.location = nx.get_node_attributes(graph, 'location')

    def plot_paths(self):
        """
        绘制AGV路径的三维静态图，z轴为时间。
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        original_xs = [self.location[node][0] for node in self.graph.nodes]
        original_ys = [self.location[node][1] for node in self.graph.nodes]
        max_y = max(original_ys)
        xs = original_xs
        ys = [max_y - y for y in original_ys]
        zs = [0] * len(xs)
        ax.scatter(xs, ys, zs, c='gray', s=10, alpha=0.15, label='Nodes')

        ax.set_xlim(min(xs) - 2, max(xs) + 2)
        ax.set_ylim(min(ys) - 2, max(ys) + 2)

        colors = ['r', 'b', 'g', 'y', 'm', 'c']
        for i, (vid, path_data) in enumerate(self.planner.paths.items()):
            path = path_data['path']
            times = path_data['times']
            path_xs = [self.location[node][0] for node in path]
            path_ys = [self.location[node][1] for node in path]
            path_ys = [max_y - y for y in path_ys]
            path_zs = times
            ax.plot(path_xs, path_ys, path_zs, c=colors[i % len(colors)], label=f'四向穿梭车ID：{vid}', linewidth=5)
            ax.scatter(path_xs, path_ys, path_zs, c=colors[i % len(colors)], s=40, alpha=0.6)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Time')
        ax.set_title('Paths with Time Axis')
        ax.legend()
        plt.show()

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
        ax.set_title('Paths with Time Axis')
        ax.legend()

        return fig  # 返回图形对象

from Program.DataModel import Model

# 假设全局变量
heights = [3.15, 2.55, 1.5]
cross_nodes = {1: [642, 674, 1116, 1148], 2: [2374, 2406, 2844, 2876], 3: [3899, 4135]}

# 创建示例图
model = Model()
G = model.combined_graph

# 初始化车辆和任务（追尾场景）
vehicles = [
    {'id': 1, 'pos': 833, 'status': 'idle'},  # 车辆 1 先出发
    {'id': 2, 'pos': 834, 'status': 'idle'}   # 车辆 2 后出发
]
tasks = [
    {'id': 1, 'source': 833, 'target': 820, 'priority': 1, 'time_window': (0, 1000), 'agv_id': 1},  # 车辆 1
    {'id': 2, 'source': 834, 'target': 820, 'priority': 1, 'time_window': (0, 1000), 'agv_id': 2}   # 车辆 2
]
# 运行规划
planner = MultiVehiclePathPlanner(G, vehicles, tasks, cross_nodes)
planner.plan_paths()

# 输出最终路径和时间
print("\nFinal Paths and Times:")
for vid, data in planner.paths.items():
    print(f"Vehicle {vid}: Path = {data['path']}, Times = {data['times']}")

# 显示结果
viz = PathVisualizer(G, planner)
viz.plot_paths()