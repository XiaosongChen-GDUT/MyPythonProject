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

# 1. 多车路径规划算法类,Grok生成的代码
class MultiVehiclePathPlanner:
    def __init__(self, graph, vehicles, tasks):
        """
        初始化多车路径规划器。

        :param graph: NetworkX加权无向图，包含节点位置和权重
        :param vehicles: 车辆列表，格式 [{'id': 1, 'pos': node_id, 'status': 'idle'}, ...]
        :param tasks: 任务列表，格式 [{'id': 1, 'source': node, 'target': node, 'priority': int, 'time_window': (start, end)}, ...]
        """
        self.graph = graph
        self.vehicles = vehicles  # 车辆状态
        self.tasks = tasks  # 任务列表
        self.cross_nodes = cross_nodes  # 换层节点
        self.edge_occupancy = defaultdict(list)  # 边占用时间表，每个记录包含 (开始时间, 结束时间, 车辆ID)格式 {edge: [(start_time, end_time), ...]}
        self.node_occupancy = defaultdict(list)  # 节点占用时间表 {node: [(start_time, end_time, vehicle_id), ...]}
        self.paths = {}  # 存储每辆车的路径，格式 {vehicle_id: {'path': [nodes], 'times': [times]}}
        self.MAX_WAIT_TIME = 5.0  # 最大等待时间阈值，可调整

        # 记录车辆初始位置占用
        for vehicle in vehicles:
            task = next((t for t in tasks if t['agv_id'] == vehicle['id']), None)
            start_time = task['time_window'][0] if task else 0
            self.node_occupancy[vehicle['pos']].append((0, start_time, vehicle['id']))
            print(f"Vehicle {vehicle['id']} initial occupancy at node {vehicle['pos']} until {start_time}")

    def improve_A_star(self, source, target, vehicle_id, task_priority, time_window, start_time=0):
        """
        改进A*算法，考虑时间窗、优先级和冲突。

        :param source: 起点节点
        :param target: 终点节点
        :param vehicle_id: 当前车辆ID
        :param task_priority: 任务优先级
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
                return path, round(nx.path_weight(self.graph, path, 'weight'), 2), times

            if current_node in explored:
                continue

            # explored[current_node] = (parent, current_time)
            # for neighbor, w in self.graph.adj[current_node].items():
            #     cost = w['weight']
            #     turn_cost = self.turn_cost(location, parent, current_node, neighbor)
            #     ncost = dist_current_node + cost + turn_cost
            #     arrival_time = current_time + cost
            #
                # # 检查时间窗
                # if arrival_time > time_window[1]:
                #     continue

                # # 检查冲突
                # edge = tuple(sorted([current_node, neighbor]))
                # conflict = self.check_conflict(edge, current_node, neighbor, arrival_time - cost, arrival_time, vehicle_id, task_priority)
                # if conflict:
                #     conflict_type, other_vehicle_id, other_priority = conflict
                #     wait_time, detour_node = self.resolve_conflict(conflict_type, current_node, neighbor, vehicle_id, task_priority, other_vehicle_id, other_priority, arrival_time)
                #     if wait_time > 0:  # 等待
                #         arrival_time += wait_time
                #     elif detour_node:  # 绕行
                #         detour_result = self.improve_A_star(detour_node, target, vehicle_id, task_priority, time_window, arrival_time)
                #         if detour_result:
                #             detour_path, detour_cost, detour_times = detour_result
                #             path_to_detour = [current_node, detour_node]
                #             times_to_detour = [current_time, arrival_time]
                #             return path_to_detour[:-1] + detour_path, ncost + detour_cost, times_to_detour[:-1] + detour_times
                #         continue
                #
                # # 安全距离检查
                # if not self.check_safety_distance(current_node, neighbor, vehicle_id, arrival_time - cost, arrival_time):
                #     continue

            explored[current_node] = (parent, current_time)
            for neighbor, w in self.graph.adj[current_node].items():
                cost = w['weight']
                turn_cost = self.turn_cost(location, parent, current_node, neighbor)
                ncost = dist_current_node + cost + turn_cost
                arrival_time = current_time + cost
                # # 检查冲突
                edge = tuple(sorted([current_node, neighbor]))
                conflict = self.check_conflict(edge, current_node, neighbor, arrival_time - cost, arrival_time, vehicle_id, task_priority)
                if conflict:
                    conflict_type, other_vehicle_id, other_priority = conflict
                    wait_time, detour_node = self.resolve_conflict(conflict_type, current_node, neighbor, vehicle_id, task_priority, other_vehicle_id, other_priority, arrival_time)
                    if wait_time > 0:  # 选择等待
                        print(f"Vehicle {vehicle_id}: Waiting {wait_time} units at node {current_node} due to {conflict_type} conflict with Vehicle {other_vehicle_id}")
                        arrival_time += wait_time
                    elif detour_node:  # 选择绕行
                        print(f"Vehicle {vehicle_id}: Detouring to node {detour_node} due to {conflict_type} conflict with Vehicle {other_vehicle_id}")
                        detour_result = self.improve_A_star(detour_node, target, vehicle_id, task_priority, time_window, arrival_time)
                        if detour_result:
                            detour_path, detour_cost, detour_times = detour_result
                            path_to_detour = [current_node, detour_node]
                            times_to_detour = [current_time, arrival_time]
                            return path_to_detour[:-1] + detour_path, ncost + detour_cost, times_to_detour[:-1] + detour_times
                        continue
                    else:  # 无等待或绕行方案，跳过此邻居
                        # print(f"Vehicle {vehicle_id}: No viable solution for {conflict_type} conflict with Vehicle {other_vehicle_id}, skipping neighbor {neighbor}")
                        # continue
                        print(f"Vehicle {vehicle_id}: No detour available for {conflict_type} conflict, waiting {wait_time} units")
                        arrival_time += wait_time  # 默认等待，即使时间为0也继续
                # #安全距离检查
                # if not self.check_safety_distance(current_node, neighbor, vehicle_id, arrival_time - cost, arrival_time):
                #     print(f"Vehicle {vehicle_id}: Safety distance violated with another vehicle at edge {edge}")
                #     continue
                h = heuristic(neighbor, target)
                D = heuristic(source, current_node) or 1
                beta = math.e ** (h / D)
                h = beta * h
                # h *= 1.2

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
        :return: 拐点成本（浮点数）
        """
        if parent is None:
            return 0
        dx1 = location[current][0] - location[parent][0]
        dy1 = location[current][1] - location[parent][1]
        dx2 = location[neighbor][0] - location[current][0]
        dy2 = location[neighbor][1] - location[current][1]
        if dx1 * dx2 + dy1 * dy2 < 0:  # 方向相反，增加成本
            return 1.2  # 示例成本，可调整
        return 0

    def check_conflict(self, edge, current_node, neighbor, start_time, end_time, vehicle_id, priority):
        """
        检查路径冲突类型。
        :param edge: 检查的边 (node1, node2)，无向图中为排序后的元组
        :param current_node: 当前节点
        :param neighbor: 邻居节点（目标节点）
        :param start_time: 当前车辆进入边的时间
        :param end_time: 当前车辆离开边的时间
        :param vehicle_id: 当前车辆的ID
        :param priority: 当前任务的优先级
        :return: (conflict_type, other_vehicle_id, other_priority) 或 None
        """
        # 检查边冲突
        for occ_start, occ_end, occ_vehicle_id in self.edge_occupancy[edge]:
            #检查当前车辆在进入和离开一条边的时间段内，是否与占用同一条边的其他车辆发生时间上的重叠，从而判断是否存在冲突。
            if occ_vehicle_id != vehicle_id and not (end_time <= occ_start or start_time >= occ_end):
                other_priority = next((t['priority'] for t in self.tasks if t['agv_id'] == occ_vehicle_id), 0)
                print(f"Vehicle {vehicle_id}: Potential conflict detected on edge {edge} with Vehicle {occ_vehicle_id}, time overlap [{start_time}, {end_time}] vs [{occ_start}, {occ_end}]")

                # 相向冲突
                for path_data in self.paths.values():
                    path = path_data['path']
                    times = path_data['times']
                    for i in range(len(path) - 1):
                        if path[i] == neighbor and path[i + 1] == current_node and times[i] <= end_time and times[i + 1] >= start_time:
                            print(f"Vehicle {vehicle_id}: Opposite conflict confirmed with Vehicle {occ_vehicle_id}")
                            return ('opposite', occ_vehicle_id, other_priority)

                # 节点冲突
                if any(n == current_node or n == neighbor for n, times in self.node_occupancy.items() if any(t[0] <= end_time and t[1] >= start_time for t in times)):
                    print(f"Vehicle {vehicle_id}: Node conflict confirmed with Vehicle {occ_vehicle_id} at node {current_node if current_node in self.node_occupancy else neighbor}")
                    return ('node', occ_vehicle_id, other_priority)

                # 追尾冲突
                for path_data in self.paths.values():
                    path = path_data['path']
                    times = path_data['times']
                    for i in range(len(path) - 1):
                        if path[i] == current_node and path[i + 1] == neighbor and times[i] <= end_time and times[i + 1] >= start_time:
                            print(f"Vehicle {vehicle_id}: Rear-end conflict confirmed with Vehicle {occ_vehicle_id}")
                            return ('rear_end', occ_vehicle_id, other_priority)
        #检查节点冲突和安全距离  检查当前车辆计划经过的两个节点（current_node 和 neighbor）是否已被其他车辆占用，并且时间上是否存在冲突。
        for node in [current_node, neighbor]:
            for occ_start, occ_end, occ_vehicle_id in self.node_occupancy[node]:
                if occ_vehicle_id != vehicle_id and not (end_time <= occ_start or start_time >= occ_end):
                    other_priority = next((t['priority'] for t in self.tasks if t['agv_id'] == occ_vehicle_id), 0)
                    print(f"Vehicle {vehicle_id}: Node conflict at {node} with Vehicle {occ_vehicle_id}")
                    return ('node', occ_vehicle_id, other_priority)

        # 检查安全距离
        for vid, path_data in self.paths.items():
            path = path_data['path']
            times = path_data['times']
            for i in range(len(path) - 1):
                if times[i] <= end_time and times[i + 1] >= start_time:
                    dist = nx.shortest_path_length(self.graph, current_node, path[i])
                    if dist < 2:
                        other_priority = next((t['priority'] for t in self.tasks if t['agv_id'] == vid), 0)
                        print(f"Vehicle {vehicle_id}: Safety distance violated with Vehicle {vid} at edge {edge}, distance {dist}")
                        return ('safety_violation', vid, other_priority)
        return None

    def resolve_conflict(self, conflict_type, current_node, neighbor, vehicle_id, priority, other_vehicle_id, other_priority, arrival_time):
        """解决冲突，优先等待"""
        wait_time = self.get_wait_time(current_node, arrival_time, other_vehicle_id)
        detour_cost = None
        detour_node = None

        for detour in self.graph.neighbors(current_node):
            if detour != neighbor and not self.check_conflict(tuple(sorted([current_node, detour])), current_node, detour, arrival_time, arrival_time + 1, vehicle_id, priority):
                detour_cost = nx.shortest_path_length(self.graph, detour, neighbor, weight='weight')
                detour_node = detour
                break

        if priority < other_priority or conflict_type == 'safety_violation':
            if wait_time <= self.MAX_WAIT_TIME and (detour_cost is None or wait_time < detour_cost):
                print(f"Vehicle {vehicle_id}: Priority {priority} <= {other_priority}, waiting {wait_time} units")
                return wait_time, None
            elif detour_node:
                print(f"Vehicle {vehicle_id}: Priority {priority} <= {other_priority}, detouring to {detour_node}")
                return 0, detour_node
            else:
                print(f"Vehicle {vehicle_id}: Priority {priority} <= {other_priority}, forced to wait {wait_time}")
                return wait_time, None
        elif priority > other_priority:
            print(f"Vehicle {vehicle_id}: Priority {priority} > {other_priority}, proceeding")
            return 0, None
        else:
            if random.choice([True, False]):
                print(f"Vehicle {vehicle_id}: Priority equal, proceeding")
                return 0, None
            else:
                print(f"Vehicle {vehicle_id}: Priority equal, waiting {wait_time}")
                return wait_time, None
    # def resolve_conflict(self, conflict_type, current_node, neighbor, vehicle_id, priority, other_vehicle_id, other_priority, arrival_time):
    #     """
    #     根据冲突类型和优先级处理冲突。
    #
    #     :return: (wait_time, detour_node) - 等待时间或绕行节点
    #     """
    #     if priority < other_priority:
    #         if conflict_type == 'opposite':  # 相向冲突，绕行
    #             for detour in self.graph.neighbors(current_node):
    #                 if detour != neighbor and not self.check_conflict(tuple(sorted([current_node, detour])), current_node, detour, arrival_time, arrival_time + 1, vehicle_id, priority):
    #                     print(f"相向冲突，绕行 车辆{vehicle_id} 到 {detour}")
    #                     return 0, detour
    #             return 0, None  # 无绕行路径，等待后续处理
    #         elif conflict_type == 'node':  # 节点冲突，等待
    #             wait_time = self.get_wait_time(current_node, arrival_time, other_vehicle_id)
    #             print(f"节点冲突，车辆 {vehicle_id} 等待 {wait_time}")
    #             return wait_time, None
    #         elif conflict_type == 'rear_end':  # 追尾冲突，重新规划
    #             print(f"追尾冲突，重新规划 {vehicle_id}")
    #             return 0, None  # 交给外层重新规划
    #     elif priority > other_priority:
    #         return 0, None  # 高优先级继续行驶
    #     else:  # 优先级相等，随机选择
    #         if random.choice([True, False]):
    #             return 0, None  # 当前车辆优先
    #         else:
    #             if conflict_type == 'opposite':
    #                 for detour in self.graph.neighbors(current_node):
    #                     if detour != neighbor and not self.check_conflict(tuple(sorted([current_node, detour])), current_node, detour, arrival_time, arrival_time + 1, vehicle_id, priority):
    #                         print(f"相向冲突，车辆 {vehicle_id} 绕行到 {detour}")
    #                         return 0, detour
    #                 return 0, None
    #             elif conflict_type == 'node':
    #                 wait_time = self.get_wait_time(current_node, arrival_time, other_vehicle_id)
    #                 print(f"优先级相等, 节点冲突，车辆 {vehicle_id} 等待 {wait_time}")
    #                 return wait_time, None
    #             elif conflict_type == 'rear_end':
    #                 return 0, None

    def get_wait_time(self, node, arrival_time, other_vehicle_id):
        """计算等待时间直到节点或边空闲"""
        max_end_time = 0
        for start, end, vid in self.node_occupancy[node]:
            if vid != other_vehicle_id and arrival_time < end:
                max_end_time = max(max_end_time, end)
        for edge in self.graph.edges(node):
            for start, end, vid in self.edge_occupancy[tuple(sorted(edge))]:
                if vid != other_vehicle_id and arrival_time < end:
                    max_end_time = max(max_end_time, end)
        return max(0, max_end_time - arrival_time)

    def check_safety_distance(self, current_node, neighbor, vehicle_id, start_time, end_time):
        """检查两车之间是否保持两个节点的安全距离"""
        for path_data in self.paths.values():
            path = path_data['path']
            times = path_data['times']
            for i in range(len(path) - 1):
                if times[i] <= end_time and times[i + 1] >= start_time:
                    dist = nx.shortest_path_length(self.graph, current_node, path[i])
                    if dist < 2:  # 小于2个节点距离
                        return False
        return True

    def plan_paths(self):
        """
        为所有车辆规划路径，考虑优先级和冲突。
        """
        task_queue = [(-task['priority'], task['id'], task) for task in self.tasks]  # 负数使高优先级在前
        heapq.heapify(task_queue)
        # 2. 遍历任务队列，为每辆车辆规划路径
        while task_queue:
            # 取出优先级最高的任务
            _, task_id, task = heapq.heappop(task_queue)
            source, target = task['source'], task['target']
            time_window = task['time_window']
            agv_id = task.get('agv_id')
            # 3. 选择车辆 优先选择任务指定的空闲车辆（agv_id 匹配且 status == 'idle'）。
            # 如果没有指定车辆或指定车辆不可用，则选择距离任务起点（source）最近的空闲车辆。
            vehicle = next((v for v in self.vehicles if v['id'] == agv_id and v['status'] == 'idle'), None) or \
                      min((v for v in self.vehicles if v['status'] == 'idle'),
                          key=lambda v: nx.shortest_path_length(self.graph, v['pos'], source, weight='weight'), default=None)
            if not vehicle:
                print(f"No available vehicle for task {task_id}")
                continue
            print(f"Planning for Vehicle {vehicle['id']} on Task {task_id}: {vehicle['pos']} -> {source} -> {target}")
            # 4. 规划从车辆当前位置到任务起点的路径
            start_time = max(time_window[0], 0)
            result = self.improve_A_star(vehicle['pos'], source, vehicle['id'], task['priority'], time_window, start_time)
            if not result:
                print(f"车辆到任务起点路径规划失败，车辆位置： {vehicle['pos']} to {source} for task {task_id}")
                continue
            path_to_source, _, times_to_source = result
            # 5. 规划从任务起点到目标的路径
            result = self.improve_A_star(source, target, vehicle['id'], task['priority'], time_window, times_to_source[-1])
            if not result:
                print(f"车辆到任务目标路径规划失败，车辆位置： {source} to {target} for task {task_id}")
                continue
            path_to_target, cost, times_to_target = result
            # # 6. 验证时间窗
            # if times_to_target[-1] > time_window[1]:
            #     print(f"Task {task_id} exceeds time window {time_window}")
            #     continue
            # 7. 合并路径和时间表
            full_path = path_to_source[:-1] + path_to_target
            full_times = times_to_source[:-1] + times_to_target
            self.paths[vehicle['id']] = {'path': full_path, 'times': full_times}
            print(f"Vehicle {vehicle['id']}: Path planned - {full_path}, Times: {full_times}")
            # 更新占用信息
            for i in range(len(full_path) - 1):
                edge = tuple(sorted([full_path[i], full_path[i + 1]]))
                self.edge_occupancy[edge].append((full_times[i], full_times[i + 1], vehicle['id']))
                self.node_occupancy[full_path[i]].append((full_times[i], full_times[i], vehicle['id']))
            self.node_occupancy[full_path[-1]].append((full_times[-1], full_times[-1], vehicle['id']))
            # 9. 更新车辆状态
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

# 示例运行
if __name__ == '__main__':
    # 创建示例图（假设已有）
    model = Model()
    G = model.combined_graph

    # 初始化车辆和任务
    vehicles = [{'id': 1, 'pos': 1300, 'status': 'idle'}, {'id': 2, 'pos': 1295, 'status': 'idle'}]
    tasks = [
        {'id': 1, 'source': 1300, 'target': 1312, 'priority': 2, 'time_window': (3.8, 1000), 'agv_id': 1},
        {'id': 2, 'source': 1295, 'target': 1312, 'priority': 1, 'time_window': (0, 1000), 'agv_id': 2}
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
