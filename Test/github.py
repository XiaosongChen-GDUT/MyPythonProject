import os
from concurrent.futures import ProcessPoolExecutor, as_completed  # 确保导入正确的as_completed

import numpy as np
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # 可选：进度条
from scipy.sparse import dok_matrix
from Program.DataModel import Model
from Algorithm.PathPlanning import Path_Planning
import numpy as np
import geatpy as ea
from NSGA2.FS5.FS5_Individual import Individual
import random
import pickle
import networkx as nx
enter_node = [51, 348, 636, 925, 1110, 1620]# 入口点
from NSGA2.FS4.FS4_Individual import Individual
from NSGA2.FS4.FS4_Population import Population
from NSGA2.FS4.FS4_Utils import FS_Utils
# 模拟真实 Pareto 前沿（假设已知）
# true_PF = []
# for i in range(10):
#     ind = Individual()
#     ind.objectives = [0.5 + i * 0.1, 50.0 - i * 2, 5 - i * 0.2]
#     true_PF.append(ind)
#
# FS_Utils.save_pareto_front(front=true_PF ,true_pf_file="pareto_fronts.csv",run_id=1)
#
# pf = FS_Utils.load_true_pareto_front(true_pf_file="pareto_fronts.csv")
# print(pf)


# max_gen = 200
# for i in range(max_gen):
#     # 自适应交叉概率
#     current_gen = i
#     scale = 2.0
#     x = scale * (1 - 2 * current_gen / max_gen)
#     # 自适应交叉概率
#     crossover_prob = 1 / (1 + np.exp(-x))  # 0.05-0.95
#     # 自适应变异概率
#     mutation_prob = 0.05 + (0.3 - 0.05) * (1 / (1 + np.exp(-x)))  # 0.05-0.3
#
#     print("crossover_prob:", crossover_prob)
#     print("mutation_prob:", mutation_prob)



class YourClass:
    def __init__(self,Model,Path_Planning):
        self.model = Model
        self.path_planning = Path_Planning
        self.TopoGraph = Model.combined_graph
        self.path_cache = {}  # 路径缓存 {(enter, target): (length, time)}
        # print(os.cpu_count())   #12核
        self._precompute_paths()  # 初始化时预计算
        '''
        path_cache = {
                        "入口A": {
                            "货位1": (path_length, time_cost),
                            "货位2": (path_length, time_cost),
                            ...
                        },
                        "入口B": {
                            "货位1": (path_length, time_cost),
                            ...
                        }
                    }
        '''
    # def _precompute_paths(self):
    #     """并行预计算所有路径"""
    #     # enter_nodes = self.get_all_enter_nodes()
    #     target_nodes = self.TopoGraph.nodes()
    #     args = [(enter, target) for enter in enter_node for target in target_nodes]
    #
    #     # 进度条（可选）
    #     print(f"预计算路径总数: {len(args)}")
    #     progress = tqdm(total=len(args), desc="路径预计算")
    #
    #     # 多进程并行计算
    #     with ProcessPoolExecutor(max_workers=4) as executor:
    #         futures = []     # 收集 future 句柄
    #
    #         for enter, target in args:  # 遍历所有路径参数
    #             future = executor.submit(    # 提交任务
    #                 self._compute_single_path, enter, target
    #             )
    #             future.add_done_callback(lambda _: progress.update(1))  # 进度条更新
    #             futures.append(future)  # 收集 future 句柄
    #
    #         # 收集结果
    #         for future in futures:
    #             enter, target = future.result()[0]  # 传入参数
    #             pl, tc = future.result()[1]         # 计算结果
    #             self.path_cache[enter][target] = (pl, tc)  # 关键修改点
    #     progress.close()
    def _precompute_paths(self):
        # 尝试从文件加载缓存
        try:
            with open("path_cache.pkl", "rb") as f:
                self.path_cache = pickle.load(f)
            print("从文件加载路径缓存")
        except FileNotFoundError:
            status = nx.get_node_attributes(self.TopoGraph, 'status')
            targets = [target for target in status if status[target] == 0]
            self.path_cache = {}
            for enter in enter_node:
                self._precompute_for_one_enter(enter, targets, batch_size=100)
            # 保存缓存到文件
            with open("path_cache.pkl", "wb") as f:
                pickle.dump(self.path_cache, f)
            print("路径缓存已保存到文件")

    #将目标节点分批提交到进程池，减少同时运行的进程数量，提高效率
    def _precompute_for_one_enter(self, enter, targets, batch_size=100):
        # 确保 self.path_cache[enter] 存在
        if enter not in self.path_cache:
            self.path_cache[enter] = {}

        # 初始化进度条
        progress = tqdm(total=len(targets), desc=f"预计算入口 {enter}", unit="path")
        targets = list(targets)  # 确保 targets 是列表

        # 分批处理
        for i in range(0, len(targets), batch_size):
            batch_targets = targets[i:i + batch_size]
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._compute_single_path, enter, target)
                           for target in batch_targets]
                for future in as_completed(futures):
                    (_, target), (pl, tc) = future.result()
                    self.path_cache[enter][target] = (pl, tc)
                    progress.update(1)
        progress.close()

    def _compute_single_path(self, enter, target):
        """单次路径计算（适配多进程调用）"""
        try:
            # 计算路径
            path, path_length, _ = self.path_planning.improve_A_star(self.TopoGraph, enter, target )
            # 计算时间
            time_cost = round(self.path_planning.cal_path_time(self.TopoGraph, path), 2)
            return (enter, target), (path_length, time_cost)
        except Exception as e:
            print(f"路径计算失败：从 {enter} 到 {target}，错误：{str(e)}")
            return (enter, target), (float('inf'), float('inf'))

if __name__ == '__main__':
    model = Model()
    path_planning = Path_Planning(model)
    YourClass = YourClass(model,path_planning)
