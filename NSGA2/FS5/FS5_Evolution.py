from collections import deque

import numpy as np
from geatpy.core.indicator import IGD, HV  # 导入IGD指标计算工具
# from geatpy.core.indicator import IGD
# from pymoo.indicators.hv import HV    # 导入Pymoo的HV指标计算工具
from tqdm import tqdm       # 进度条工具

from NSGA2.FS5.FS5_Utils import FS_Utils
from NSGA2.FS5.FS5_Population import Population


class Evolution:

    def __init__(self, problem, num_of_generations=500, num_of_individuals=100, num_of_tour_particips=2,
                 tournament_prob=0.9, mutation_param=0.01, use_threshold_flag=True,
                 crossover_type="two_point", mutation_type="swap", init_type="random"):
        """
        初始化进化类，添加可配置的交叉、变异和种群初始化方法。

        :param problem: 问题实例
        :param num_of_generations: 最大迭代次数
        :param num_of_individuals: 种群大小
        :param num_of_tour_particips: 锦标赛参与个体数
        :param tournament_prob: 锦标赛选择概率
        :param mutation_param: 变异概率
        :param use_threshold_flag: 是否使用阈值停止条件
        :param crossover_type: 交叉类型
        :param mutation_type: 变异类型
        :param init_type: 种群初始化类型
        """
        self.utils = FS_Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob,
                              mutation_param, crossover_type, mutation_type, init_type)
        self.use_threshold_flag = use_threshold_flag
        self.population = None
        self.num_of_generations = num_of_generations
        self.num_of_individuals = num_of_individuals
        self.igd_history = []
        self.hv_history = []
        # 记录每代 Pareto 前沿的目标函数最优值
        self.objective_history = {'F1': [], 'F2': [], 'F3': []}
        #读取真实pareto前沿返回个体列表，个体有目标函数值
        self.true_pareto_front = FS_Utils.load_true_pareto_front(true_pf_file="pareto_fronts.csv")



    def calculate_IGD(self, ObjV, PF):
        """
            indicator.IGD : function - 计算多目标优化反转世代距离(IGD)评价指标的值
            语法:
            igd = IGD(ObjV, PF)
            igd = IGD(ObjV, PF, Parallel)
        描述:
        IGD是一个综合评价指标，其值越小越好。
        输入参数:
        ObjV : array - 目标函数值矩阵。
        PF   : array - 真实全局帕累托最优解的目标函数值矩阵，
        注意: 实际上可以传入不是真实PF的目标函数值矩阵，
        可以是经验所得的非支配解的目标函数值。
        Parallel : bool  - (可选参数)表示是否采用并行计算，缺省或为None时默认为False。
        输出参数:
        igd : float - IGD评价指标。
        """
        pf_objects = np.array([ind.objectives for ind in PF])
        # 检查输入是否为空
        if pf_objects.size == 0 or ObjV.size == 0:
            return 0.0
        # 确保 ObjV 和 pf_objects 是二维数组
        ObjV = np.atleast_2d(ObjV)
        pf_objects = np.atleast_2d(pf_objects)

        # 计算全局最小值和最大值
        # 综合考虑 ObjV 和 pf_objects，计算每个目标的最小值和最大值
        objectives_min = np.min(ObjV, axis=0)
        objectives_max = np.max(ObjV, axis=0)
        pf_min = np.min(pf_objects, axis=0)
        pf_max = np.max(pf_objects, axis=0)

        global_min = np.minimum(objectives_min, pf_min)
        global_max = np.maximum(objectives_max, pf_max)
        # 避免除以 0
        range_ = global_max - global_min
        range_[range_ == 0] = 1.0  # 如果最大值等于最小值，设范围为 1，避免除以 0

        # 归一化 ObjV 和 pf_objects
        normalized_ObjV = (ObjV - global_min) / range_
        normalized_pf = (pf_objects - global_min) / range_
        igd = IGD(normalized_ObjV, normalized_pf)
        return igd

    #超体积
    def calculate_hv(self, ObjV,pareto_front):
        """
         语法:
        hv = HV(ObjV)
        hv = HV(ObjV, PF)
        hv = HV(ObjV, PF, Parallel)
    输入参数:
        ObjV : array - 目标函数值矩阵。
        PF   : array - 真实全局帕累托最优解的目标函数值矩阵，
                       注意: 实际上可以传入不是真实PF的目标函数值矩阵，
                       可以是经验所得的非支配解的目标函数值。
        Parallel : bool  - (可选参数)表示是否采用并行计算，缺省或为None时默认为False。
    输出参数:
        hv : float - 多目标优化问题的全局历史最优值。
        :param pareto_front:
        :return:
        """
        # 提取第一层 Pareto 前沿的目标值
        pf_objects = np.array([ind.objectives for ind in pareto_front])
        # 检查输入是否为空
        if pf_objects.size == 0 or ObjV.size == 0:
            return 0.0

        # 确保 ObjV 和 pf_objects 是二维数组
        ObjV = np.atleast_2d(ObjV)
        pf_objects = np.atleast_2d(pf_objects)

        # 检查列数是否匹配
        if ObjV.shape[1] != pf_objects.shape[1]:
            raise ValueError(f"ObjV 和 PF 的列数必须相同，当前 ObjV 列数为 {ObjV.shape[1]}，PF 列数为 {pf_objects.shape[1]}")

        # 归一化目标值
        objectives_min = np.min(ObjV, axis=0)
        objectives_max = np.max(ObjV, axis=0)
        pf_min = np.min(pf_objects, axis=0)
        pf_max = np.max(pf_objects, axis=0)

        global_min = np.minimum(objectives_min, pf_min)
        global_max = np.maximum(objectives_max, pf_max)

        # 避免除以 0
        range_ = global_max - global_min
        range_[range_ == 0] = 1.0  # 如果最大值等于最小值，设范围为 1

        # 归一化 ObjV 和 pf_objects
        normalized_ObjV = (ObjV - global_min) / range_
        normalized_pf = (pf_objects - global_min) / range_

        # 调整 PF 的最大值，确保参考点大于 ObjV 的最大值
        max_objv = np.max(normalized_ObjV, axis=0)
        adjusted_pf = normalized_pf.copy()
        for i in range(adjusted_pf.shape[1]):  # 对每个目标
            adjusted_pf[:, i] = np.maximum(adjusted_pf[:, i], max_objv[i] * 1.1)  # 确保 PF 的值大于 ObjV 的最大值

        # 调用 geatpy 的 HV 函数
        hv = HV(normalized_ObjV, adjusted_pf)
        return hv


    def threshold_calculator(self, now_pare_to_front, last_pare_to_front):
        num_objectives = 3  # 目标数量

        # 存储当前和上一代的目标值
        objectives_now = [[] for _ in range(num_objectives)]
        objectives_last = [[] for _ in range(num_objectives)]

        # 填充数据
        for j in now_pare_to_front:
            for i in range(num_objectives):
                objectives_now[i].append(j.objectives[i])
        for j in last_pare_to_front:
            for i in range(num_objectives):
                objectives_last[i].append(j.objectives[i])

        # 归一化参数
        norm_params = []
        for i in range(num_objectives):
            f_max = max(objectives_now[i])
            f_min = min(objectives_now[i])
            norm_params.append((f_min, f_max))

        # 归一化处理
        normalized_now = []
        for j in now_pare_to_front:
            norm_point = []
            for i in range(num_objectives):
                f_min, f_max = norm_params[i]
                if f_max == f_min:
                    norm_val = 0.0
                else:
                    norm_val = (j.objectives[i] - f_min) / (f_max - f_min)
                norm_point.append(norm_val)
            normalized_now.append(norm_point)

        normalized_last = []
        for j in last_pare_to_front:
            norm_point = []
            for i in range(num_objectives):
                f_min, f_max = norm_params[i]
                if f_max == f_min:
                    norm_val = 0.0
                else:
                    norm_val = (j.objectives[i] - f_min) / (f_max - f_min)
                norm_point.append(norm_val)
            normalized_last.append(norm_point)

        # 计算最大差异度
        differences = []
        for i in range(num_objectives):
            f_max_last = max(objectives_last[i])
            f_min_last = min(objectives_last[i])
            f_max_now = norm_params[i][1]
            f_min_now = norm_params[i][0]

            if (f_max_now - f_min_now) == 0:
                diff_max = 0
                diff_min = 0
            else:
                diff_max = (f_max_last - f_max_now) / (f_max_now - f_min_now)
                diff_min = (f_min_last - f_min_now) / (f_max_now - f_min_now)

            differences.append(max(diff_max, diff_min))

        difference_max = max(differences)
        difference_min = min(differences)
        """
        indicator.IGD : function - 计算多目标优化反转世代距离(IGD)评价指标的值
        语法:
            igd = IGD(ObjV, PF)
            igd = IGD(ObjV, PF, Parallel)
        描述:
            IGD是一个综合评价指标，其值越小越好。
        输入参数:
            ObjV : array - 目标函数值矩阵。
            PF   : array - 真实全局帕累托最优解的目标函数值矩阵，
                           注意: 实际上可以传入不是真实PF的目标函数值矩阵，
                           可以是经验所得的非支配解的目标函数值。
            Parallel : bool  - (可选参数)表示是否采用并行计算，缺省或为None时默认为False。"""
        # 计算IGD
        # ind = IGD(np.array(normalized_now))
        # igd = ind(np.array(normalized_last))
        igd = IGD(np.array(normalized_now), np.array(normalized_last),Parallel=True)
        print(f"IGD: {igd}, Shape of now: {np.array(normalized_now).shape}, Shape of last: {np.array(normalized_last).shape}")
        return difference_max, difference_min, igd



    # 动态精英比例函数 前期高比例（加速收敛），后期低比例（提升多样性）
    def dynamic_elite_ratio(self,current_gen, max_gen, initial_ratio=0.1, final_ratio=0.02):
        return initial_ratio - (initial_ratio - final_ratio) * (current_gen / max_gen)

    '''
    初始化种群并计算其目标函数值。
    对种群进行非支配排序，并根据拥挤距离选择个体。
    通过锦标赛选择、交叉和变异生成下一代个体。
    如果启用了阈值停止条件，当连续几代的前沿差异度小于设定阈值时停止进化。
    返回最终的前沿个体集和实际进化代数
    '''
    def evolve(self):
        """
        进化过程，记录每代 HV 和目标函数最优值。
        :return: Pareto 前沿、实际迭代次数、IGD 历史、HV 历史、目标函数历史
        """
        print("开始进化...初始化种群...")
        self.population = self.utils.create_initial_population()    # 初始化种群
        print("种群初始化完成，开始非支配排序...")
        # self.population = self.utils.create_initial_population_good_lattice_points()    # 初始化佳集点种群
        self.utils.ens_ss_nondominated_sort(self.population)          # ENS-SS 非支配排序
        print("非支配排序完成，开始计算拥挤距离...")
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance_pymoo(front)            # 计算拥挤距离
        pare_to_front_of_last_generation = self.population.fronts[0] if self.population.fronts else []  # 记录上一代前沿个体集
        print("拥挤距离计算完成，开始进化...")
        finished_generation_number = 0                             # 实际进化代数
        window_size = 10                                            # 阈值停止条件检查窗口大小
        threshold_flag = False                                     # 阈值停止条件是否满足
        # 初始化滑动窗口
        differences_max_window = deque(maxlen=window_size)
        differences_min_window = deque(maxlen=window_size)
        igds_window = deque(maxlen=window_size)
        # 主循环
        #tqdm是一个Python库，用于在循环中动态显示进度条，使得你可以直观地看到当前循环的进度以及预计的剩余时间。
        for i in tqdm(range(self.num_of_generations)):
            #动态精英比例
            current_elite_ratio = self.dynamic_elite_ratio(
                current_gen=i,
                max_gen=self.num_of_generations,
                initial_ratio=0.1,
                final_ratio=0.02
            )
            elite_size = int(self.num_of_individuals * current_elite_ratio)
            # 交叉变异生成子代（包含移民策略）
            children = self.utils.create_children(self.population,current_gen=i, max_gen=self.num_of_generations)
            self.population.extend(children)                           # 合并父代和子代，将下一代个体加入种群
            # 对合并种群进行非支配排序
            # 必要：合并种群后需要重新排序，以确定新的非支配层级
            self.utils.ens_ss_nondominated_sort(self.population)       # ENS-SS 非支配排序
            # 为所有前沿计算拥挤距离
            # 必要：环境选择需要基于拥挤距离选择个体，必须在排序后计算
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance_pymoo(front)
            new_population = Population()                            # 初始化下一代新种群
            seen_features = set()                                     # 全局去重哈希表
            # ============= 加入精英保留策略 ============
            # 精英保留策略：选择前沿个体作为精英个体，并将其直接加入新种群
            if pare_to_front_of_last_generation and elite_size > 0:
                # 按拥挤距离排序，选择前 elite_size 个个体作为精英
                elite_candidates = sorted(
                    pare_to_front_of_last_generation,
                    key=lambda individual: individual.crowding_distance,
                    reverse=True
                )
                elite_individuals = elite_candidates[:min(elite_size, len(elite_candidates))]
                # 去重逻辑（使用哈希表）
                unique_elite = []
                for ind in elite_individuals:
                    features_tuple = tuple(ind.features)  # 转换为元组以便哈希
                    if features_tuple not in seen_features:
                        seen_features.add(features_tuple)
                        unique_elite.append(ind)
                # 去重结束
                elite_individuals = unique_elite[:min(elite_size, len(unique_elite))]
                # 将精英个体直接加入新种群
                new_population.extend(elite_individuals)
                print(f"- 精英比例：{current_elite_ratio}, 精英个体数量：{elite_size}, 实际添加精英个体：{len(elite_individuals)}")
                # ----------- 精英保留策略结束 ------------------
            #====================环境选择==================
            front_num = 0
            # 环境选择：优先选择前沿层级高的个体，同一前沿内选择拥挤距离大的个体
            remaining_size = self.num_of_individuals - len(new_population)  # 剩余需要填充的个体数
            while (front_num < len(self.population.fronts) and  remaining_size > 0 ):
                    current_front = self.population.fronts[front_num]
                    print("当前前沿编号：", front_num, "当前前沿个体数：", len(self.population.fronts[front_num]), "种群剩余需要填充的个体数：", remaining_size)
                    # 如果当前前沿个体数小于等于剩余空间，直接加入所有个体
                    if len(current_front) <= remaining_size:
                        unique_individuals = []
                        for ind in current_front:
                            features_tuple = tuple(ind.features)
                            if features_tuple not in seen_features:
                                seen_features.add(features_tuple)
                                unique_individuals.append(ind)
                        new_population.extend(unique_individuals)
                        remaining_size -= len(unique_individuals)
                    else:
                        # 当前前沿个体数大于剩余空间，按拥挤距离排序选择
                        # 注意：此处无需重新计算拥挤距离，因为在合并种群排序后已计算
                        current_front.sort(key=lambda individual: individual.crowding_distance, reverse=True)
                        # 选择前 remaining_size 个个体
                        unique_individuals = []
                        for ind in current_front:
                            if len(unique_individuals) >= remaining_size:
                                break
                            features_tuple = tuple(ind.features)
                            if features_tuple not in seen_features:
                                seen_features.add(features_tuple)
                                unique_individuals.append(ind)
                        new_population.extend(unique_individuals)
                        remaining_size -= len(unique_individuals)
                    front_num += 1
            # # 如果新种群数量小于总体数量，随机选择种群中剩余个体加入新种群
            if len(new_population.population) < self.num_of_individuals:
                print(f"新种群数量小于总体数量{remaining_size}，随机生成新种群加入...")
                remaining_size = self.num_of_individuals - len(new_population.individuals)
                remaining_individuals = self.utils.create_immigrants_population(size=remaining_size)
                # remaining_individuals = []
                # for front in self.population.fronts[front_num:]:
                #     for ind in front:
                        # features_tuple = tuple(ind.features)
                        # if features_tuple not in seen_features:
                        # remaining_individuals.append(ind)
                # np.random.shuffle(remaining_individuals)
                # new_population.extend(remaining_individuals[: ])
                new_population.extend(remaining_individuals)
            self.population = new_population                       # 更新种群
            #====================环境选择结束================

            # 环境选择后重新进行非支配排序
            # 必要：环境选择改变了种群组成，需要重新排序以更新 Pareto 前沿
            self.utils.ens_ss_nondominated_sort(self.population)
            # 只为 Pareto 前沿（fronts[0]）重新计算拥挤距离
            # 必要：pare_to_front_of_last_generation 用于下一代精英保留，需要准确的拥挤距离
            # 优化：只计算 fronts[0] 的拥挤距离，减少计算开销
            if self.population.fronts:
                self.utils.calculate_crowding_distance_pymoo(self.population.fronts[0])
            # for front in self.population.fronts:                       # 计算拥挤距离
            #     self.utils.calculate_crowding_distance_pymoo(front)
            # 更新 pare_to_front_of_last_generation
            pare_to_front_of_last_generation = self.population.fronts[0] if self.population.fronts else []
            # 计算平均拥挤距离
            all_crowding_distances = []
            if self.population.fronts:  # 确保 fronts 不为空
                for ind in self.population.fronts[0]:  # 只计算 fronts[0]
                    if hasattr(ind, 'crowding_distance') and ind.crowding_distance is not None:
                        if not np.isinf(ind.crowding_distance):# 排除无效值inf
                            all_crowding_distances.append(ind.crowding_distance)
            average_crowding_distance = np.mean(all_crowding_distances) if all_crowding_distances else 0.0
            # 记录目标函数最优值（取最小值）
            if pare_to_front_of_last_generation:
                objectives = np.array([ind.objectives for ind in pare_to_front_of_last_generation])
                # # 计算当前帕累托前沿的IGD值
                igd = self.calculate_IGD(objectives, self.true_pareto_front)
                self.igd_history.append(igd)
                # 计算当前帕累托前沿的HV值
                hv = self.calculate_hv(objectives,self.true_pareto_front)
                self.hv_history.append(hv)
                print(f"****第 {i+1} 次迭代, IGD: {igd},  HV: {hv},Pareto 前沿平均拥挤距离：{average_crowding_distance}")
                self.objective_history['F1'].append(np.min(objectives[:, 0]))
                self.objective_history['F2'].append(np.min(objectives[:, 1]))
                self.objective_history['F3'].append(np.min(objectives[:, 2]))
            else:
                self.objective_history['F1'].append(float('inf'))
                self.objective_history['F2'].append(float('inf'))
                self.objective_history['F3'].append(float('inf'))
            finished_generation_number = i                           # 记录实际进化代数

            #动态阈值停止条件
            if self.use_threshold_flag and i > 0:  # 从第 1 代开始检查阈值停止条件
                difference_max, difference_min, igd = self.threshold_calculator(
                    now_pare_to_front=self.population.fronts[0],
                    last_pare_to_front=pare_to_front_of_last_generation
                )
                differences_max_window.append(difference_max)
                differences_min_window.append(difference_min)
                igds_window.append(igd)
                # 检查最近 window_size 次迭代的差异
                if len(differences_max_window) == window_size:
                    if (max(differences_max_window) < 0.03 and
                            max(differences_min_window) < 0.01 and
                            max(igds_window) < 0.001):
                        print("满足阈值停止条件，停止进化")
                        break
                else:
                    print("当前前沿差异度 difference_max：", difference_max, " difference_min: ",difference_min, " 与上一代前沿解IGD：", igd)


        # pareto_allocations = [
        #     {"features": ind.features, "objectives": ind.objectives, "allocated_nodes": ind.allocated_nodes}
        #     for ind in pare_to_front_of_last_generation
        # ]
        # 返回最终的前沿个体集和实际进化代数
        return (
            pare_to_front_of_last_generation,    # Pareto 前沿个体集
            finished_generation_number,         # 实际进化代数
            self.igd_history,          # IGD 历史
            self.hv_history,          # HV 历史
            self.objective_history   # 目标函数历史
        )