import os
from copy import deepcopy

import numpy as np
import pandas as pd
from geatpy import etour
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance

from NSGA2.FS5.FS5_Individual import Individual
from NSGA2.FS5.FS5_Population import Population
import random
import geatpy as ea
from deap import tools  # 导入 deap 的工具模块 进行顺序交叉，打乱变异
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class FS_Utils:
    #接收problem实例，设置种群大小，选择参加锦标赛的个体数目，锦标赛的概率，变异概率
    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9, mutation_param=0.01,
                 crossover_type="two_point", mutation_type="swap", init_type="random"):

        self.problem = problem
        self.num_of_individuals = num_of_individuals    # 种群大小
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob  # 锦标赛概率
        self.mutation_param = mutation_param    # 变异概率
        self.crossover_type = crossover_type    # 交叉类型
        self.mutation_type = mutation_type       # 变异类型
        self.init_type = init_type               # 初始化种群类型
        self.num_variables = self.problem.num_of_variables  # 变量个数
        self.aisles_dict = self.problem.aisles_dict  # 货道字典
        self.max_aisle_id = max(self.aisles_dict.keys())  # 最大货道编号
        self.aisle_ids = list(self.aisles_dict.keys())  # 货道编号列表


        self.recOper = ea.Recsbx(XOVR=1, n=20,Parallel=True)  # 生成模拟二进制交叉算子对象
        self.mutOper = ea.Mutpolyn(Pm=1 / self.num_variables, DisI=20)  # 生成多项式变异算子对象
        # 构造 FieldDR
        Nind = self.num_of_individuals  #染色体数
        Lind = self.num_variables  #染色体长度
        self.FieldDR = np.array([[0]*Lind, [self.max_aisle_id-1]*Lind, [1]*Lind])  # 构造 FieldDR
        # lb = np.ones(self.num_variables)  # 下界全为 0
        # ub = np.full(self.num_variables,self.num_variables)  # 上界为 num_of_variables
        # varTypes = np.ones(self.num_variables)  # 变量类型为离散 (1)
        # #varTypes 是一个连续/离散标记 (numpy 的 array 类型的行向量)，表示对应的变量是
        # 连续抑或是离散的（0 表示连续，1 表示离散）。
        # self.FieldDR = np.vstack([lb, ub, varTypes])

    #创建初始种群
    def create_initial_population(self):
        Nind = self.num_of_individuals # 染色体数
        Chrom = ea.crtpp(Nind,FieldDR=self.FieldDR)  # 创建一个排列编码种群染色体矩阵
        print("种群初始化染色体长度：",len(Chrom[0])," 染色体：", Chrom[0]," 开始计算目标函数值....")
        population = Population()
        for i in range(Nind):
            individual = Individual()  # 假设 Individual 是你的个体类
            individual.features = Chrom[i, :].tolist()  # 将每行的数据赋值给个体的染色体属性
            population.append(individual)  # 将个体添加到种群中
            # 计算个体的目标值
            self.problem.calculate_objectives(individual)
        return population
        # for _ in range(self.num_of_individuals):
        #     individual = self.problem.generate_individual()
        #     # print("种群初始化染色体长度：",len(individual.features)," 染色体：", individual.features)
        #     population.append(individual)
        #     # 在种群创建完成后，计算所有个体的目标值
        #     self.problem.calculate_objectives(individual)
        # return population

    #创建移民种群并计算目标值
    def create_immigrants_population(self,size):
        population = Population()
        for _ in range(size):
            individual = self.problem.generate_individual()
            population.append(individual)
            # 在种群创建完成后，计算所有个体的目标值
            self.problem.calculate_objectives(individual)
        return population

    #创建初始种群
    # def create_initial_population(self):
    #     population = Population()  # 创建一个空的种群
    #     available_aisles = list(range(self.max_aisle_id ))  # 所有可用货道编号
    #     existing_features = set()  # 记录已生成的染色体，避免重复
    #
    #     # 生成 num_of_individuals 个个体
    #     for i in range(self.num_of_individuals):
    #         max_attempts = 10  # 最大尝试次数
    #         # for attempt in range(max_attempts):
    #             # 如果染色体长度小于等于可用货道数量，可以完全不重复
    #         if self.num_variables <= len(available_aisles):
    #             features = np.random.choice(available_aisles, size=self.num_variables, replace=False)
    #         else:
    #             # 否则，先填满不重复的货道
    #             unique_aisles = np.random.choice(available_aisles, size=len(available_aisles), replace=False)
    #             # 剩余部分随机选择（允许重复）
    #             remaining_length = self.num_variables - len(available_aisles)
    #             remaining_aisles = np.random.choice(available_aisles, size=remaining_length, replace=True)
    #             features = np.concatenate([unique_aisles, remaining_aisles])
    #         np.random.shuffle(features)  # 打乱顺序，增加多样性
    #         features_tuple = tuple(features)  # 转换为元组，用于比较
    #         # # 如果染色体已存在，重新生成
    #         # if features_tuple in existing_features:
    #         #     print(f"Duplicate features detected for individual {i}, regenerating...")
    #         #     continue
    #         existing_features.add(features_tuple)  # 记录新生成的染色体
    #         individual = self.problem.generate_individual()  # 创建新个体
    #         individual.features = features.tolist()  # 设置染色体
    #         print("种群初始化：", individual.features)  # 打印染色体
    #             # self.problem.decode_individual(individual)  # 解码
    #             # # # 如果解码成功（分配了节点），跳出尝试循环
    #             # if individual.allocated_nodes:
    #             #     break
    #             # print(f"Individual {i} allocation failed, attempt {attempt+1}/{max_attempts}")
    #             # # 如果尝试次数用尽，抛出异常
    #             # if attempt == max_attempts - 1:
    #             #     raise ValueError(f"Failed to generate valid allocation for individual {i}")
    #         self.problem.calculate_objectives(individual)  # 计算目标函数值
    #         population.append(individual)  # 将个体添加到种群
    #
    #     unique_features = len(existing_features)  # 统计唯一染色体数量
    #     print(f"种群中唯一个体数量: {unique_features}/{self.num_of_individuals}")
    #     return population



    def find_nearest_prime(self, n):
        """
        找到大于等于 n 的最小素数。
        :param n: 输入数字
        :return: 大于等于 n 的最小素数
        """
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(np.sqrt(num)) + 1):
                if num % i == 0:
                    return False
            return True
        num = n
        while not is_prime(num):
            num += 1
        return num

    def generate_good_lattice_points(self, num_points, num_dimensions):
        """
        使用佳点集方法生成均匀分布的点集。
        :param num_points: 点集大小（种群大小）
        :param num_dimensions: 维度（染色体长度）
        :return: 佳点集，形状为 (num_points, num_dimensions)，值在 [0, 1] 范围内
        """
        # 选择一个大于 num_points 的素数 p
        p = self.find_nearest_prime(num_points)
        # 构造生成向量 h = (1, p, p^2, ..., p^(d-1)) mod p
        h = [1] * num_dimensions
        for i in range(1, num_dimensions):
            # h[i] = (h[i-1] * p) % p
            h[i] = i + 1  # 保证 h[i] > 1
        # 生成佳点集
        points = np.zeros((num_points, num_dimensions))
        for i in range(num_points):
            for j in range(num_dimensions):
                points[i, j] = ((i+1) * h[j] % p) / p  # 值在 [0, 1] 范围内
        return points

    def create_initial_population_good_lattice_points(self):
        """
        使用佳点集方法生成初始种群。
        :return: 初始种群（Population 对象）
        """
        population = Population()

        # 使用佳点集生成均匀分布的点集
        points = self.generate_good_lattice_points(self.num_of_individuals, self.num_variables)

        # 将佳点集映射到货道编号范围
        for i in range(self.num_of_individuals):
            # 将 [0, 1] 映射到 [0, max_aisle_id]
            features = points[i] * self.max_aisle_id+ 1  # 映射到 [1, max_aisle_id+1]
            # 四舍五入为整数，确保是有效的货道编号
            features = np.round(features).astype(int)
            # 限制在合法范围内
            features = np.clip(features, 0, self.max_aisle_id)
            # 创建个体
            individual = self.problem.generate_individual()
            # print("种群初始化：",features.tolist())
            individual.features = features.tolist()
            # 计算目标函数值
            self.problem.calculate_objectives(individual)
            population.append(individual)
        # unique_features = len(set(tuple(ind.features) for ind in population))
        # print(f"种群中唯一个体数量: {unique_features}/{len(population)}")
        return population


    # ENS-SS 排序
    def ens_ss_nondominated_sort(self, Population):
        # 确保种群不为空
        if not Population.population:
            Population.fronts = []
            return
        for ind in Population.population:
            if len(ind.objectives) != 3:
                print(f"Inconsistent objectives length for individual: {ind}, length: {len(ind.objectives)}")
        # 提取种群中所有个体的目标值,转换为numpy数组
        objective_array = np.array([ind.objectives for ind in Population.population])   # (100, 3)
        # 使用pymoo的ENS-SS非支配排序（这里使用更高效的算法实现）
        # 注意：pymoo默认处理最小化问题，如果你的问题是最大化需要乘以-1
        nds = NonDominatedSorting(method="efficient_non_dominated_sort")
        fronts_indices = nds.do(objective_array, only_non_dominated_front=False)
        # 清空原有前沿信息
        Population.fronts = []

        # 将索引转换为实际个体并填充前沿
        for indices in fronts_indices:
            front = [Population.population[i] for i in indices]
            Population.fronts.append(front)
        # 设置个体的rank属性
        for rank, front in enumerate(Population.fronts):
            for individual in front:
                individual.rank = rank

    #基于整个种群范围归一化，保证跨代公平性,赋予较大值但非绝对，平衡边界保留和中间解选择压力;
    #极小尺度（<1e-6）时按1处理，避免对无效目标的过度加权;对非边界个体的拥挤度进行最大值归一化，使选择更稳定
    #测试后发现解 多样新降低了
    def calculate_crowding_distance_Improved(self, front, global_min, global_max):
        """
        front: 当前前沿的个体列表
        global_min: 各目标全局最小值列表，如 [min_f1, min_f2, min_f3]
        global_max: 各目标全局最大值列表，如 [max_f1, max_f2, max_f3]
        """
        if len(front) == 0:
            return

        num_objs = len(front[0].objectives)
        scales = []

        # 全局归一化：使用整个种群的目标值范围（避免前沿局部范围失真）
        global_min = [float('inf')] * num_objs
        global_max = [-float('inf')] * num_objs

        # 计算每个目标的归一化尺度（避免除零）
        scales = []
        for m in range(num_objs):
            scale = global_max[m] - global_min[m]
            scales.append(1.0 if scale < 1e-6 else scale)  # 极小尺度按1处理

        # 初始化拥挤度
        for ind in front:
            ind.crowding_distance = 0.0

        # 逐目标计算贡献
        for m in range(num_objs):
            # 按当前目标排序
            front.sort(key=lambda x: x.objectives[m])

            # 边界个体处理（赋予优先值但不绝对）
            front[0].crowding_distance += 1e6  # 保证边界保留，但不完全压制其他
            front[-1].crowding_distance += 1e6

            # 中间个体计算归一化后的相邻差
            if len(front) > 2:
                for i in range(1, len(front)-1):
                    delta = (front[i+1].objectives[m] - front[i-1].objectives[m]) / scales[m]
                    front[i].crowding_distance += delta

        # 对非边界个体拥挤度平滑处理
        max_crowd = max(ind.crowding_distance for ind in front)
        for ind in front:
            if ind.crowding_distance < 1e6:  # 非边界个体
                ind.crowding_distance = (ind.crowding_distance / max_crowd) if max_crowd > 0 else 0.0


    #使用 pymoo 计算拥挤距离
    def calculate_crowding_distance_pymoo(self, front):
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        # 提取目标值矩阵
        objectives = np.array([individual.objectives for individual in front])
        if np.any(np.isnan(objectives)) or np.any(np.isinf(objectives)):
            print("使用 pymoo 计算拥挤距离时，目标值矩阵中存在 NaN 或 inf")
        # 使用 pymoo 计算拥挤距离
        crowding_distances = calc_crowding_distance(objectives)

        # 将计算结果赋值给个体
        for individual, cd in zip(front, crowding_distances):
            # print("个体", individual.features, "拥挤度", cd)
            individual.crowding_distance = cd

    #原始计算当前前沿种群的拥挤度
    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10 ** 9
                front[solutions_num - 1].crowding_distance = 10 ** 9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num - 1):
                    front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale

    #选择个体
    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_children(self, population, current_gen, max_gen):
        """
        创建子代种群，使用自适应交叉和变异。
        参数:
            population: Population - 当前种群。
            num_aisles: int - 货道数量（基因长度）。
            current_gen: int - 当前迭代次数。
            max_gen: int - 最大迭代次数。
        返回:
            children: list - 子代个体列表。
        """
        children = []
        scale = 1.0 #数值越小，越平缓
        x = scale * (1 - 2 * current_gen / max_gen)
        # 自适应交叉概率
        Sigmoid = 1 / (1 + np.exp(-x))  #【0，1】
        crossover_prob = 0.3 + (0.7 - 0.3) * Sigmoid  # 【0.3，0.7】
        # 自适应变异概率
        mutation_prob = 0.05 + (0.3 - 0.05) * Sigmoid    # 0.05-0.3
        # print(f"Generation {current_gen}: crossover_prob={crossover_prob:.3f}，mutation_prob={mutation_prob:.3f}")

        # 记录种群中重复个体的比例，便于调试
        unique_features = len(set(tuple(ind.features) for ind in population))
        print(f"种群中唯一个体数量: {unique_features}/{len(population)}")
        # 如果种群多样性过低，提前引入随机个体
        diversity_threshold = 0.5
        if unique_features / len(population) < diversity_threshold:
            print(f"第{current_gen}代，种群多样性过低，提前引入 10% 的随机个体")
            num_random = int(self.num_of_individuals * 0.1)  # 引入 10% 的随机个体
            immigrants = self.create_immigrants_population(size=num_random)
            children.extend(immigrants)

        while len(children) < self.num_of_individuals:
            parent1 = self.__tournament(population)     #随机选择父代
            parent2 = self.__tournament(population)     #随机选择父代
            attempts = 0
            max_attempts = 5
            # 确保父代不同
            while parent1 == parent2 and attempts < max_attempts:
                parent2 = self.__tournament(population)
                attempts += 1
            if parent1 == parent2:
                print("create_children 时，parent1 == parent2 after max attempts\n"
                      "parent1.features =", parent1.features,"\n"
                      "parent2.features =", parent2.features)
                # 如果种群中还有不同个体，随机选择一个
                other_individuals = [ind for ind in population.population if ind != parent1]
                if other_individuals:
                    parent2 = np.random.choice(other_individuals)
                else:
                    print("种群中所有个体相同，无法选择不同父代")
            #PBX交叉
            if random.random() < crossover_prob:
                # child1, child2 = self.__crossover_order(parent1, parent2)
                # 计算 num_positions，并确保它是整数且在合理范围内
                num_positions = int(crossover_prob * self.num_variables)
                num_positions = max(1, min(num_positions, self.num_variables - 1))  # 确保 1 <= num_positions < num_variables
                # print(f"Generation {current_gen}代，使用PBX交叉,crossover_prob={crossover_prob:.3f},num_positions={num_positions}")
                child1, child2 = self.custom_cxPositionBased(parent1, parent2,num_positions)
            else:
                child1 = Individual()
                child2 = Individual()
                child1.features = parent1.features.copy()
                child2.features = parent2.features.copy()
            # 变异
            child1 = self.swap_inverse_mutation( current_gen,child1, mutation_prob)
            child2 = self.swap_inverse_mutation(current_gen,child2, mutation_prob)
            # 计算目标函数值
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            if len(children) < self.num_of_individuals:
                children.append(child2)

        children = children[:self.num_of_individuals]  # 切片子代列表，确保子代数量正确
        return children

    #交叉变异+移民操作 生成子代
    # def create_children(self, population,  current_gen, max_gen):
    #     """
    #     根据配置的交叉和变异类型生成子代。
    #     :param population: 当前种群
    #     :param current_gen: 当前代数
    #     :param max_gen: 最大代数
    #     :return: 子代列表
    #     """
    #     children = []
    #
    #     # 记录种群中重复个体的比例，便于调试
    #     unique_features = len(set(tuple(ind.features) for ind in population))
    #     print(f"种群中唯一个体数量: {unique_features}/{len(population)}")
    #     # 如果种群多样性过低，提前引入随机个体
    #     diversity_threshold = 0.5
    #     if unique_features / len(population) < diversity_threshold:
    #         print(f"第{current_gen}代，种群多样性过低，提前引入 10% 的随机个体")
    #         num_random = int(self.num_of_individuals * 0.1)  # 引入 10% 的随机个体
    #         immigrants = self.create_immigrants_population(size=num_random)
    #         children.extend(immigrants)
    #
    #     while len(children) < self.num_of_individuals:
    #         parent1 = self.__tournament(population)     #随机选择父代
    #         parent2 = self.__tournament(population)     #随机选择父代
    #         attempts = 0
    #         max_attempts = 5
    #         # 确保父代不同
    #         while parent1 == parent2 and attempts < max_attempts:
    #             parent2 = self.__tournament(population)
    #             attempts += 1
    #         if parent1 == parent2:
    #             print("create_children 时，parent1 == parent2 after max attempts\n"
    #                   "parent1.features =", parent1.features,"\n"
    #                   "parent2.features =", parent2.features)
    #             # 如果种群中还有不同个体，随机选择一个
    #             other_individuals = [ind for ind in population.population if ind != parent1]
    #             if other_individuals:
    #                 parent2 = np.random.choice(other_individuals)
    #             else:
    #                 print("种群中所有个体相同，无法选择不同父代")
    #         # 根据配置选择交叉方法
    #         if self.crossover_type == "two_point":  # 两点交叉
    #             child1, child2 = self.__crossover_two_point(parent1, parent2)
    #         elif self.crossover_type == "pbx":  # 自适应 PBX 交叉
    #             child1, child2 = self.adaptive_pbx_crossover(parent1, parent2, current_gen, max_gen, self.num_variables)
    #         elif self.crossover_type == "pmx":        # 部分映射交叉
    #             child1, child2 = self.__crossover_pmx(parent1, parent2)
    #         elif self.crossover_type == "sbx":        # 模拟二进制交叉
    #             child1, child2 = self.SBX__crossover(parent1, parent2)
    #         elif self.crossover_type == "crossover":    # Deap的顺序交叉
    #             child1, child2 = self.__crossover_order(parent1, parent2)
    #         else:
    #             raise ValueError(f"Unsupported crossover type: {self.crossover_type}")
    #         child1 = self.adaptive_swap_inverse_mutation(child1, self.num_variables, current_gen, max_gen, 0.3, 0.05)
    #         child2 = self.adaptive_swap_inverse_mutation(child2, self.num_variables, current_gen, max_gen, 0.3, 0.05)
    #
    #         # 根据配置选择变异方法
    #         # if self.__choose_with_prob(self.mutation_param):
    #         #     if self.mutation_type == "swap":     # 交换变异
    #         #         child1 = self.__mutate_swap(child1)
    #         #     elif self.mutation_type == "shuffle":   #打乱变异
    #         #         child1 = self.__mutate_shuffle(child1)
    #         #     elif self.mutation_type == "inversion":     # 反转变异
    #         #         child1 = self.__mutate_inversion(child1)
    #         #     elif self.mutation_type == "mutpolyn":       # 多项式变异
    #         #         child1 = self.mutpolyn__mutate(child1)
    #         #     elif self.mutation_type == "mutate":       # deap打乱变异
    #         #         child1 = self.mutate(child1)
    #         #     else:
    #         #         raise ValueError(f"Unsupported mutation type: {self.mutation_type}")
    #         # # 检查 child1 和 child2 是否为 None
    #         # if child1 is None or child2 is None:
    #         #     raise ValueError(f"Crossover returned None: child1={child1}, child2={child2}")
    #         # if self.__choose_with_prob(self.mutation_param):
    #         #     if self.mutation_type == "swap":
    #         #         child2 = self.__mutate_swap(child2)
    #         #     elif self.mutation_type == "shuffle":
    #         #         child2 = self.__mutate_shuffle(child2)
    #         #     elif self.mutation_type == "inversion":
    #         #         child2 = self.__mutate_inversion(child2)
    #         #     elif self.mutation_type == "mutpolyn":
    #         #         child2 = self.mutpolyn__mutate(child2)
    #         #     elif self.mutation_type == "mutate":       # deap打乱变异
    #         #         child2 = self.mutate(child2)
    #         #     else:
    #         #         raise ValueError(f"Unsupported mutation type: {self.mutation_type}")
    #         #计算目标函数值
    #         self.problem.calculate_objectives(child1)
    #         self.problem.calculate_objectives(child2)
    #         children.append(child1)
    #         children.append(child2)
    #     children = children[:self.num_of_individuals]  # 切片子代列表，确保子代数量正确
    #
    #     return children

    # def create_children_adaptive_pbx(self, population, num_aisles, current_gen, max_gen):
    #     """
    #     创建子代种群，使用自适应 PBX 交叉和 Sigmoid 自适应概率。
    #     参数:
    #         population: Population - 当前种群。
    #         num_aisles: int - 货道数量（基因长度）。
    #         current_gen: int - 当前迭代次数。
    #         max_gen: int - 最大迭代次数。
    #     返回:
    #         children: list - 子代个体列表。
    #     """
    #     def sigmoid(x):
    #         """
    #         计算 Sigmoid 函数值。
    #         参数:
    #             x: float - 输入值。
    #         返回:
    #             float - Sigmoid 函数值，范围 (0, 1)。
    #         """
    #         return 1 / (1 + np.exp(-x))
    #     def adaptive_probability(current_gen, max_gen, scale):
    #         """
    #         使用 Sigmoid 函数计算自适应概率。
    #         参数:
    #             current_gen: int - 当前迭代次数。
    #             max_gen: int - 最大迭代次数。
    #             scale: float - 控制 Sigmoid 函数斜率的参数。
    #             增大 scale（例如，10 → 15），概率变化更陡峭，前期更高，后期更低。
    #             减小 scale（例如，5 → 3），概率变化更平缓。
    #         返回:
    #             float - 自适应概率，范围 (0, 1)。
    #         """
    #         x = scale * (1 - 2 * current_gen / max_gen)
    #         return sigmoid(x)
    #     children = []
    #     # 计算自适应交叉和变异概率
    #     crossover_prob = adaptive_probability(current_gen, max_gen, scale=5.0)  # 前期 0.9，后期 0.5
    #     mutation_prob = adaptive_probability(current_gen, max_gen, scale=10.0)  # 前期 0.3，后期 0.05
    #
    #     print(f"Generation {current_gen}: crossover_prob={crossover_prob:.3f}, mutation_prob={mutation_prob:.3f}")
    #
    #     for _ in range(len(population.individuals)):
    #         # 随机选择两个父代
    #         parent1, parent2 = np.random.choice(population.individuals, 2, replace=False)
    #
    #         # 以 crossover_prob 的概率进行交叉
    #         if np.random.random() < crossover_prob:
    #             child = adaptive_pbx_crossover(
    #                 parent1, parent2,
    #                 current_gen=current_gen,
    #                 max_gen=max_gen,
    #                 gene_length=num_aisles,
    #                 initial_scale=0.5,  # 初始选择 50% 的基因位
    #                 final_scale=0.1     # 最终选择 10% 的基因位
    #             )
    #         else:
    #             # 如果不进行交叉，随机选择一个父代作为子代
    #             child = deepcopy(parent1 if np.random.random() < 0.5 else parent2)
    #
    #         # 变异操作
    #         child = self.mutate(child, num_aisles, mutation_prob)
    #
    #         # 计算子代的目标值
    #         self.problem.calculate_objectives(child)
    #
    #         children.append(child)
    #
    #     return children

    #两点交叉
    def __crossover_two_point(self, parent1, parent2):
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        crossover_point1 = random.randint(0, len(parent1.features) - 2)
        crossover_point2 = random.randint(crossover_point1 + 1, len(parent1.features) - 1)
        child1.features = parent1.features[:crossover_point1] + parent2.features[crossover_point1:crossover_point2] + parent1.features[crossover_point2:]
        child2.features = parent2.features[:crossover_point1] + parent1.features[crossover_point1:crossover_point2] + parent2.features[crossover_point2:]
        return child1, child2

    def adaptive_pbx_crossover(self, parent1, parent2, current_gen, max_gen, gene_length, initial_scale=0.5, final_scale=0.1):
        """
        自适应 PBX 交叉算子。
        参数:
            parent1: Individual - 第一个父代个体。
            parent2: Individual - 第二个父代个体。
            current_gen: int - 当前迭代次数。
            max_gen: int - 最大迭代次数。
            gene_length: int - 基因长度（货道数量）。
            initial_scale: float - 初始缩放系数。
            final_scale: float - 最终缩放系数。
        返回:
            child: Individual - 交叉生成的子代个体。
        """
        # 使用 Sigmoid 函数计算缩放系数
        scale = 5.0  # 控制 Sigmoid 斜率
        x = scale * (1 - 2 * current_gen / max_gen)
        scale_factor = 1 / (1 + np.exp(-x))
        # 线性插值缩放系数
        scale_factor = final_scale + (initial_scale - final_scale) * scale_factor
        # 计算基因位选择数量
        num_positions = int(gene_length * scale_factor)
        num_positions = max(1, num_positions)
        print(f"Generation {current_gen}: scale_factor={scale_factor:.3f}, num_positions={num_positions}")
        # 调用改进的 PBX 交叉
        # child = self.improved_pbx_crossover(parent1, parent2, num_positions, head_ratio=0.1)
        child = self.base_pbx_crossover(parent1, parent2, num_positions, head_ratio=0.1)
        return child

    '''改进的 PBX 交叉算子，确保头部基因覆盖小序号和大序号货道'''
    def improved_pbx_crossover(self, parent1, parent2, num_positions, head_ratio=0.1):
        """
        改进的 PBX 交叉算子，确保头部基因覆盖小序号和大序号货道。

        参数:
            parent1: Individual - 第一个父代个体，包含 features 属性（货道序号排列）。
            parent2: Individual - 第二个父代个体。
            num_positions: int - 选择的基因位数量。
            head_ratio: float - 头部比例（默认 10%）。

        返回:
            child: Individual - 交叉生成的子代个体。
        """
        # 深拷贝父代个体
        child = deepcopy(parent1)
        gene_length = len(parent1.features)

        # 确定头部长度
        head_length = int(gene_length * head_ratio)

        # 确保 num_positions 不超过基因长度
        num_positions = min(num_positions, gene_length)

        # 分段选择基因位：头部优先选择小序号和大序号
        head_positions = np.random.choice(range(head_length), size=min(num_positions // 2, head_length), replace=False)

        # 剩余基因位从整个染色体中随机选择
        remaining_positions = np.random.choice(range(head_length, gene_length), size=(num_positions - len(head_positions)), replace=False)
        selected_positions = np.concatenate([head_positions, remaining_positions])

        # 步骤 1：将 parent1 中选定位置的基因值复制到子代
        child_features = np.full(gene_length, -1, dtype=int)
        for pos in selected_positions:
            child_features[pos] = parent1.features[pos]

        # 步骤 2：从 parent2 中按顺序填充剩余位置，确保不重复
        remaining_positions = [i for i in range(gene_length) if i not in selected_positions]
        parent2_values = parent2.features.copy()
        parent2_index = 0

        # 优先填充头部，确保小序号和大序号的覆盖
        for pos in remaining_positions:
            if pos < head_length:  # 头部位置
                # 优先选择小序号或大序号
                while parent2_index < gene_length:
                    value = parent2_values[parent2_index]
                    parent2_index += 1
                    if value not in child_features:
                        # 优先选择小序号（1-100）或大序号（400-490）
                        if (value <= 300 or value >= 300) or parent2_index == gene_length:
                            child_features[pos] = value
                            break
            else:  # 非头部位置
                while parent2_index < gene_length:
                    value = parent2_values[parent2_index]
                    parent2_index += 1
                    if value not in child_features:
                        child_features[pos] = value
                        break

        # 确保所有位置都被填充
        if -1 in child_features:
            raise ValueError("PBX 交叉失败：子代基因未完全填充")

        child.features = child_features
        return child

    '''简化的 PBX 交叉算子，仅保留基本功能'''
    def base_pbx_crossover(self, parent1, parent2, num_positions, head_ratio=0.1):
        """
        简化的 PBX 交叉算子，确保子代基因完全填充。

        参数:
            parent1: Individual - 第一个父代个体，包含 features 属性（货道序号排列）。
            parent2: Individual - 第二个父代个体。
            num_positions: int - 选择的基因位数量。
            head_ratio: float - 头部比例（仅用于选择基因位数量）。

        返回:
            child: Individual - 交叉生成的子代个体。
        """
        child = deepcopy(parent1)
        gene_length = len(parent1.features)
        head_length = int(gene_length * head_ratio)

        # 确保 num_positions 不超过基因长度
        num_positions = min(num_positions, gene_length)
        head_positions_to_select = min(num_positions // 2, head_length)
        head_positions = np.random.choice(range(head_length), size=head_positions_to_select, replace=False)
        remaining_positions_to_select = num_positions - len(head_positions)
        remaining_positions = np.random.choice(range(head_length, gene_length), size=remaining_positions_to_select, replace=False)
        selected_positions = np.concatenate([head_positions, remaining_positions])

        # 初始化子代基因
        child_features = np.full(gene_length, -1, dtype=int)
        for pos in selected_positions:
            child_features[pos] = parent1.features[pos]

        # 收集已使用的基因值
        used_values = set(child_features[child_features != -1])

        # 从 parent2 中提取未使用的基因值
        remaining_positions = [i for i in range(gene_length) if i not in selected_positions]
        available_values = [value for value in parent2.features if value not in used_values]

        # 确保可用基因值数量足够
        if len(available_values) != len(remaining_positions):
            raise ValueError(f"可用基因值数量不足：需要 {len(remaining_positions)} 个，实际有 {len(available_values)} 个")

        # 按顺序填充剩余位置
        for i, pos in enumerate(remaining_positions):
            child_features[pos] = available_values[i]

        # 验证是否完全填充
        if -1 in child_features:
            raise ValueError("PBX 交叉失败：子代基因未完全填充")

        child.features = child_features
        return child

    '''自定义基于位置的交叉（PBX），允许控制选择的位置数量'''
    def custom_cxPositionBased(self, parent1, parent2, num_positions=None):
        """
        自定义基于位置的交叉（PBX），允许控制选择的位置数量。
        参数：
        ind1, ind2: 两个父代的特征列表（排列）
        num_positions: 要选择的位置数量（可选），如果为 None，则随机选择
        返回：
        两个子代的特征列表
        """
        child1 = Individual()
        child2 = Individual()
        ind1 = parent1.features
        ind2 = parent2.features
        size = len(ind1)
        # 确保输入是列表
        child1_features = ind1.copy()
        child2_features = ind2.copy()
        # 确定要选择的位置数量
        if num_positions is None:
            # 如果未指定，随机选择 1 到 size-1 个位置
            num_positions = random.randint(1, size - 1)
        else:
            # 确保 num_positions 在合理范围内
            if not 1 <= num_positions < size:
                raise ValueError(f"num_positions must be between 1 and {size-1}, got {num_positions}")

        # 随机选择 num_positions 个位置 ,从 0 到 size-1 的范围内随机选择 num_positions 个不重复的位置。
        selected_positions = random.sample(range(size), num_positions)

        # 提取父代的基因
        selected_genes1 = [ind1[pos] for pos in selected_positions]  # 父代 1 在这些位置的基因
        selected_genes2 = [ind2[pos] for pos in selected_positions]  # 父代 2 在这些位置的基因

        # 填充子代 1：保留 ind1 的选中位置，按照 ind2 的顺序填充剩余部分
        remaining1 = [x for x in ind2 if x not in selected_genes1]  # ind2 中未被选中的基因
        pos = 0
        for i in range(size):
            if i in selected_positions:
                child1_features[i] = ind1[i]  # 保留父代 1 的基因
            else:
                child1_features[i] = remaining1[pos]  # 按父代 2 的顺序填充
                pos += 1

        # 填充子代 2：保留 ind2 的选中位置，按照 ind1 的顺序填充剩余部分
        remaining2 = [x for x in ind1 if x not in selected_genes2]  # ind1 中未被选中的基因
        pos = 0
        for i in range(size):
            if i in selected_positions:
                child2_features[i] = ind2[i]  # 保留父代 2 的基因
            else:
                child2_features[i] = remaining2[pos]  # 按父代 1 的顺序填充
                pos += 1
        child1.features = child1_features
        child2.features = child2_features
        return child1, child2
    '''固定概率 两点交换和逆序变异'''
    def swap_inverse_mutation(self, current_gen,individual, mutation_prob):
        """
        两点交换和逆序变异。
        参数:
            current_gen: int - 当前迭代次数。
            individual: Individual - 需要变异的个体。
            mutation_prob: float - 变异概率。
        返回:
            mutated: Individual - 变异后的个体。
        """
        child = Individual()
        mutated = deepcopy(individual.features)
        gene_length = len(mutated)
        # Swap 交换变异
        if np.random.random() < mutation_prob:
            if np.random.random() < 0.5:    # 50% 概率交换
                pos1, pos2 = np.random.choice(range(gene_length), size=2, replace=False)
                mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
                # 打印 mutation_prob（调试用）
                # print(f"Generation {current_gen}: Swap 变异 mutation_prob={mutation_prob:.3f}")
            else:    # 50% 概率反转 变异
                # Inverse反转 变异
                pos1, pos2 = np.random.choice(range(gene_length), size=2, replace=False)
                start, end = min(pos1, pos2), max(pos1, pos2)
                mutated[start:end+1] = mutated[start:end+1][::-1]
                # print(f"Generation {current_gen}: Inverse 变异 mutation_prob={mutation_prob:.3f}")
        child.features = mutated
        return child

    '''自适应变异，两点交换变异和逆序变异'''
    def adaptive_swap_inverse_mutation(self,individual, num_aisles, current_gen, max_gen, initial_prob=0.3, final_prob=0.05):
        """
        自适应变异，仅使用 Sigmoid 函数计算变异概率。

        参数:
            individual: Individual - 需要变异的个体。
            num_aisles: int - 货道数量（基因长度）。
            current_gen: int - 当前迭代次数。
            max_gen: int - 最大迭代次数。
            initial_prob: float - 初始变异概率（例如，0.3）。
            final_prob: float - 最终变异概率（例如，0.05）。

        返回:
            mutated: Individual - 变异后的个体。
        """
        # 使用 Sigmoid 函数计算变异概率
        scale = 5.0  # 控制 Sigmoid 斜率
        x = scale * (1 - 2 * current_gen / max_gen)
        sigmoid_value = 1 / (1 + np.exp(-x))

        # 直接在 Sigmoid 中调整范围
        mutation_prob = final_prob + (initial_prob - final_prob) * sigmoid_value
        # 调用 Swap 和 Inverse 变异
        mutated = self.swap_inverse_mutation( current_gen,individual, mutation_prob)

        return mutated

    #"""使用模拟二进制交叉（SBX）生成子代"""
    def SBX__crossover(self, parent1, parent2):

        child1 = Individual()
        child2 = Individual()

        # 将父代的 features 转换为 numpy 数组
        parent1_features = np.array(parent1.features, dtype=float)
        parent2_features = np.array(parent2.features, dtype=float)

        # 构造染色体矩阵
        OldChrom = np.vstack([parent1_features, parent2_features])

        # 使用 geatpy 的 recsbx 进行交叉
        XOVR = 0.7  # 交叉概率
        n = 20      # 分布指数
        new_chrom = self.recOper.do(OldChrom)

        # 提取子代 features，并四舍五入为整数
        child1_features = np.round(new_chrom[0]).astype(int)
        child2_features = np.round(new_chrom[1]).astype(int)

        # 限制子代 features 在合法范围内
        child1_features = np.clip(child1_features, 0, self.problem.num_of_variables - 1)
        child2_features = np.clip(child2_features, 0, self.problem.num_of_variables - 1)

        # 赋值给子代
        child1.features = child1_features.tolist()
        child2.features = child2_features.tolist()

        # 重置解码状态
        child1.reset_decode()
        child2.reset_decode()

        return child1, child2

    #顺序交叉（Order Crossover, OX）
    def __crossover_order(self, parent1, parent2):

        child1 = Individual()
        child2 = Individual()
        child1.features = parent1.features.copy()
        child2.features = parent2.features.copy()
        # 使用 deap 的顺序交叉（cxOrdered）
        # 注意：cxOrdered 修改的是输入列表，所以我们需要确保 child1.features 和 child2.features 是列表
        child1.features, child2.features = tools.cxOrdered(child1.features, child2.features)
        return child1, child2
        # size = len(parent1.features)
        # cxpoint1 = random.randint(0, size - 1)
        # cxpoint2 = random.randint(cxpoint1 + 1, size)
        # # child1 从 parent1 复制交叉片段
        # child1.features[cxpoint1:cxpoint2] = parent1.features[cxpoint1:cxpoint2]
        # # child2 从 parent2 复制交叉片段
        # child2.features[cxpoint1:cxpoint2] = parent2.features[cxpoint1:cxpoint2]
        # # 填充 child1 剩余位置
        # pos = cxpoint2
        # for i in range(size):
        #     gene = parent2.features[(cxpoint2 + i) % size]
        #     if gene not in child1.features[cxpoint1:cxpoint2]:
        #         if pos >= size:
        #             pos = 0
        #         while pos < cxpoint1 or (pos >= cxpoint1 and pos < cxpoint2):
        #             print(f"pos: {pos}, cxpoint1: {cxpoint1}, cxpoint2: {cxpoint2}, i: {i}, size: {size}")
        #             pos += 1
        #             if pos >= size:
        #                 pos = 0
        #         child1.features[pos] = gene
        #         pos += 1
        # # 填充 child2 剩余位置
        # pos = cxpoint2
        # for i in range(size):
        #     gene = parent1.features[(cxpoint2 + i) % size]
        #     if gene not in child2.features[cxpoint1:cxpoint2]:
        #         if pos >= size:
        #             pos = 0
        #         while pos < cxpoint1 or (pos >= cxpoint1 and pos < cxpoint2):
        #             print(f"pos: {pos}, cxpoint1: {cxpoint1}, cxpoint2: {cxpoint2}, i: {i}, size: {size}")
        #             pos += 1
        #             if pos >= size:
        #                 pos = 0
        #         child2.features[pos] = gene
        #         pos += 1
        # 调整重复基因
        # child1.features = self.adjust_duplicates(child1.features)
        # child2.features = self.adjust_duplicates(child2.features)
        # return child1, child2

    # 调整染色体中的重复基因，尽量减少重复
    def adjust_duplicates(self, features):
        # 统计每个基因（货道编号）的出现次数
        gene_counts = {}
        for gene in features:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1
        # 计算每个基因允许的最大重复次数
        # max_allowed_repeats = max(1, self.num_variables // (self.max_aisle_id + 1) + 1)
        max_allowed_repeats = 10  # 最大重复次数
        available_aisles = list(range(self.max_aisle_id))  # 所有可用货道编号
        # 遍历染色体
        for i in range(len(features)):
            gene = features[i]
            # 如果某个基因出现次数超过允许的最大值
            if gene_counts[gene] > max_allowed_repeats:
                # 选择一个出现次数较少的基因进行替换
                candidates = [g for g in available_aisles if gene_counts.get(g, 0) < max_allowed_repeats]
                if candidates:
                    new_gene = np.random.choice(candidates)  # 随机选择一个候选基因
                    gene_counts[gene] -= 1  # 减少旧基因的计数
                    features[i] = new_gene  # 替换基因
                    gene_counts[new_gene] = gene_counts.get(new_gene, 0) + 1  # 增加新基因的计数
                    print(f"基因 {gene} 出现次数超过允许的最大值 {max_allowed_repeats}, 被替换为 {new_gene}")
        return features

    #部分映射交叉（Partially Mapped Crossover, PMX）
    #部分匹配交叉保证了每个染色体中的基因仅出现一次，通过该交叉策略在一个染色体中不会出现重复的基因，
    # 所以PMX经常用于旅行商（TSP）或其他排序问题编码。
    def __crossover_pmx(self, parent1, parent2):
        """
        """
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        # size = len(parent1.features)
        size = self.problem.num_of_variables     # 基因数量
        cxpoint1 = random.randint(0, size - 1)
        cxpoint2 = random.randint(cxpoint1 + 1, size)

        child1.features = parent1.features.copy()
        child2.features = parent2.features.copy()

        for i in range(cxpoint1, cxpoint2):
            child1.features[i], child2.features[i] = child2.features[i], child1.features[i]

        for i in range(cxpoint1, cxpoint2):
            gene1 = child1.features[i]
            gene2 = parent2.features[i]
            if gene1 != gene2:
                for j in range(size):
                    if j < cxpoint1 or j >= cxpoint2:
                        if child1.features[j] == gene1:
                            child1.features[j] = gene2
                            break
            gene1 = parent1.features[i]
            gene2 = child1.features[i]
            if gene1 != gene2:
                for j in range(size):
                    if j < cxpoint1 or j >= cxpoint2:
                        if child2.features[j] == gene2:
                            child2.features[j] = gene1
                            break

        return child1, child2

    def mutate(self, individual):
        # print("mutate", individual.features)
        # 使用 deap 的 mutShuffleIndexes 进行变异
        # indpb 是每个基因的变异概率，这里设为 0.1（10% 的基因会被打乱）随机打乱染色体中的基因
        individual.features = tools.mutShuffleIndexes(individual.features, indpb=0.1)[0]
        return individual

    #两点变异：交换变异，通过交换个体中的两个基因位置的值，来产生新的个体
    def __mutate_swap(self, child):
        mutation_points1 = random.randint(0,self.problem.num_of_variables - 1)
        mutation_points2 = random.randint(0,self.problem.num_of_variables - 1)
        while mutation_points1 == mutation_points2:
            mutation_points2 = random.randint(0,self.problem.num_of_variables - 1)
        child.features[mutation_points1], child.features[mutation_points2] = child.features[mutation_points2], child.features[mutation_points1]
        return child

    #打乱变异
    def __mutate_shuffle(self, child):
        """
        """
        size = self.problem.num_of_variables
        start = random.randint(0, size - 1)
        end = random.randint(start + 1, size)
        segment = child.features[start:end]
        random.shuffle(segment)
        child.features[start:end] = segment
        return child

    #反转变异
    def __mutate_inversion(self, child):
        """
        """
        size = self.problem.num_of_variables
        start = random.randint(0, size - 1)
        end = random.randint(start + 1, size)
        child.features[start:end] = child.features[start:end][::-1]
        return child

    #使用多项式变异生成新个体
    def mutpolyn__mutate(self, child):

        # 将子代的 features 转换为 numpy 数组
        # child_features = np.array(child.features, dtype=float)
        # 将子代的 features 转换为 numpy 数组，并调整为二维数组
        child_features = np.array(child.features, dtype=float).reshape(1, -1)  # 形状 (1, num_of_variables)
        # 调用 geatpy 的 mutpolyn 进行变异
        new_chrom = self.mutOper.do(Encoding='RI',OldChrom=child_features,FieldDR=self.FieldDR)
        # 提取变异后的结果（new_chrom 仍为二维数组，取第一行）
        new_features = np.round(new_chrom[0]).astype(int)  # 四舍五入为整数
        # 限制子代 features 在合法范围内（与 FieldDR 一致）
        new_features = np.clip(new_features, 0, self.problem.num_of_variables-1)
        # 赋值给子代
        child.features = new_features.tolist()
        # 重置解码状态
        child.reset_decode()
        return child

    # 锦标赛选择，用于选择父代
    def __tournament(self, population):
        # 随机选择两个候选个体
        candidates = np.random.choice(population.population, size=2, replace=False)
        # 比较 rank（非支配排序等级），选择 rank 较小的（更优）
        if candidates[0].rank < candidates[1].rank:
            return candidates[0]
        elif candidates[1].rank < candidates[0].rank:
            return candidates[1]
        else:
            # 如果 rank 相同，比较拥挤距离，选择较大的（种群分布更均匀）
            return candidates[0] if candidates[0].crowding_distance > candidates[1].crowding_distance else candidates[1]


    #随机决定是否发生变异/交叉
    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False

    #保存当前 Pareto 前沿到文件
    def save_pareto_front(front,true_pf_file, run_id):
        """
        保存当前 Pareto 前沿到 CSV 文件
        """
        if not front:
            return

        objectives = np.array([ind.objectives for ind in front])
        # 创建 DataFrame
        df = pd.DataFrame(objectives, columns=['F1', 'F2', 'F3'])
        df['RunID'] = run_id
        # 调整列顺序
        df = df[['RunID', 'F1', 'F2', 'F3']]

        # 如果文件不存在，写入表头
        if not os.path.exists(true_pf_file):
            df.to_csv(true_pf_file, index=False)
        else:
            # 追加写入，不写表头
            df.to_csv(true_pf_file, mode='a', header=False, index=False)

    def load_true_pareto_front(true_pf_file=None):
        """
        从 CSV 文件加载真实 Pareto 前沿
        """
        if not os.path.exists(true_pf_file):
            print(f"True Pareto front file {true_pf_file} does not exist.")
            return

        # 读取 CSV 文件
        df = pd.read_csv(true_pf_file)
        true_PF = []
        for _, row in df.iterrows():
            ind = Individual()
            ind.objectives = [row['F1'], row['F2'], row['F3']]
            true_PF.append(ind)

        true_PF = true_PF
        print(f"Loaded true Pareto front with {len(true_PF)} solutions")
        return true_PF


    #读取历史解集，构造近似真实 Pareto 前沿
    def load_historical_pareto_fronts(self, max_runs=10):
        """
        读取历史解集，构造近似真实 Pareto 前沿
        """
        if not os.path.exists(self.history_file):
            print(f"History file {self.history_file} does not exist. Starting fresh.")
            self.true_PF = []
            return

        # 读取 CSV 文件
        df = pd.read_csv(self.history_file)
        run_counts = df['RunID'].nunique()
        print(f"Loaded {run_counts} runs, total solutions: {len(df)}")

        # 如果运行次数过多，只保留最近 max_runs 次运行
        if run_counts > max_runs:
            recent_runs = df['RunID'].unique()[-max_runs:]
            df = df[df['RunID'].isin(recent_runs)]

        # 转换为 Individual 对象
        all_solutions = []
        for _, row in df.iterrows():
            ind = Individual()
            ind.objectives = [row['F1'], row['F2'], row['F3']]
            all_solutions.append(ind)

        # 非支配排序
        if all_solutions:
            population = Population()
            population.individuals = all_solutions
            self.utils.ens_ss_nondominated_sort(population) # 非支配排序
            self.true_PF = population.fronts[0]
            print(f"Constructed true PF from history with {len(self.true_PF)} solutions")
        else:
            self.true_PF = []