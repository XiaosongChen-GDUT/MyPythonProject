import os

import numpy as np
import pandas as pd

from NSGA2.FS4.FS4_Individual import Individual
from NSGA2.FS4.FS4_Population import Population
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
        lb = np.ones(self.num_variables)  # 下界全为 0
        ub = np.full(self.num_variables,self.num_variables)  # 上界为 num_of_variables
        varTypes = np.ones(self.num_variables)  # 变量类型为离散 (1)
        self.FieldDR = np.vstack([lb, ub, varTypes])

    #创建初始种群
    def create_initial_population(self):
        population = Population()
        for _ in range(self.num_of_individuals):
            individual = self.problem.generate_individual()
            population.append(individual)
            # 在种群创建完成后，计算所有个体的目标值
            self.problem.calculate_objectives(individual)
        return population

    #创建移民种群
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
            print("种群初始化：",features.tolist())
            individual.features = features.tolist()
            # 计算目标函数值
            self.problem.calculate_objectives(individual)
            population.append(individual)

        return population
#集佳值点初始化种群
# # 50% 使用佳点集，50% 使用随机生成
# glp_points = self.generate_good_lattice_points(self.num_of_individuals // 2, self.chromosome_length)
# for i in range(self.num_of_individuals):
#     if i < self.num_of_individuals // 2:
#         features = glp_points[i] * self.max_aisle_id
#         features = np.round(features).astype(int)
#         features = np.clip(features, 0, self.max_aisle_id)
#     else:
#         features = np.random.randint(0, self.max_aisle_id + 1, size=self.chromosome_length)
#     individual = self.problem.generate_individual()
#     individual.features = features.tolist()
#     self.problem.calculate_objectives(individual)
#     population.append(individual)

    #非支配排序
    # 计算非支配排序

    # ENS-SS 排序
    def ens_ss_nondominated_sort(self, Population):
        # 确保种群不为空
        if not Population.population:
            Population.fronts = []
            return
        # 提取种群中所有个体的目标值,转换为numpy数组
        objective_array = np.array([ind.objectives for ind in Population.population])
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

    # 传统方式计算非支配排序
    def fast_nondominated_sort(self, population):
        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            population.fronts.append(temp)

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

    #计算当前前沿种群的拥挤度
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

    #生成自子代
    def create_children(self, population):
        """
        根据配置的交叉和变异类型生成子代。
        :param population: 当前种群
        :return: 子代列表
        """
        children = []
        # 记录种群中重复个体的比例，便于调试
        unique_features = len(set(tuple(ind.features) for ind in population))
        print(f"种群中唯一个体数量: {unique_features}/{len(population)}")
        while len(children) < len(population):
            parent1 = self.__tournament(population)     #
            parent2 = self.__tournament(population)
            attempts = 0
            max_attempts = 5
            # 确保父代不同
            while parent1 == parent2 and attempts < max_attempts:
                parent2 = self.__tournament(population)
                attempts += 1
            if parent1 == parent2:
                print("create_children 时，parent1 == parent2 after max attempts")
                # 如果种群中还有不同个体，随机选择一个
                other_individuals = [ind for ind in population.population if ind != parent1]
                if other_individuals:
                    parent2 = np.random.choice(other_individuals)
                else:
                    print("种群中所有个体相同，无法选择不同父代")
            # 根据配置选择交叉方法
            if self.crossover_type == "two_point":  # 两点交叉
                child1, child2 = self.__crossover_two_point(parent1, parent2)
            elif self.crossover_type == "pmx":        # 部分映射交叉
                child1, child2 = self.__crossover_pmx(parent1, parent2)
            elif self.crossover_type == "sbx":        # 模拟二进制交叉
                child1, child2 = self.SBX__crossover(parent1, parent2)
            elif self.crossover_type == "crossover":    # 顺序交叉并调整基因
                # print("create_children 时，crossover_type 为 crossover")
                child1, child2 = self.__crossover_order(parent1, parent2)

            else:
                raise ValueError(f"Unsupported crossover type: {self.crossover_type}")

            # 根据配置选择变异方法
            if self.__choose_with_prob(self.mutation_param):
                if self.mutation_type == "swap":     # 交换变异
                    child1 = self.__mutate_swap(child1)
                elif self.mutation_type == "shuffle":   #打乱变异
                    child1 = self.__mutate_shuffle(child1)
                elif self.mutation_type == "inversion":     # 反转变异
                    child1 = self.__mutate_inversion(child1)
                elif self.mutation_type == "mutpolyn":       # 多项式变异
                    child1 = self.mutpolyn__mutate(child1)
                elif self.mutation_type == "mutate":       # deap打乱变异
                    child1 = self.mutate(child1)
                else:
                    raise ValueError(f"Unsupported mutation type: {self.mutation_type}")
            # 检查 child1 和 child2 是否为 None
            if child1 is None or child2 is None:
                raise ValueError(f"Crossover returned None: child1={child1}, child2={child2}")
            if self.__choose_with_prob(self.mutation_param):
                if self.mutation_type == "swap":
                    child2 = self.__mutate_swap(child2)
                elif self.mutation_type == "shuffle":
                    child2 = self.__mutate_shuffle(child2)
                elif self.mutation_type == "inversion":
                    child2 = self.__mutate_inversion(child2)
                elif self.mutation_type == "mutpolyn":
                    child2 = self.mutpolyn__mutate(child2)
                elif self.mutation_type == "mutate":       # deap打乱变异
                    child2 = self.mutate(child2)
                else:
                    raise ValueError(f"Unsupported mutation type: {self.mutation_type}")
            #计算目标函数值
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)

        return children

    #产生子代

    # def create_children(self, population):
    #     children = []
    #     while len(children) < len(population):
    #         parent1 = self.__tournament(population) #锦标赛选择父代
    #         parent2 = parent1
    #         while parent1 == parent2:
    #             parent2 = self.__tournament(population)
    #         #两点交叉
    #         child1, child2 = self.__crossover(parent1, parent2)
    #         # child1, child2 = self.SBX__crossover(parent1, parent2)
    #         #变异
    #         if self.__choose_with_prob(self.mutation_param):
    #             child1 = self.__mutate(child1)
    #             # child1 = self.mutpolyn__mutate(child1)
    #         if self.__choose_with_prob(self.mutation_param):
    #             child2 = self.__mutate(child2)
    #             # child2 = self.mutpolyn__mutate(child2)
    #         #计算目标函数值
    #         self.problem.calculate_objectives(child1)
    #         self.problem.calculate_objectives(child2)
    #         children.append(child1)
    #         children.append(child2)
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



    #顺序交叉+调整重复基因（Order Crossover, OX）
    def __crossover_order(self, parent1, parent2):
        # child1 = self.problem.generate_individual()
        # child2 = self.problem.generate_individual()
        child1 = Individual()
        child2 = Individual()
        child1.features = parent1.features.copy()
        child2.features = parent2.features.copy()
        # print("child1.features:", child1.features)
        # print("child2.features:", child2.features)
        # 使用 deap 的顺序交叉（cxOrdered）
        # 注意：cxOrdered 修改的是输入列表，所以我们需要确保 child1.features 和 child2.features 是列表
        child1.features, child2.features = tools.cxOrdered(child1.features, child2.features)

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
        return child1, child2

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
    def __crossover_pmx(self, parent1, parent2):
        """
        """
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        size = len(parent1.features)
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

    #锦标赛
    # def __tournament(self, population):
    #     participants = random.sample(population.population, self.num_of_tour_particips) #
    #     best = None
    #     for participant in participants:
    #         if best is None or (
    #                 self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
    #             best = participant
    #     return best

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
            true_PF = []
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
            # self.utils.fast_nondominated_sort(population)
            self.utils.ens_ss_nondominated_sort(population) # 调用 ens_ss_nondominated_sort 进行非支配排序
            self.true_PF = population.fronts[0]
            print(f"Constructed true PF from history with {len(self.true_PF)} solutions")
        else:
            self.true_PF = []