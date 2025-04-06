import numpy as np
from NSGA2.FS.FS_Individual import Individual
from NSGA2.FS.FS_Population import Population
import random
import geatpy as ea

class FS_Utils:
    #接收problem实例，设置种群大小，选择参加锦标赛的个体数目，锦标赛的概率，变异概率
    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9, mutation_param=0.01):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.mutation_param = mutation_param
        num_variables = self.problem.num_of_variables   # 变量个数

        self.recOper = ea.Recsbx(XOVR=1, n=20,Parallel=True)  # 生成模拟二进制交叉算子对象
        self.mutOper = ea.Mutpolyn(Pm=1 / num_variables, DisI=20)  # 生成多项式变异算子对象
        # 构造 FieldDR
        lb = np.ones(num_variables)  # 下界全为 0
        ub = np.full(num_variables,num_variables)  # 上界为 num_of_variables
        varTypes = np.ones(num_variables)  # 变量类型为离散 (1)
        self.FieldDR = np.vstack([lb, ub, varTypes])

    #创建初始种群
    def create_initial_population(self):
        population = Population()
        for _ in range(self.num_of_individuals):
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            population.append(individual)
        return population

    #非支配排序
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

    #产生子代
    def create_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population) #锦标赛选择父代
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.__tournament(population)
            #两点交叉
            child1, child2 = self.__crossover(parent1, parent2)
            # child1, child2 = self.SBX__crossover(parent1, parent2)
            #变异
            if self.__choose_with_prob(self.mutation_param):
                child1 = self.__mutate(child1)
                # child1 = self.mutpolyn__mutate(child1)
            if self.__choose_with_prob(self.mutation_param):
                child2 = self.__mutate(child2)
                # child2 = self.mutpolyn__mutate(child2)
            #计算目标函数值
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)

        return children

    def SBX__crossover(self, parent1, parent2):
        """使用模拟二进制交叉（SBX）生成子代"""

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

    def mutpolyn__mutate(self, child):
        """使用多项式变异生成新个体"""

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

    #两点交叉
    def __crossover(self, parent1, parent2):
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        crossover_point1 = random.randint(0, len(parent1.features) - 2)
        crossover_point2 = random.randint(crossover_point1 + 1, len(parent1.features) - 1)
        child1.features = parent1.features[:crossover_point1] + parent2.features[crossover_point1:crossover_point2] + parent1.features[crossover_point2:]
        child2.features = parent2.features[:crossover_point1] + parent1.features[crossover_point1:crossover_point2] + parent2.features[crossover_point2:]
        return child1, child2

    #两点变异：交换变异，通过交换个体中的两个基因位置的值，来产生新的个体
    def __mutate(self, child):
        mutation_points1 = random.randint(0,self.problem.num_of_variables - 1)
        mutation_points2 = random.randint(0,self.problem.num_of_variables - 1)
        while mutation_points1 == mutation_points2:
            mutation_points2 = random.randint(0,self.problem.num_of_variables - 1)
        child.features[mutation_points1], child.features[mutation_points2] = child.features[mutation_points2], child.features[mutation_points1]
        return child

    #锦标赛
    def __tournament(self, population):
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (
                    self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant
        return best

    #随机决定是否发生变异/交叉
    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False