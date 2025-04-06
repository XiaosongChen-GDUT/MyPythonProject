
'''种群类：用于存储种群中所有个体的集合'''
class Population:

    def __init__(self):
        self.population = []     # 个体集
        self.fronts = []         # 种群的前沿

    def __len__(self):
        return len(self.population)   # 个体数
    #返回种群的迭代器对象，允许用户通过for循环遍历种群中的每个个体。这个方法使得Population对象可以像列表一样被迭代。
    def __iter__(self):
        return self.population.__iter__()   # 迭代器

    def extend(self, new_individuals):   # 扩展种群
        self.population.extend(new_individuals)

    #追加个体到self.population列表的末尾
    def append(self, new_individual):   # 追加个体
        self.population.append(new_individual)

    #计算全局目标值的最小和最大值
    def compute_global_objective_bounds(self):
        if not self.population:
            return None, None
        num_objs = len(self.population[0].objectives)
        global_min = [float('inf')] * num_objs
        global_max = [-float('inf')] * num_objs
        for ind in self.population:
            for m in range(num_objs):
                val = ind.objectives[m]
                if val < global_min[m]:
                    global_min[m] = val
                if val > global_max[m]:
                    global_max[m] = val
        return global_min, global_max