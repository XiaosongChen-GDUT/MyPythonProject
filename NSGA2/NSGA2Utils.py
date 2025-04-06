import random

from NSGA2.Population import Population


class NSGA2Utils:
    #接收problem实例，设置种群大小，选择参加锦标赛的个体数目，锦标赛的概率，变异概率
    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9, mutation_param=0.01):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.mutation_param = mutation_param


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

    #计算拥挤度
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
            parent1 = self.__tournament(population)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.__tournament(population)
            child1, child2 = self.__crossover(parent1, parent2)
            if self.__choose_with_prob(self.mutation_param):
                child1 = self.__mutate(child1)
            if self.__choose_with_prob(self.mutation_param):
                child2 = self.__mutate(child2)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)

        return children
    #交叉
    def __crossover(self, parent1, parent2):
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        crossover_point1 = random.randint(0, len(parent1.features) - 2)
        crossover_point2 = random.randint(crossover_point1 + 1, len(parent1.features) - 1)
        child1.features = parent1.features[:crossover_point1] + parent2.features[crossover_point1:crossover_point2] + parent1.features[crossover_point2:]
        child2.features = parent2.features[:crossover_point1] + parent1.features[crossover_point1:crossover_point2] + parent2.features[crossover_point2:]
        return child1, child2
    #变异
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