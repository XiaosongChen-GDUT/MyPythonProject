
from NSGA2.Individual import Individual
import random
'''
用于初始化Problem类的对象，
设置目标函数数量、变量数量、目标函数列表以及变量的取值范围。
'''

class Problem:

    def __init__(self, objectives, num_of_variables, variables_range):
        '''
        num_of_objectives: 表示问题中有多少个目标函数。
        num_of_variables: 表示问题中每个个体需要优化的变量数量。
        objectives: 是一个目标函数的列表，这些函数将在优化过程中被评估。
        variables_range: 表示每个变量可以取的值的范围，这里假设是从1到variables_range之间的整数。'''
        self.num_of_objectives = len(objectives)    #目标个数
        self.num_of_variables = num_of_variables    #变量个数
        self.objectives = objectives                 #目标函数列表
        self.variables_range = variables_range      #变量取值范围



    # 随机生成货道索引的随机序列
    def generate_individual(self):
        individual = Individual()
        # 随机生成变量
        individual.features = [random.randint(1,self.variables_range) for _ in range(self.num_of_variables)]
        return individual

    # 计算个体的目标函数值
    def calculate_objectives(self, individual):
        # 计算目标函数值,遍历所有的目标函数，并将这些目标函数应用于个体的特征
        individual.objectives = [f(individual.features) for f in self.objectives]