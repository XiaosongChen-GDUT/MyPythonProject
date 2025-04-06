import geatpy as ea
import numpy as np


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 优化目标个数
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 1  # 初始化Dim（决策变量维数）
        varTypes = [0]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [-10]  # 决策变量下界
        ub = [10]  # 决策变量上界
        lbin = [1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 获取决策变量矩阵，它等于种群的表现型矩阵Phen
        f1 = Vars ** 2
        f2 = (Vars - 2) ** 2
        pop.ObjV = np.hstack([f1, f2])  # 计算目标函数值矩阵，赋值给种群对象的ObjV属性
        pop.CV = -Vars ** 2 + 2.5 * Vars - 1.5  # 构建违反约束程度矩阵，赋值给种群对象的CV属性

# 实例化问题对象
problem = MyProblem()
# 构建算法
algorithm = ea.moea_NSGA2_templet(problem,
                                  ea.Population(Encoding='RI', NIND=50),
                                  MAXGEN=200,  # 最大进化代数
                                  logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
# 求解
res = ea.optimize(algorithm, seed=1, verbose=False, drawing=1, outputMsg=True, drawLog=False, saveFlag=False)
