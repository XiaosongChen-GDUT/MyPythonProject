# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea

class MyProblem(ea.Problem):
    def __init__(self):
        name = 'MyProblem'  # 问题名字
        M = 2  # 目标维数
        maxormins = [1] * M  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
        Dim = 2  # 决策变量维数
        varTypes = [0] * Dim  # 决策变量的类型，0：实数；1：整数
        lb = [0] * Dim  # 决策变量下界
        ub = [5,3]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    # 目标函数
    def aimFunc(self, pop):
        Vars = pop.Phen  # 得到决策变量矩阵
        x1 = Vars[:, [0]]  # 取出第1列
        x2 = Vars[:, [1]]  # 取出第2列
        f1 =  4*x1**2 + 4*x2**2  # 目标函数1
        f2 =  (x1 - 5)**2 + (x2 - 5)**2 # 目标函数2
        # 采用可行性法则处理约束条件
        pop.CV = np.hstack([(x1 - 5)**2 + x2**2 - 25,-(x1 - 8)**2 - (x2 - 3)**2 + 7.7])
        # 把求得的目标函数值赋值给种群pop的ObjV
        pop.ObjV = np.hstack([f1, f2])

    def calReferObjV(self): # 计算全局最优解
        N = 10000  # 欲得到10000个真实前沿点
        x1 = np.random.uniform(0, 5, N)  # 随机生成10000个x1
        x2 = x1.copy()
        x2[x1 >= 3] = 3  # 使x2满足约束条件
        return np.vstack((4 * x1**2 + 4 * x2**2,
                          (x1 - 5)**2 + (x2 - 5)**2)).T
# 实例化问题对象
Problem = MyProblem()
Encoding = 'RI'  # 编码方式
NIND = 100  # 种群规模
Field = ea.crtfld(Encoding, Problem.varTypes, Problem.ranges, Problem.borders)  # 创建区域描述器
population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
Algorithm = ea.moea_NSGA2_templet(Problem, population)  # 实例化算法模板对象
Algorithm.mutOper.Pm = 0.5  # 设置变异算子的变异概率
Algorithm.recOper.XOVR = 0.9  # 设置重组算子的交叉概率
Algorithm.MAXGEN = 200  # 设置最大进化代数
Algorithm.logTras = 1  # 设置是否记录日志，1：记录日志；0：不记录日志
Algorithm.drawing = 2  # 设置绘图方式，0：不绘图；1：绘制结果图；2：绘制目标空间过程动画
"""==========================调用算法模板进行种群进化==============
调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。
NDSet是一个种群类Population的对象。
NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
详见Population.py中关于种群类的定义。
"""
[NDSet, population] = Algorithm.run()  # 运行算法模板，得到结果
NDSet.save() # 把非支配种群的信息保存到文件中
"""===========================输出结果========================"""
print('用时：%f 秒'%(Algorithm.passTime))
print('非支配个体数：%d 个'%(NDSet.sizes) if NDSet.sizes!= 0 else print('没有找到可行解！'))
if Algorithm.log is not None and NDSet.sizes != 0:
    print('GD', Algorithm.log['gd'][-1])
    print('IGD', Algorithm.log['igd'][-1])
    print('HV', Algorithm.log['hv'][-1])
    print('Spacing', Algorithm.log['spacing'][-1])
    """======================进化过程指标追踪分析=================="""
    metricName = [['igd'], ['hv']]
    Metrics = np.array([Algorithm.log[metricName[i][0]] for i in
                        range(len(metricName))]).T
    # 绘制指标追踪分析图
    ea.trcplot(Metrics, labels=metricName,title=metricName)

