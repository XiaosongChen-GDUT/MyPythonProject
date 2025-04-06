import random
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from Algorithm.PathPlanning import Path_Planning
from NSGA2.FS.MOOVisualizer import MOOVisualizer
from Program.DataModel import Model
from NSGA2.FS.FS_Evolution import Evolution
from NSGA2.FS.FS_Problem import Problem
import plotly.express as px
import pandas as pd
import networkx as nx
import geatpy as ea
import warnings


if __name__ == '__main__':
    Model = Model()
    path_planning = Path_Planning(Model)

    # 待分配货物信息
    pending_Loc = {1:{'enter_node': 51,'num':50,'rate':0.4,'quality':10,'dimension':'A'},
                   2:{'enter_node': 348,'num':50,'rate':0.3,'quality':50,'dimension':'B'},
                   3:{'enter_node': 636,'num':50,'rate':0.2,'quality':150,'dimension':'C'},
                   4:{'enter_node': 925,'num':30,'rate':0.1,'quality':100,'dimension':'C'},
                   5:{'enter_node': 1110,'num':20,'rate':0.5,'quality':300,'dimension':'D'},
                   # 6:{'enter_node': 1620,'num':30,'rate':0.6,'quality':60,'dimension':'B'},
                   }
    #数据预处理
    asiles = Model.aisles  # 货道信息
    TopoGraph = Model.combined_graph    # 地图拓扑图
    free_aisles_keys = list(asiles.keys())  # 货道节点ID列表

    # 初始化问题
    problem = Problem(
        num_of_variables=len(free_aisles_keys),
        variables_range=len(free_aisles_keys),
        model=Model,
        aisles_dict=asiles,
        pending_Loc=pending_Loc,
    )

    # 配置算法参数
    evo = Evolution(
        problem=problem,
        num_of_individuals=100,
        num_of_generations=100,
        num_of_tour_particips=5,    # 参与tournament的个体数
        tournament_prob=0.9,       # 选择tournament的概率
        mutation_param=0.2,       # 变异概率
        use_threshold_flag=True,    # 是否使用阈值
        )

    # 执行优化
    front, finished_generation_number , igd_history, hv_history= evo.evolve()
    print('迭代次数:', finished_generation_number)
    # print('最后一代pareto前沿:', front)
    # 提取 Pareto 前沿的目标值
    objectives = np.array([ind.objectives for ind in front])
    print(f"Objectives shape: {objectives.shape}")
    print(f"Sample objectives: {objectives[:5]}")  # 打印前 5 个解的目标值
    if objectives.size == 0 or np.any(np.isnan(objectives)):
        raise ValueError("Pareto front contains invalid or empty objectives")

    # 绘制 IGD 和 HV 曲线
    plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(igd_history) + 1), igd_history, label='IGD', color='blue')
    plt.plot(range(1, len(hv_history) + 1), hv_history, label='HV', color='orange')
    plt.xlabel('Generation')
    plt.ylabel('Metric Value')
    plt.title(' HV over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig('hv_curves.png', dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(igd_history) + 1), igd_history, label='IGD', color='blue')
    # plt.plot(range(1, len(hv_history) + 1), hv_history, label='HV', color='orange')
    plt.xlabel('Generation')
    plt.ylabel('Metric Value')
    plt.title(' IGD over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig('IGD_curves.png', dpi=300)
    plt.show()

    # 提取 Pareto 前沿的目标值
    objectives = np.array([ind.objectives for ind in front])
    obj_names = ['Weight', 'Balance', 'Efficiency']  # 根据你的目标函数命名
    # 初始化可视化器
    vis = MOOVisualizer(objectives, obj_names=obj_names)

    # 2. 3D 散点图
    vis.plot_3d_scatter()
    plt.savefig('pareto_3d.png', dpi=300)
    plt.show()
    # plt.close()

    #生成可旋转/缩放的三维图
    fig = px.scatter_3d(
        x=objectives[:,0], y=objectives[:,1], z=objectives[:,2],
        labels={'x':'F1', 'y':'F2', 'z':'F3'},
        title="Interactive 3D Pareto Front"
    )
    fig.show()

    # # 保存 Pareto 前沿到 CSV 文件
    df = pd.DataFrame(objectives, columns=['F1', 'F2', 'F3'])
    df.to_csv("pareto_front.csv", index=False)




