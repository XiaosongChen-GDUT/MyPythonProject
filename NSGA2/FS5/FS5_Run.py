import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from NSGA2.FS.MOOVisualizer import MOOVisualizer
from NSGA2.AllocationVisualizer import AllocationVisualizer
from NSGA2.FS5.FS5_Utils import FS_Utils
from Program.DataModel import Model
from NSGA2.FS5.FS5_Evolution import Evolution
from NSGA2.FS5.FS5_Problem import Problem
import plotly.express as px
enter_node = [51, 348, 636, 925, 1110, 1620]# 入口点
if __name__ == '__main__':
    Model = Model()
    # path_planning = Path_Planning(Model)

    # 待分配货物信息
    pending_Loc = {1:{'enter_node': 51,'num':250,'rate':0.4,'quality':50,'dimension':'A'},
                   2:{'enter_node': 348,'num':150,'rate':0.3,'quality':100,'dimension':'B'},
                   3:{'enter_node': 636,'num':150,'rate':0.6,'quality':150,'dimension':'C'},
                   4:{'enter_node': 1110,'num':150,'rate':0.8,'quality':300,'dimension':'D'},
                   5:{'enter_node': 1620,'num':150,'rate':0.85,'quality':400,'dimension':'D'},
                   6:{'enter_node': 51,'num':100,'rate':0.5,'quality':200,'dimension':'C'},
                   7:{'enter_node': 348,'num':100,'rate':0.7,'quality':250,'dimension':'C'},
                   8:{'enter_node': 636,'num':200,'rate':0.9,'quality':350,'dimension':'B'},
                   9:{'enter_node': 925,'num':200,'rate':0.2,'quality':50,'dimension':'A'},
                   10:{'enter_node': 1110,'num':200,'rate':0.3,'quality':100,'dimension':'B'},
                   }

    # 货道信息
    asiles = Model.aisles  # 货道信息
    TopoGraph = Model.combined_graph    # 地图拓扑图
    free_aisles_keys = list(asiles.keys())  # 货道节点ID列表


    # allocated_nodes = [(3822, 1, 0), (3823, 1, 1), (3824, 1, 2), (3825, 1, 3), (3872, 1, 4), (3873, 1, 5), (3874, 1, 6), (3875, 1, 7), (3915, 1, 8), (3916, 1, 9), (3917, 1, 10), (3918, 1, 11), (3847, 1, 12), (3848, 1, 13), (3849, 1, 14), (3850, 1, 15), (3931, 1, 16), (3930, 1, 17), (3929, 1, 18), (3928, 1, 19), (3927, 1, 20), (3983, 1, 21), (3984, 1, 22), (3985, 1, 23), (3986, 1, 24), (3956, 1, 25), (3955, 1, 26), (3954, 1, 27), (3953, 1, 28), (3952, 1, 29), (3890, 1, 30), (3891, 1, 31), (3892, 1, 32), (3893, 1, 33), (3913, 1, 34), (3912, 1, 35), (3911, 1, 36), (3910, 1, 37), (3909, 1, 38), (3933, 1, 39),
    #                    (3934, 1, 40), (3935, 1, 41), (3936, 1, 42), (3958, 1, 43), (3959, 1, 44), (3960, 1, 45), (3961, 1, 46), (4008, 1, 47), (4009, 1, 48), (4010, 1, 49), (3858, 2, 50), (3857, 2, 51), (3856, 2, 52), (3855, 2, 53), (3854, 2, 54), (3853, 2, 55), (3852, 2, 56), (3969, 2, 57),
    #                    (3968, 2, 58), (3967, 2, 59), (3966, 2, 60), (3965, 2, 61), (3964, 2, 62), (3963, 2, 63), (3926, 2, 64), (3925, 2, 65), (3924, 2, 66), (3923, 2, 67), (3922, 2, 68), (3921, 2, 69), (3783, 2, 70), (3782, 2, 71), (3781, 2, 72), (3780, 2, 73), (3779, 2, 74), (3778, 2, 75), (3777, 2, 76), (3951, 2, 77), (3950, 2, 78), (3949, 2, 79), (3948, 2, 80), (3947, 2, 81), (3946, 2, 82), (3908, 2, 83), (3907, 2, 84), (3906, 2, 85), (3905, 2, 86), (3904, 2, 87), (3903, 2, 88), (3833, 2, 89), (3832, 2, 90), (3831, 2, 91), (3830, 2, 92), (3829, 2, 93), (3828, 2, 94), (3827, 2, 95), (4019, 2, 96),
    #                    (4018, 2, 97), (4017, 2, 98), (4016, 2, 99), (3133, 3, 100), (3132, 3, 101), (3131, 3, 102), (3130, 3, 103), (3129, 3, 104), (3128, 3, 105), (2292, 3, 106), (2291, 3, 107), (2290, 3, 108), (2289, 3, 109), (2288, 3, 110), (2287, 3, 111), (2286, 3, 112), (2190, 3, 113),
    #                    (2189, 3, 114), (2188, 3, 115), (2187, 3, 116), (2186, 3, 117), (2185, 3, 118), (2184, 3, 119), (2139, 3, 120), (2138, 3, 121), (2137, 3, 122), (2136, 3, 123), (2135, 3, 124), (2134, 3, 125), (2133, 3, 126), (2388, 3, 127), (2387, 3, 128), (2386, 3, 129), (2385, 3, 130), (2384, 3, 131), (2383, 3, 132), (2382, 3, 133), (2381, 3, 134), (2380, 3, 135), (2379, 3, 136), (2378, 3, 137), (2509, 3, 138), (2508, 3, 139), (2507, 3, 140), (2506, 3, 141), (2505, 3, 142), (2504, 3, 143), (2503, 3, 144), (2458, 3, 145), (2457, 3, 146), (2456, 3, 147), (2455, 3, 148), (2454, 3, 149)]
    #
    # # 可视化货位分配
    # vis = AllocationVisualizer(TopoGraph, title_prefix="Cargo Allocation - First Pareto Solution")
    # vis.visualize_allocation(allocated_nodes, asiles)

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
        num_of_generations=1000,
        num_of_tour_particips=30,    # 参与tournament的个体数
        tournament_prob=0.9,       # 选择tournament的概率
        mutation_param=0.2,       # 变异概率
        use_threshold_flag=False,    # 是否使用阈值
        crossover_type="crossover", # two_point两点交叉      order 顺序交叉  pmx部分交叉   sbx模拟二进制交叉  # crossover Deap的顺序交叉
        mutation_type="mutpolyn",       # swap交换变异    shuffle打乱变异  inversion反转变异     mutpolyn 多项式变异  mutate10%打乱变异
        init_type="random",
        )

    # 执行优化，front是Pareto 前沿（Pareto Front）的个体列表
    front, finished_generation_number , igd_history, hv_history, objective_history = evo.evolve()
    try:
        # 保存 NSGA-II 数据
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nsga2_file = f'nsga2_metrics_{timestamp}.csv'
        nsga2_data = pd.DataFrame({
            'Generation': list(range(len(hv_history))),
            'HV': hv_history,
            'IGD': igd_history,
            'F1': objective_history['F1'],
            'F2': objective_history['F2'],
            'F3': objective_history['F3'],
        })
        nsga2_data.to_csv(nsga2_file, index=False)
        print(f"NSGA-II metrics saved to {nsga2_file}")

        # 保存 Pareto 前沿到 CSV 文件
        FS_Utils.save_pareto_front(front=front ,true_pf_file="pareto_fronts.csv",run_id=4)
        # 提取 Pareto 前沿的目标值
        objectives = np.array([ind.objectives for ind in front])
        print(f"Objectives shape: {objectives.shape}")
        # print(f"Sample objectives: {objectives[:5]}")  # 打印前 5 个解的目标值
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

        #h绘制F1函数值变化曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(objective_history['F1']) + 1), objective_history['F1'], label='Weight', color='blue')
        plt.xlabel('Generation')
        plt.ylabel('Objective Value')
        plt.title(' Objective Value over Generations')
        plt.legend()
        plt.grid(True)
        plt.savefig('F1_curves.png', dpi=300)
        plt.show()

        #h绘制F2函数值变化曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(objective_history['F2']) + 1), objective_history['F2'], label='Balance', color='orange')
        plt.xlabel('Generation')
        plt.ylabel('Objective Value')
        plt.title(' Objective Value over Generations')
        plt.legend()
        plt.grid(True)
        plt.savefig('F2_curves.png', dpi=300)
        plt.show()


        #h绘制F3函数值变化曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(objective_history['F3']) + 1), objective_history['F3'], label='Efficiency', color='green')
        plt.xlabel('Generation')
        plt.ylabel('Objective Value')
        plt.title(' Objective Value over Generations')
        plt.legend()
        plt.grid(True)
        plt.savefig('F3_curves.png', dpi=300)
        plt.show()


        # 提取 Pareto 前沿的目标值
        objectives = np.array([ind.objectives for ind in front])
        obj_names = ['Weight', 'Balance', 'Efficiency']  # 根据你的目标函数命名
        # 初始化可视化器
        vis = MOOVisualizer(objectives, obj_names=obj_names)
        # 可视化选项

        # 2. 3D 散点图
        vis.plot_3d_scatter()
        plt.savefig('pareto_3d.png', dpi=300)
        plt.show()
        # plt.close()

        # #生成可旋转/缩放的三维图
        # fig = px.scatter_3d(
        #     x=objectives[:,0], y=objectives[:,1], z=objectives[:,2],
        #     labels={'x':'F1', 'y':'F2', 'z':'F3'},
        #     title="Interactive 3D Pareto Front"
        # )
        # fig.show()

        # # # 保存 Pareto 前沿到 CSV 文件
        # df = pd.DataFrame(objectives, columns=['F1', 'F2', 'F3'])
        # df.to_csv("pareto_front.csv", index=False)

        # 获取第一个个体的分配方案
        if front:
            first_ind = front[0]
            allocated_nodes = first_ind.allocated_nodes
            if len(allocated_nodes) != len(pending_Loc):
                warnings.warn("Warning: The number of allocated nodes is not equal to the number of pending locations.")
            print(f"Allocated Nodes for First Pareto Solution: {allocated_nodes}")

            # 可视化货位分配
            vis = AllocationVisualizer(TopoGraph, title_prefix="Cargo Allocation - First Pareto Solution")
            vis.visualize_allocation(allocated_nodes, asiles)
        else:
            print("No Pareto front solutions found.")
    except Exception as e:
        print(f"Error: {e}")

