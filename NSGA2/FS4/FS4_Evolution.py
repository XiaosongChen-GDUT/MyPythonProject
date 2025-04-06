import numpy as np
from geatpy.core.indicator import IGD, HV  # 导入IGD指标计算工具
# from geatpy.core.indicator import IGD
# from pymoo.indicators.hv import HV    # 导入Pymoo的HV指标计算工具
from tqdm import tqdm       # 进度条工具

from NSGA2.FS4.FS4_Utils import FS_Utils
from NSGA2.FS4.FS4_Population import Population


class Evolution:

    def __init__(self, problem, num_of_generations=500, num_of_individuals=100, num_of_tour_particips=2,
                 tournament_prob=0.9, mutation_param=0.01, use_threshold_flag=True,
                 crossover_type="two_point", mutation_type="swap", init_type="random"):
        """
        初始化进化类，添加可配置的交叉、变异和种群初始化方法。

        :param problem: 问题实例
        :param num_of_generations: 最大迭代次数
        :param num_of_individuals: 种群大小
        :param num_of_tour_particips: 锦标赛参与个体数
        :param tournament_prob: 锦标赛选择概率
        :param mutation_param: 变异概率
        :param use_threshold_flag: 是否使用阈值停止条件
        :param crossover_type: 交叉类型
        :param mutation_type: 变异类型
        :param init_type: 种群初始化类型
        """
        self.utils = FS_Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob,
                              mutation_param, crossover_type, mutation_type, init_type)
        self.use_threshold_flag = use_threshold_flag
        self.population = None
        self.num_of_generations = num_of_generations
        self.num_of_individuals = num_of_individuals
        self.igd_history = []
        self.hv_history = []
        # 记录每代 Pareto 前沿的目标函数最优值
        self.objective_history = {'F1': [], 'F2': [], 'F3': []}
        self.true_pareto_front = FS_Utils.load_true_pareto_front(true_pf_file="pareto_fronts.csv")



    def calculate_IGD(self, ObjV, PF):
        """
            indicator.IGD : function - 计算多目标优化反转世代距离(IGD)评价指标的值
            语法:
            igd = IGD(ObjV, PF)
            igd = IGD(ObjV, PF, Parallel)
        描述:
        IGD是一个综合评价指标，其值越小越好。
        输入参数:
        ObjV : array - 目标函数值矩阵。
        PF   : array - 真实全局帕累托最优解的目标函数值矩阵，
        注意: 实际上可以传入不是真实PF的目标函数值矩阵，
        可以是经验所得的非支配解的目标函数值。
        Parallel : bool  - (可选参数)表示是否采用并行计算，缺省或为None时默认为False。
        输出参数:
        igd : float - IGD评价指标。
        """
        objectives = np.array([ind.objectives for ind in PF])
        if objectives.size == 0:
            return 0.0
        igd = IGD(ObjV, objectives)
        return igd

    #超体积
    def calculate_hv(self, pareto_front):
        """
         语法:
        hv = HV(ObjV)
        hv = HV(ObjV, PF)
        hv = HV(ObjV, PF, Parallel)
    输入参数:
        ObjV : array - 目标函数值矩阵。
        PF   : array - 真实全局帕累托最优解的目标函数值矩阵，
                       注意: 实际上可以传入不是真实PF的目标函数值矩阵，
                       可以是经验所得的非支配解的目标函数值。
        Parallel : bool  - (可选参数)表示是否采用并行计算，缺省或为None时默认为False。
    输出参数:
        hv : float - 多目标优化问题的全局历史最优值。
        :param pareto_front:
        :return:
        """
        # 提取第一层 Pareto 前沿的目标值
        objectives = np.array([ind.objectives for ind in pareto_front])
        if objectives.size == 0:
            return 0.0
        # 归一化目标值（可选）
        # # 假设你知道目标的最优值和最差值，或者通过解集估算
        # min_values = np.min(objectives, axis=0)
        # max_values = np.max(objectives, axis=0)
        # if np.any(max_values == min_values):  # 避免除以 0
        #     max_values = max_values + 1e-6
        # normalized_objectives = (objectives - min_values) / (max_values - min_values)
        # 选择参考点（基于归一化后的最大值）
        # ref_point = np.max(normalized_objectives, axis=0) + 0.1  # 增加一个小的偏移量
        # 如果不归一化，可以直接使用原始目标值的最大值
        # ref_point = np.max(objectives, axis=0) + 1.0
        # # 使用 Pymoo 的 HV 计算
        # hv_calculator = HV(ref_point=ref_point)
        # hv = hv_calculator(normalized_objectives)  # 或者使用 objectives（如果不归一化）
        hv = HV(objectives)
        return hv

    def threshold_calculator(self, now_pare_to_front, last_pare_to_front):
        num_objectives = 3  # 目标数量

        # 存储当前和上一代的目标值
        objectives_now = [[] for _ in range(num_objectives)]
        objectives_last = [[] for _ in range(num_objectives)]

        # 填充数据
        for j in now_pare_to_front:
            for i in range(num_objectives):
                objectives_now[i].append(j.objectives[i])
        for j in last_pare_to_front:
            for i in range(num_objectives):
                objectives_last[i].append(j.objectives[i])

        # 归一化参数
        norm_params = []
        for i in range(num_objectives):
            f_max = max(objectives_now[i])
            f_min = min(objectives_now[i])
            norm_params.append((f_min, f_max))

        # 归一化处理
        normalized_now = []
        for j in now_pare_to_front:
            norm_point = []
            for i in range(num_objectives):
                f_min, f_max = norm_params[i]
                if f_max == f_min:
                    norm_val = 0.0
                else:
                    norm_val = (j.objectives[i] - f_min) / (f_max - f_min)
                norm_point.append(norm_val)
            normalized_now.append(norm_point)

        normalized_last = []
        for j in last_pare_to_front:
            norm_point = []
            for i in range(num_objectives):
                f_min, f_max = norm_params[i]
                if f_max == f_min:
                    norm_val = 0.0
                else:
                    norm_val = (j.objectives[i] - f_min) / (f_max - f_min)
                norm_point.append(norm_val)
            normalized_last.append(norm_point)

        # 计算最大差异度
        differences = []
        for i in range(num_objectives):
            f_max_last = max(objectives_last[i])
            f_min_last = min(objectives_last[i])
            f_max_now = norm_params[i][1]
            f_min_now = norm_params[i][0]

            if (f_max_now - f_min_now) == 0:
                diff_max = 0
                diff_min = 0
            else:
                diff_max = (f_max_last - f_max_now) / (f_max_now - f_min_now)
                diff_min = (f_min_last - f_min_now) / (f_max_now - f_min_now)

            differences.append(max(diff_max, diff_min))

        difference_max = max(differences)
        difference_min = min(differences)
        """
        indicator.IGD : function - 计算多目标优化反转世代距离(IGD)评价指标的值
        语法:
            igd = IGD(ObjV, PF)
            igd = IGD(ObjV, PF, Parallel)
        描述:
            IGD是一个综合评价指标，其值越小越好。
        输入参数:
            ObjV : array - 目标函数值矩阵。
            PF   : array - 真实全局帕累托最优解的目标函数值矩阵，
                           注意: 实际上可以传入不是真实PF的目标函数值矩阵，
                           可以是经验所得的非支配解的目标函数值。
            Parallel : bool  - (可选参数)表示是否采用并行计算，缺省或为None时默认为False。"""
        # 计算IGD
        # ind = IGD(np.array(normalized_now))
        # igd = ind(np.array(normalized_last))
        igd = IGD(np.array(normalized_now), np.array(normalized_last),Parallel=True)
        print(f"IGD: {igd}, Shape of now: {np.array(normalized_now).shape}, Shape of last: {np.array(normalized_last).shape}")
        return difference_max, difference_min, igd



    # 动态精英比例函数 前期高比例（加速收敛），后期低比例（提升多样性）
    def dynamic_elite_ratio(self,current_gen, max_gen, initial_ratio=0.1, final_ratio=0.02):
        return initial_ratio - (initial_ratio - final_ratio) * (current_gen / max_gen)

    '''
    初始化种群并计算其目标函数值。
    对种群进行非支配排序，并根据拥挤距离选择个体。
    通过锦标赛选择、交叉和变异生成下一代个体。
    如果启用了阈值停止条件，当连续几代的前沿差异度小于设定阈值时停止进化。
    返回最终的前沿个体集和实际进化代数
    '''
    def evolve(self):
        """
        进化过程，记录每代 HV 和目标函数最优值。
        :return: Pareto 前沿、实际迭代次数、IGD 历史、HV 历史、目标函数历史
        """
        self.population = self.utils.create_initial_population()    # 初始化种群
        # self.population = self.utils.create_initial_population_good_lattice_points()    # 初始化佳集点种群
        # self.utils.fast_nondominated_sort(self.population)          # 非支配排序
        self.utils.ens_ss_nondominated_sort(self.population)          # ENS-SS 非支配排序
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)            # 计算拥挤距离
        children = self.utils.create_children(self.population)       # 生成下一代个体+计算目标值（通过选择、交叉和变异）
        pare_to_front_of_last_generation = []                        # 上一代前沿个体集
        pare_to_front_of_check_threshold = []                       # 用于检查阈值停止条件的前沿个体集
        start_threshold = 5                                       # 阈值停止条件开始的代数
        window_size = 20                                            # 阈值停止条件检查窗口大小
        threshold_flag = False                                     # 阈值停止条件是否满足
        finished_generation_number = 0                             # 实际进化代数

        # ========精英策略：设置精英比例（例如 10%）========
        # elite_ratio = 0.1
        # 计算精英个体数量
        # number_of_all_fronts = 0  # 所有前沿的个数
        # returned_population = None
        #tqdm是一个Python库，用于在循环中动态显示进度条，使得你可以直观地看到当前循环的进度以及预计的剩余时间。
        for i in tqdm(range(self.num_of_generations)):
            #动态精英比例
            current_elite_ratio = self.dynamic_elite_ratio(
                current_gen=i,
                max_gen=self.num_of_generations,
                initial_ratio=0.1,
                final_ratio=0.02
            )
            elite_size = int(self.num_of_individuals * current_elite_ratio)
            self.population.extend(children)                        # 合并父代和子代，将下一代个体加入种群
            # self.utils.fast_nondominated_sort(self.population)       # 非支配排序
            self.utils.ens_ss_nondominated_sort(self.population)       # ENS-SS 非支配排序
            new_population = Population()                            # 初始化新种群
            # ============= 加入精英保留策略 ============
            # 精英保留策略：选择前沿个体作为精英个体，并将其直接加入新种群
            if pare_to_front_of_last_generation and elite_size > 0:
                # 按拥挤距离排序，选择前 elite_size 个个体作为精英
                elite_candidates = sorted(
                    pare_to_front_of_last_generation,
                    key=lambda individual: individual.crowding_distance,
                    reverse=True
                )
                elite_individuals = elite_candidates[:min(elite_size, len(elite_candidates))]
                # 去重逻辑（基于特征向量）
                unique_elite = []
                for ind in elite_individuals:
                    if not any(np.allclose(ind.features, u.features, atol=1e-3) for u in unique_elite):
                        unique_elite.append(ind)
                elite_individuals = unique_elite[:min(elite_size, len(unique_elite))]
                # 将精英个体直接加入新种群
                new_population.extend(elite_individuals)
                print(f"- 精英比例：{current_elite_ratio}, 精英个体数量：{elite_size}, 实际添加精英个体：{len(elite_individuals)}")
                # ----------- 精英保留策略结束 ------------------
            #====================环境选择==================
            front_num = 0
            # 环境选择：按前沿层级填充新种群（考虑精英个体后剩余的个体数）
            remaining_size = self.num_of_individuals - len(new_population)  # 剩余需要填充的个体数
            while (front_num < len(self.population.fronts) and  #确保当前前沿编号不超过总前沿数量。
                   remaining_size > 0 and   # 确保剩余需要填充的个体数大于0。
                   len(self.population.fronts[front_num]) <= remaining_size):  # 确保当前前沿的个体数不超过剩余需要填充的个体数。
                print("当前前沿编号：", front_num, "当前前沿个体数：", len(self.population.fronts[front_num]), "剩余需要填充的个体数：", remaining_size)
                #计算当前前沿的拥挤距离
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                ## 将当前前沿的个体加入新种群
                new_population.extend(self.population.fronts[front_num])
                # 更新剩余需要填充的个体数
                remaining_size -= len(self.population.fronts[front_num])
                #移动到下一层级
                front_num += 1
            ## 处理最后一个前沿（按拥挤距离选择）
            if front_num < len(self.population.fronts) and len(new_population) < self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                # 选择拥挤距离最高的个体
                self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
                new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals - len(new_population)])
            self.population = new_population                       # 更新种群
            #====================环境选择结束================

            # ============= 添加随机移民策略 ============
            # 替换种群中 10% 的个体为随机新个体（保留精英和优质解）
            immigration_rate = 0.05
            num_immigrants = int(self.num_of_individuals * immigration_rate)
            immigrants = self.utils.create_immigrants_population(size=num_immigrants)  # 生成新个体
            # 替换种群中拥挤距离最小的个体（避免破坏精英） # 按拥挤距离升序排列（最小在前）
            sorted_individuals = sorted(
                self.population.population,
                key=lambda individual: individual.crowding_distance,
                reverse=False
            )
            sorted_individuals[:num_immigrants] = immigrants          # 替换最差的个体
            self.population.population = sorted_individuals           # 更新种群
            # ============= 随机移民策略结束 ============

            # self.utils.fast_nondominated_sort(self.population)     # 再次进行非支配排序
            self.utils.ens_ss_nondominated_sort(self.population)     # 再次进行ENS-SS 非支配排序
            all_crowding_distances = []## 初始化一个列表来存储所有个体的拥挤距离
            for front in self.population.fronts:                   # 计算拥挤距离
                self.utils.calculate_crowding_distance(front)
                # 将当前前沿中所有个体的拥挤距离添加到列表中
                for ind in front:
                    all_crowding_distances.append(ind.crowding_distance)
            # 计算平均拥挤距离
            average_crowding_distance = np.mean(all_crowding_distances)
            # 记录当前帕累托前沿
            pare_to_front_of_last_generation = self.population.fronts[0]

            # 记录目标函数最优值（取最小值）
            if pare_to_front_of_last_generation:
                objectives = np.array([ind.objectives for ind in pare_to_front_of_last_generation])
                # # 计算当前帕累托前沿的IGD值
                # igd = self.calculate_IGD(objectives, self.true_pareto_front)
                # self.igd_history.append(igd)
                # 计算当前帕累托前沿的HV值
                hv = self.calculate_hv(pare_to_front_of_last_generation)
                self.hv_history.append(hv)
                print(f"第 {i+1} 次迭代,  HV: {hv},平均拥挤距离：{average_crowding_distance}")
                self.objective_history['F1'].append(np.min(objectives[:, 0]))
                self.objective_history['F2'].append(np.min(objectives[:, 1]))
                self.objective_history['F3'].append(np.min(objectives[:, 2]))
            else:
                self.objective_history['F1'].append(float('inf'))
                self.objective_history['F2'].append(float('inf'))
                self.objective_history['F3'].append(float('inf'))
            finished_generation_number = i                           # 记录实际进化代数
            # 动态阈值停止条件
            if(self.use_threshold_flag):
                if i > start_threshold :
                    pare_to_front_of_check_threshold.append(self.population.fronts[0])
                if i == start_threshold + window_size :  # 阈值停止条件检查窗口大小
                    # 计算前沿差异度和IGD指标
                    differences_max = []
                    differnces_min = []
                    igds = []
                    for window_i in range(1,window_size):    # 计算前 window_size 代的前沿差异度和IGD指标
                        difference_max, differnce_min, igd = self.threshold_calculator(
                            now_pare_to_front  = pare_to_front_of_check_threshold[window_i],
                            last_pare_to_front = pare_to_front_of_check_threshold[window_i - 1])
                        differences_max.append(difference_max)
                        differnces_min.append(differnce_min)
                    # 阈值停止条件判断
                    if ((max(differences_max) < 0.03) and (max(differnces_min) < 0.01) and (max(igds) < 0.001)) :
                        threshold_flag = True
                    # 重置阈值停止条件检查窗口
                    pare_to_front_of_check_threshold = []    #
                    start_threshold += window_size
                if threshold_flag :
                    print("满足阈值停止条件，停止进化")
                    break
            children = self.utils.create_children(self.population)  # 生成下一代个体

        # pareto_allocations = [
        #     {"features": ind.features, "objectives": ind.objectives, "allocated_nodes": ind.allocated_nodes}
        #     for ind in pare_to_front_of_last_generation
        # ]
        # 返回最终的前沿个体集和实际进化代数
        return (
            pare_to_front_of_last_generation,    # Pareto 前沿个体集
            finished_generation_number,         # 实际进化代数
            self.igd_history,          # IGD 历史
            self.hv_history,          # HV 历史
            self.objective_history   # 目标函数历史
        )