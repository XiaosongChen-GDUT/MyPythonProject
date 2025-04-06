import numpy as np
from geatpy.core.indicator import IGD, HV  # 导入IGD指标计算工具
from tqdm import tqdm       # 进度条工具

from NSGA2.FS.FS_Utils import FS_Utils
from NSGA2.FS.FS_Population import Population


class Evolution:

    def __init__(self, problem, num_of_generations=500, num_of_individuals=100, num_of_tour_particips=2,
                 tournament_prob=0.9, mutation_param=0.01, use_threshold_flag = True):
        self.utils = FS_Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob,
                                mutation_param)
        self.use_threshold_flag = use_threshold_flag    # 是否启用阈值停止条件
        self.population = None
        self.num_of_generations = num_of_generations    # 进化代数
        self.num_of_individuals = num_of_individuals    # 个体数
        self.igd_history = []  # 记录 IGD
        self.hv_history = []   # 记录 HV

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
        objectives = np.array([ind.objectives for ind in pareto_front])
        if objectives.size == 0:
            return 0.0
        # # 设置参考点为目标值的最大值加偏移量（例如 10%）
        # ref_point = np.max(objectives, axis=0) * 1.1
        # hv_calculator = HV(ref_point)
        # hv = hv_calculator(objectives)
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


    '''
    初始化种群并计算其目标函数值。
    对种群进行非支配排序，并根据拥挤距离选择个体。
    通过锦标赛选择、交叉和变异生成下一代个体。
    如果启用了阈值停止条件，当连续几代的前沿差异度小于设定阈值时停止进化。
    返回最终的前沿个体集和实际进化代数
    '''
    def evolve(self):
        self.population = self.utils.create_initial_population()    # 初始化种群
        self.utils.fast_nondominated_sort(self.population)          # 非支配排序
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)            # 计算拥挤距离
        children = self.utils.create_children(self.population)       # 生成下一代个体
        pare_to_front_of_last_generation = []                        # 上一代前沿个体集
        pare_to_front_of_check_threshold = []                       # 用于检查阈值停止条件的前沿个体集
        start_threshold = 30                                       # 阈值停止条件开始的代数
        window_size = 20                                            # 阈值停止条件检查窗口大小
        threshold_flag = False  # 阈值停止条件是否满足
        finished_generation_number = 0  # 实际进化代数

        # number_of_all_fronts = 0  # 所有前沿的个数
        # returned_population = None
        #tqdm是一个Python库，用于在循环中动态显示进度条，使得你可以直观地看到当前循环的进度以及预计的剩余时间。
        for i in tqdm(range(self.num_of_generations)):
            # 计算全局目标值范围
            self.population.extend(children)                        # 合并父代和子代，将下一代个体加入种群
            self.utils.fast_nondominated_sort(self.population)       # 非支配排序
            new_population = Population()
            front_num = 0
            # 计算全局目标值范围
            # global_min, global_max = self.population.compute_global_objective_bounds()
            # 环境选择：按前沿层级填充新种群
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:  # 选择拥挤距离最高的个体
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                # self.utils.calculate_crowding_distance_Improved(front, global_min, global_max)
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            ## 处理最后一个前沿（按拥挤距离选择）
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            # 选择拥挤距离最高的个体
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals - len(new_population)])
            self.population = new_population                       # 更新种群
            self.utils.fast_nondominated_sort(self.population)     # 非支配排序
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            # 记录当前帕累托前沿
            pare_to_front_of_last_generation = self.population.fronts[0]
            # 计算当前帕累托前沿的HV值
            hv = self.calculate_hv(pare_to_front_of_last_generation)
            self.hv_history.append(hv)
            finished_generation_number = i
            # 动态阈值停止条件
            if(self.use_threshold_flag):
                if i > start_threshold :
                    pare_to_front_of_check_threshold.append(self.population.fronts[0])
                if i == start_threshold + window_size :  # 阈值停止条件检查窗口大小
                    # 计算前沿差异度和IGD指标
                    differences_max = []
                    differnces_min = []
                    igds = []
                    for window_i in range(1,window_size):
                        difference_max, differnce_min, igd = self.threshold_calculator(now_pare_to_front  = pare_to_front_of_check_threshold[window_i], last_pare_to_front = pare_to_front_of_check_threshold[window_i - 1])
                        differences_max.append(difference_max)
                        differnces_min.append(differnce_min)
                        self.igd_history.append(igd)

                        igds.append(igd)
                    # 阈值停止条件判断
                    if ((max(differences_max) < 0.03) and (max(differnces_min) < 0.01) and (max(igds) < 0.001)) :
                        threshold_flag = True
                    # 重置阈值停止条件检查窗口
                    pare_to_front_of_check_threshold = []
                    start_threshold += window_size
                if threshold_flag :
                    print("满足阈值停止条件，停止进化")
                    break
            children = self.utils.create_children(self.population)  # 生成下一代个体
        # 返回最终的前沿个体集和实际进化代数
        return  pare_to_front_of_last_generation, finished_generation_number,self.igd_history, self.hv_history