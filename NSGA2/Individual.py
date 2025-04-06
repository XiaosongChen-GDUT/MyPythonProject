
'''个体类：在多目标优化中，支配的概念是指：
如果个体A在所有目标上的表现都不劣于个体B，
并且至少在一个目标上优于个体B并且至少在一个目标上优于个体B，
那么我们说个体A支配个体B。'''
class Individual(object):

    def __init__(self):
        self.rank = None        # 秩
        self.crowding_distance = None   # 累计距离
        self.domination_count = None    # 支配计数
        self.dominated_solutions = None  # 被支配的个体
        self.features = None             # 个体的特征
        self.objectives = None           # 个体的目标值
        self.allocated_nodes = []   # 新增：缓存解码后的货位分配

    def __eq__(self, other):                    # 重载 == 操作符
        if isinstance(self, other.__class__):    # 判断是否是同一类对象
            return self.features == other.features  # 判断特征是否相同
        return False

    def dominates(self, other_individual):     # 判断是否支配
        and_condition = True
        or_condition = False
        # 逐一比较目标值
        for first, second in zip(self.objectives, other_individual.objectives):
            # 若第一个目标值小于第二个目标值，则第一个个体不支配第二个个体
            and_condition = and_condition and first <= second
            # 若第一个目标值大于第二个目标值，则第一个个体支配第二个个体
            or_condition = or_condition or first < second
        return (and_condition and or_condition)
