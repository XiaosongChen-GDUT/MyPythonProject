
Acc_lift = 0.15;                              #提升机加速度
Dec_lift = 0.15;                              #提升机减速度
Max_speed_lift = 1.4;                         #提升机最大速度
class Elevator:
    def __init__(self, ID ,initial_floor):
        self.ID = ID  # 电梯ID
        self.current_floor = initial_floor  # 电梯当前楼层

        self.current_status = 0  # 电梯当前状态 0-空闲 1-忙碌 2-故障
        self.current_speed = 0  # 电梯当前速度
        self.wait_time = 1      # 电梯等待时间

        self.load_status = 0  # 电梯载重状态 0-空 1-满
        self.load_direction = 0  # 电梯载重方向 0-上 1-下
        self.Task_list = []  # 电梯任务列表，{start，end}，start-起始楼层，end-终止楼层

    '''评估提升机总耗时'''
    def evaluate_lift(self, lift, height_diff, car_ready_time):

        # 计算当前垂直任务时间
        vertical_time = self.calculate_vertical_time(height_diff)

        # 计算队列中最后一个任务的结束时间
        queue_end_time = lift.queue[-1].end_time if lift.queue else 0

        # 协同等待时间 = max(提升机前序任务完成时间, 四向车就绪时间)
        sync_delay = max(queue_end_time, car_ready_time)

        # 总耗时 = 队列等待 + 同步等待 + 垂直任务时间
        total_time = sync_delay + vertical_time

        return total_time

    '''计算垂直段运行时间：'''
    def calculate_vertical_time(dz):
        t_acc = Max_speed_lift / Acc_lift      # 加速度阶段时间
        t_dec = Max_speed_lift / Dec_lift      # 减速阶段时间
        s_acc = 0.5 * Acc_lift * t_acc**2      # 加速度阶段位移
        s_dec = 0.5 * Dec_lift * t_dec**2            # 减速阶段位移
        total_acc_dec_distance = s_acc + s_dec  # 临界加速减速距离
        if dz <= total_acc_dec_distance:
            # math.sqrt(2*height_diff*(1/Acc + 1/Dcc))
            return (2*dz*(1/Acc_lift + 1/Dec_lift))**0.5
        else:
            t_cruise = (dz - total_acc_dec_distance) / Max_speed_lift
            return  t_acc + t_dec + t_cruise