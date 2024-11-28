import simpy

from Program.DataModel import Model
from Program.MapView import MapView
from Program.DataModel import Vehicle
class Control(object):
    def __init__(self,Env):
        # , model, view
        # self.model = None
        self.view = None
        self.env = Env
        self.Time = 0
        self.AGVs = {}   # AGV字典
        self.AGV_num = 0 # AGV数量

    def set_view(self, view):
        self.view = view

    #添加AGV
    def Add_AGV(self, id, node):
        if not node:  # 判断 node 是否为空
            print("添加AGV失败！node 参数不能为空")
            return False
        if(id not in self.AGVs.keys()):#判断是否已经创建过该AGV
            vehicle = Vehicle(self.view,self.env, id, node)#创建AGV
            self.AGVs.update({id:vehicle})
            self.AGV_num += 1
        else:
            print("添加AGV失败！该AGV已经存在")
            return False
        return True

    #删除AGV
    def Del_AGV(self, id):
        if(id in self.AGVs.keys()):#判断是否存在该AGV
            self.AGVs.pop(id)
            self.AGV_num -= 1

    def run(self):
        try:
            #绘制地图
            # self.view.draw_graph(DG,pos,node_colors)
            path = self.model.findPath(751,326)
            self.view.draw_paths(path)
        except Exception as e:
            print(f"发生未知错误: {e}")

def main():
    env = simpy.Environment()
    model = Model()
    #读取地图数据
    DG,pos,node_colors = model.read_map()
    control = Control(env)#初始化控制类
    view = MapView(DG,pos,node_colors,env, control)

    view.mainloop()

if __name__ == '__main__':
    main()
    # input("Press any key to exit...")