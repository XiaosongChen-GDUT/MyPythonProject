import simpy
import Control
import MapView
import DataModel

def main():
    env = simpy.Environment()
    DG = DataModel.Model().read_map()#加载地图数据，返回有向图，节点位置，节点颜色
    # DG, pos, node_colors = DataModel.Model().read_map()#加载地图数据，返回有向图，节点位置，节点颜色
    control = Control.Control(env)#初始化控制类
    view = MapView.MapView(DG,env, control)#初始化视图类
    # view = MapView.MapView(DG, pos, node_colors,env, control)#初始化视图类
    control.set_view(view)#设置视图类
    view.mainloop()

if __name__ == '__main__':
    main()
