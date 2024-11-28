import struct
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.lines as lines
from math import sin,cos,tan,pi
import numpy as np

# 创建初始图形
fig, ax = plt.subplots(figsize=(6, 6))
def rotate_translate(points, angle, translation):
    # 构建旋转矩阵
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # 平移向量
    translation_vector = np.array(translation)

    # 转换顶点列表为正确的形状
    vertices = np.array(points).reshape(-1, 2)

    # 进行旋转和平移
    transformed_vertices = np.dot(rotation_matrix, vertices.T).T + translation_vector

    # 将转换后的顶点列表转换为元组形式
    transformed_points = [tuple(vertex) for vertex in transformed_vertices]

    return transformed_points
class AkmanCar:
    def __init__(self,ax,init_pos) -> None:
        self.ax = ax
        self.init_pos = init_pos  #初始化位置
        self.pos = [0,init_pos]   #车辆位置
        self.Len = 3              #车长
        self.Width = 2            #车宽
        self.angle_wheel = 0      #前轮转角
        self.yaw = 0              #车身航向角
        self.yaw_w = 0            #车身航向角速度
        self.vy = 0               #车身前后方向速度

        self.body_points = [(-0.5*self.Width  , 0            ),(-0.5*self.Width  , self.Len     ),
                            ( 0.5*self.Width  , self.Len     ),( 0.5*self.Width  , 0            )]        #车身多边形顶点
        self.body = patches.Polygon(self.body_points, closed=True,lw = 1, edgecolor='k', facecolor='none')#车身多边形对象

        self.wheel_shape = [(-0.15,-0.3),(-0.15,0.3),(0.15,0.3),(0.15,-0.3)]                                 #轮子多边形顶点

        self.wheel_FL = patches.Polygon(self.wheel_shape, closed=True,lw = 1, edgecolor='k', facecolor='k') #轮子对象
        self.wheel_FR = patches.Polygon(self.wheel_shape, closed=True,lw = 1, edgecolor='k', facecolor='k')
        self.wheel_BL = patches.Polygon(self.wheel_shape, closed=True,lw = 1, edgecolor='k', facecolor='k')
        self.wheel_BR = patches.Polygon(self.wheel_shape, closed=True,lw = 1, edgecolor='k', facecolor='k')

        self.x = []  #轨迹点
        self.y = []
        self.trajectory, = self.ax.plot([], [], 'r-')#轨迹线对象，要加逗号

    def update_graph(self,frame): #图形更新

        body_points = rotate_translate(self.body_points, self.yaw, self.pos)  #旋转平移
        self.body.set_xy(body_points)   #设定图形坐标

        wheel_points_FL = rotate_translate(self.wheel_shape, self.angle_wheel, [-self.Width/2, self.Len])  #旋转平移至相对车身对应位置
        wheel_points_FL = rotate_translate(wheel_points_FL, self.yaw, self.pos)   #旋转平移至车身对应位置

        wheel_points_FR = rotate_translate(self.wheel_shape, self.angle_wheel, [self.Width/2, self.Len])
        wheel_points_FR = rotate_translate(wheel_points_FR, self.yaw, self.pos)

        wheel_points_BL = rotate_translate(self.wheel_shape, 0, [-self.Width/2, 0])
        wheel_points_BL = rotate_translate(wheel_points_BL, self.yaw, self.pos)

        wheel_points_BR = rotate_translate(self.wheel_shape, 0, [self.Width/2, 0])
        wheel_points_BR = rotate_translate(wheel_points_BR, self.yaw, self.pos)

        self.wheel_FL.set_xy(wheel_points_FL) #设定坐标
        self.wheel_FR.set_xy(wheel_points_FR)
        self.wheel_BL.set_xy(wheel_points_BL)
        self.wheel_BR.set_xy(wheel_points_BR)

        plt.draw()
        # 清空坐标轴
        ax.clear()

        # 添加图形到坐标轴

        self.ax.add_artist(self.body)
        self.ax.add_artist(self.wheel_FL)
        self.ax.add_artist(self.wheel_FR)
        self.ax.add_artist(self.wheel_BL)
        self.ax.add_artist(self.wheel_BR)
        self.ax.add_artist(self.trajectory)

        self.x.append(self.pos[0][0]) #添加轨迹点
        self.y.append(self.pos[0][1])
        self.trajectory.set_data(self.x,self.y)


    def update_state(self,step):  #更新状态数据
        self.yaw_w = self.vy*tan(self.angle_wheel)/self.Len #计算车身航向角速度

        self.yaw = self.yaw + self.yaw_w*step               #计算车身航向角
        v = rotate_translate((0,self.vy), self.yaw, [0,0])
        self.pos = self.pos +  np.array(v)*step             #计算车身位置

        # print("pos:",self.pos)
t1 = AkmanCar(ax,0)
t = 0.1
def updateall(frame):
    global t
    ax.clear()
    t = t+0.04
    t1.angle_wheel = 0.6*cos(t)
    t1.vy = 5
    t1.update_state(0.1)
    t1.update_graph(frame)

    # 设置坐标轴范围
    t1.ax.set_xlim(t1.pos[0][0] - 10, t1.pos[0][0] + 10)
    t1.ax.set_ylim(t1.pos[0][1] - 10, t1.pos[0][1] + 10)


def main():

    # 创建动画
    animation = FuncAnimation(fig, updateall, frames=range(100), interval=1)
    # 显示动画
    plt.show()

if __name__ == '__main__':
    main()