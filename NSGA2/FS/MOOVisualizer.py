import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import plotly.express as px

class MOOVisualizer:
    """
        初始化可视化器，支持多实验对比。

        :param objectives: Pareto 前沿的目标值数组
        :param obj_names: 目标函数名称列表
        :param experiment_results: 实验结果字典，格式为 {exp_name: {'hv_history': [], 'objective_history': {'F1': [], 'F2': [], 'F3': []}}}
        """

    def __init__(self, objectives, obj_names=None, experiment_results=None):
        self.raw_data = np.array(objectives)
        self.n_obj = self.raw_data.shape[1]
        self.obj_names = obj_names if obj_names else [f"F{i+1}" for i in range(self.n_obj)]
        self.scaler = MinMaxScaler()
        self.norm_data = self.scaler.fit_transform(self.raw_data)
        self.df = pd.DataFrame(self.raw_data, columns=self.obj_names)
        self.df_norm = pd.DataFrame(self.norm_data, columns=self.obj_names)

    """
    绘制不同实验的 HV 曲线对比。
    """
    def plot_hv_comparison(self):
        plt.figure(figsize=(10, 6))
        for exp_name, result in self.experiment_results.items():
            hv_history = result['hv_history']
            generations = range(1, len(hv_history) + 1)
            plt.plot(generations, hv_history, label=exp_name)
        plt.xlabel('Generation')
        plt.ylabel('Hypervolume (HV)')
        plt.title('HV Comparison Across Experiments')
        plt.legend()
        plt.grid(True)
        plt.savefig('hv_comparison.png', dpi=300)
        plt.show()

    def _prepare_plot(self, figsize=(10,6)):
        """准备绘图基础设置"""
        plt.figure(figsize=figsize)
        plt.grid(True)

    def plot_single_objective(self, ax=None, **kwargs):
        """
        单目标可视化（箱线图+分布图）

        参数：
        - ax: matplotlib轴对象
        - kwargs: 传递给seaborn的绘图参数
        """
        if self.n_obj != 1:
            raise ValueError("单目标可视化仅支持1个目标")

        ax = ax or plt.gca()
        sns.boxplot(data=self.df, width=0.3, ax=ax, **kwargs)
        sns.stripplot(data=self.df, jitter=True, color='black', ax=ax, **kwargs)
        ax.set_title("Single Objective Distribution", fontsize=12)
        return ax

    def plot_parallel_coordinates(self, color_by=None, alpha=0.5, **kwargs):
        """
        绘制平行坐标图

        参数：
        - color_by: 着色依据的列名
        - alpha: 线条透明度
        - kwargs: 传递给parallel_coordinates的参数
        """
        df = self.df_norm.copy()
        df['Solution'] = range(len(df))

        self._prepare_plot()
        parallel_coordinates(df, 'Solution', color=color_by,
                             colormap='viridis', alpha=alpha, **kwargs)
        plt.title(f"Parallel Coordinates ({self.n_obj} Objectives)", fontsize=12)
        return plt.gcf()

    def plot_3d_scatter(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(self.df[self.obj_names[0]],
                        self.df[self.obj_names[1]],
                        self.df[self.obj_names[2]],
                        c=range(len(self.df)),  # 添加颜色映射
                        cmap='viridis')
        ax.set_xlabel(self.obj_names[0])
        ax.set_ylabel(self.obj_names[1])
        ax.set_zlabel(self.obj_names[2])
        ax.set_title('3D Scatter Plot of Pareto Front')
        plt.colorbar(sc, label='Solution Index')

    def plot_scatter_matrix(self, diag='hist', **kwargs):
        """
        绘制散点矩阵图

        参数：
        - diag: 对角线图类型 ('hist'/'kde')
        - kwargs: 传递给PairGrid的参数
        """
        g = sns.PairGrid(self.df, diag_sharey=False, **kwargs)
        g.map_upper(sns.scatterplot, s=15, alpha=0.5)
        g.map_lower(sns.kdeplot, cmap='Blues_d')
        g.map_diag(sns.histplot if diag=='hist' else sns.kdeplot)
        plt.suptitle("Scatter Plot Matrix", y=1.02)
        return g

    def plot_radial(self, n_samples=None, **kwargs):
        """
        径向可视化图

        参数：
        - n_samples: 采样数量（避免过度拥挤）
        """
        plot_data = self.df_norm.sample(n_samples) if n_samples else self.df_norm

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, polar=True)
        categories = self.obj_names
        N = len(categories)

        angles = [n/float(N)*2*np.pi for n in range(N)]
        angles += angles[:1]

        for idx, row in plot_data.iterrows():
            values = row.values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title("Radial Visualization", fontsize=12)
        return fig

    def interactive_parallel(self, color_by=None, **kwargs):
        """
        生成交互式平行坐标图（Plotly）

        参数：
        - color_by: 着色依据的列名
        - kwargs: 传递给plotly的参数
        """
        return px.parallel_coordinates(
            self.df,
            color=color_by,
            labels={col:col for col in self.df.columns},
            color_continuous_scale=px.colors.diverging.Tealrose,
            **kwargs
        )

    def plot_tsne_projection(self, perplexity=30, **kwargs):
        """
        t-SNE降维投影可视化

        参数：
        - perplexity: t-SNE困惑度参数
        """
        tsne = TSNE(n_components=2, perplexity=perplexity)
        proj = tsne.fit_transform(self.df)

        fig, ax = plt.subplots(figsize=(10,6))
        sc = ax.scatter(proj[:,0], proj[:,1], c=np.arange(len(self.df)),
                        cmap='viridis', **kwargs)
        plt.colorbar(sc, label='Solution Index')
        ax.set_title("t-SNE Projection of Pareto Front", fontsize=12)
        return fig

    def auto_plot(self, save_path=None, **kwargs):
        """
        自动选择合适可视化方法

        参数：
        - save_path: 图片保存路径
        """
        if self.n_obj == 1:
            fig = self.plot_single_objective()
        elif self.n_obj == 2:
            fig = self.plot_2d_scatter(**kwargs)
        elif self.n_obj == 3:
            fig = self.plot_3d_scatter(**kwargs)
        else:
            fig = self.plot_parallel_coordinates(**kwargs)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return fig

    def plot_2d_scatter(self, color_by=None, **kwargs):
        """
        二维散点图（支持2-3目标，第三目标用颜色/大小表示）
        """
        if self.n_obj < 2:
            raise ValueError("需要至少2个目标")

        fig, ax = plt.subplots(figsize=(8,6))
        if self.n_obj >= 3 and color_by:
            sc = ax.scatter(self.df[self.obj_names[0]],
                            self.df[self.obj_names[1]],
                            c=self.df[color_by],
                            cmap='viridis', **kwargs)
            plt.colorbar(sc, label=color_by)
        else:
            ax.scatter(self.df[self.obj_names[0]],
                       self.df[self.obj_names[1]], **kwargs)

        ax.set_xlabel(self.obj_names[0])
        ax.set_ylabel(self.obj_names[1])
        plt.title("2D Scatter Plot" +
                  (" with Color Mapping" if color_by else ""), fontsize=12)
        return fig

if __name__ == '__main__':
    # 生成测试数据
    np.random.seed(42)
    n_obj = 4
    data = np.random.rand(100, n_obj) * [10, 5, 8, 2] + [1, 3, 0, 5]
    obj_names = ['Cost', 'Time', 'Risk', 'Quality']

    # 初始化可视化器
    vis = MOOVisualizer(data, obj_names=obj_names)

    # 自动选择可视化方法
    vis.auto_plot(save_path='auto_plot.png')

    # 交互式平行坐标图（需要plotly）
    fig = vis.interactive_parallel(color_by='Quality')
    # fig.write_html("interactive_parallel.html")

    # 散点矩阵图
    g = vis.plot_scatter_matrix()
    # g.figure.savefig('scatter_matrix.png', dpi=300)

    # 3D散点图（第四维用颜色表示）
    fig = vis.plot_3d_scatter(color_by='Quality', s=50)
    # fig.savefig('3d_plot.png', dpi=300)