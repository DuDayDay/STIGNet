import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

def farthest_point_sampling(points, num_samples):
    N, _ = points.shape
    centroids = np.zeros((num_samples,), dtype=int)
    distances = np.full((N,), np.inf)
    farthest = np.random.randint(0, N)

    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)

    return points[centroids], centroids


def plot_sphere(ax, center, radius=0.05, color='red', alpha=0.3):
    """
    在3D图中以指定的中心和半径绘制一个半透明球体。

    参数:
    ax: 3D轴对象。
    center: 球心坐标，格式为 (x, y, z)。
    radius: 球的半径。
    color: 球体颜色。
    alpha: 球体的透明度。
    """
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, rstride=1, cstride=1)