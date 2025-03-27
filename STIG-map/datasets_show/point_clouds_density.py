
import numpy as np
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


# 计算密度
def compute_density(point_cloud_frames, radius=0.5):
    all_points = np.vstack(point_cloud_frames)
    tree = KDTree(all_points)
    density = tree.query_radius(all_points, r=radius, count_only=True)
    return all_points, density


def density_based_sampling(all_points, density, num_samples):
    """
    根据密度进行点云采样，低密度区域更多保留，密度高的区域减少保留。

    参数:
        all_points (np.ndarray): 所有点云数据 (N, 3)。
        density (np.ndarray): 每个点的密度值。
        num_samples(float): 表示采样后点数。

    返回:
        sampled_points (np.ndarray): 采样后的点云数据。
    """
    # 为避免密度为0的问题，添加一个小值 epsilon
    epsilon = 1e-6
    sampling_prob = 1 / (density + epsilon)
    sampling_prob /= sampling_prob.sum()  # 归一化为概率分布

    # 根据采样概率选择点
    sampled_indices = np.random.choice(len(all_points), size=num_samples, replace=False, p=sampling_prob)
    sampled_points = all_points[sampled_indices]

    return sampled_points
# 根据密度获取颜色映射

def get_color_scale(density, invert_colors=False):
    density_normalized = (density - np.min(density)) / (np.max(density) - np.min(density))
    if invert_colors:
        density_normalized = 1 - density_normalized
    colors = plt.get_cmap('jet')(density_normalized)
    return [f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})' for r, g, b, a in colors]


# 聚类函数，按颜色聚类
def cluster_point_cloud_by_color(points, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)  # 添加 n_init 参数
    colors = kmeans.fit_predict(points)
    return colors


# 根据聚类结果获取颜色
def get_cluster_colors_by_labels(labels, n_clusters):
    cmap = plt.get_cmap("tab20", n_clusters)
    colors = [cmap(i / n_clusters) for i in range(n_clusters)]
    color_map = [f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})' for r, g, b, a in colors]
    return [color_map[label] for label in labels]


# 创建可选择聚类数的可视化
def create_plot_with_cluster_selection(points, density):
    # 原始和反转颜色
    color_scale_original = get_color_scale(density, invert_colors=False)
    color_scale_inverted = get_color_scale(density, invert_colors=True)
    # print(color_scale_original)

    # 默认聚类数
    default_clusters = 3
    labels = cluster_point_cloud_by_color(points, n_clusters=default_clusters)
    color_scale_clustered = get_cluster_colors_by_labels(labels, n_clusters=default_clusters)

    # 创建3D点云图
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=color_scale_original, opacity=0.8)
    )

    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title="3D Point Cloud with Color-Based Clustering",
        updatemenus=[
            dict(
                buttons=[
                    dict(label="Original Colors", method="update", args=[{"marker.color": [color_scale_original]}]),
                    dict(label="Inverted Colors", method="update", args=[{"marker.color": [color_scale_inverted]}]),
                ],
                direction="down",
                showactive=True,
            ),
            dict(
                buttons=[
                    dict(
                        label=f"{i} Clusters",
                        method="update",
                        args=[{"marker.color": [
                            get_cluster_colors_by_labels(cluster_point_cloud_by_color(points, n_clusters=i),
                                                         n_clusters=i)]}]
                    ) for i in range(2, 11)  # 聚类数从2到10
                ],
                direction="down",
                showactive=True,
                x=0.15,
                y=1.15,
                xanchor="left",
                yanchor="top"
            ),
        ]
    )

    fig = go.Figure(data=[trace], layout=layout)
    # fig.show()
    fig.write_html("plot.html")


