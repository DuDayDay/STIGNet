import json
import numpy as np
from sklearn.neighbors import KDTree
import os
import open3d as o3d
import re
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from traits.trait_types import self
import warnings
def visualize_point_cloud(all_points, sampled_points=None):
    """
    使用 Plotly 可视化点云数据，并突出显示采样点（如果存在）。

    :param all_points: 原始点云数据，包含坐标 (x, y, z) 和特征。
    :param sampled_points: 采样点的索引，显示在点云中的采样点。如果没有采样点，传递 None。
    """
    # 提取原始点云的坐标
    x, y, z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    features = all_points[:, 3] if all_points.shape[1] > 3 else np.zeros_like(x)
    # 创建散点图（所有点）
    scatter_all = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,  # 点的大小
            color=features,  # 原始点云使用灰色
            opacity=0.5 , # 设置透明度以便更好查看采样点
            colorbar = dict(title="Feature"),  # 显示颜色条
        ),
        name='All Points'
    )

    # 初始化采样点的散点图为空
    scatter_sampled = None

    # 如果采样点存在，绘制采样点
    if sampled_points is not None and len(sampled_points) > 0:
        # 提取采样点的坐标（根据索引）
        sampled_x = all_points[sampled_points, 0]
        sampled_y = all_points[sampled_points, 1]
        sampled_z = all_points[sampled_points, 2]
        # sampled_x = sampled_points[:, 0]
        # sampled_y = sampled_points[:, 1]
        # sampled_z = sampled_points[:, 2]
        # 创建散点图（采样点）
        scatter_sampled = go.Scatter3d(
            x=sampled_x,
            y=sampled_y,
            z=sampled_z,
            mode='markers',
            marker=dict(
                size=5,  # 采样点的大小
                color='red',  # 采样点使用红色
                opacity=1.0  # 设置完全不透明
            ),
            name='Sampled Points'
        )

    # 设置布局
    layout = go.Layout(
        title="Point Cloud Visualization with Sampled Points",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    # 创建并显示图表
    data = [scatter_all]

    if scatter_sampled is not None:
        data.append(scatter_sampled)

    fig = go.Figure(data=data, layout=layout)
    fig.show()
def extract_object_pointcloud_o3d(point_cloud, object_info):
    # 提取对象的中心和尺寸
    centroid = object_info["centroid"]
    dimensions = object_info["dimensions"]

    # 获取中心坐标和尺寸
    center_x, center_y, center_z = centroid["x"], centroid["y"], centroid["z"]
    length, width, height = dimensions["length"], dimensions["width"], dimensions["height"]

    # 计算边界框的范围
    x_min, x_max = center_x - length / 2, center_x + length / 2
    y_min, y_max = center_y - width / 2, center_y + width / 2
    z_min, z_max = center_z - height / 2, center_z + height / 2

    # 将点云转换为numpy数组
    points = np.asarray(point_cloud.points)

    # 根据边界框条件过滤点云
    mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    filtered_points = points[mask]

    # 将过滤后的点云转换回open3d点云对象
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_point_cloud

# def align_point_cloud_to_maximize_x_keepZ(point_cloud):
#     """
#     在保持 z 轴不变的情况下旋转点云，以最大化点云在 x 轴上的投影。
#
#     参数:
#     point_cloud (np.ndarray): 输入点云数据，形状为 (N, 3)，每行表示一个 (x, y, z) 坐标点。
#
#     返回:
#     np.ndarray: 处理后的点云数据。
#     """
#     # 将点云数据投影到 x-y 平面，忽略 z 轴
#     xy_points = point_cloud[:, :2]
#
#     # 使用 PCA 分析点云在 x-y 平面的主方向
#     pca = PCA(n_components=2)
#     pca.fit(xy_points)
#
#     # 获取主要方向，即第一个主成分
#     main_direction = pca.components_[0]
#
#     # 计算将 main_direction 与 x 轴对齐的旋转角度
#     x_axis = np.array([1, 0])
#     angle = np.arctan2(main_direction[1], main_direction[0])
#
#     # 创建绕 z 轴旋转的旋转矩阵
#     cos_angle = np.cos(-angle)
#     sin_angle = np.sin(-angle)
#     rotation_matrix = np.array([
#         [cos_angle, -sin_angle, 0],
#         [sin_angle, cos_angle, 0],
#         [0, 0, 1]
#     ])
#
#     # 应用旋转矩阵到点云
#     # aligned_point_cloud = np.dot(point_cloud, rotation_matrix.T)
#     return rotation_matrix
# def align_point_cloud_to_maximize_z(point_cloud):
#     """
#     将点云数据的主轴与 z 轴对齐，以使 z 轴的绝对值之和最大化，同时保持原始点的正负方向。
#
#     参数:
#     point_cloud (np.ndarray): 输入点云数据，形状为 (N, 3)，每一行表示一个 (x, y, z) 坐标点。
#
#     返回:
#     np.ndarray: 处理后的旋转矩阵，保证主轴与 z 轴对齐且不改变原始点的正负方向。
#     """
#     # 使用 PCA 分析点云的主方向
#     pca = PCA(n_components=3)
#     pca.fit(point_cloud)
#
#     # 获取 PCA 的主成分方向向量
#     principal_axes = pca.components_
#
#     # 找到与 z 轴最接近的主轴
#     z_axis = np.array([0, 0, 1])
#     axis_projections = [np.abs(np.dot(principal_axis, z_axis)) for principal_axis in principal_axes]
#     max_index = np.argmax(axis_projections)
#     target_axis = principal_axes[max_index]
#
#     # 确保 target_axis 的 z 方向为正
#     if np.dot(target_axis, z_axis) < 0:
#         target_axis = -target_axis  # 翻转方向以保持一致的正方向
#
#     # 计算旋转矩阵，将目标主轴与 z 轴对齐
#     rotation_matrix = align_vector_to_z(target_axis)
#     return rotation_matrix
# def align_vector_to_z(vector):
#     """
#     计算将指定向量旋转到 z 轴的旋转矩阵。
#
#     参数:
#     vector (np.ndarray): 要对齐的向量，形状为 (3,)
#
#     返回:
#     np.ndarray: 旋转矩阵，形状为 (3, 3)
#     """
#     v = vector / np.linalg.norm(vector)
#     z_axis = np.array([0, 0, 1])
#
#     # 计算旋转轴和角度
#     axis = np.cross(v, z_axis)
#     angle = np.arccos(np.dot(v, z_axis))
#
#     # 如果角度为零，说明已经对齐，无需旋转
#     if np.isclose(angle, 0):
#         return np.eye(3)
#
#     # 归一化旋转轴
#     axis = axis / np.linalg.norm(axis)
#
#     # 使用 Rodrigues 旋转公式生成旋转矩阵
#     K = np.array([
#         [0, -axis[2], axis[1]],
#         [axis[2], 0, -axis[0]],
#         [-axis[1], axis[0], 0]
#     ])
#     rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
#
#     return rotation_matrix
def align_point_cloud_to_maximize_x_keepZ(point_cloud):
    """
    在保持 z 轴不变的情况下旋转点云，以最大化点云在 x 轴上的投影。

    参数:
    point_cloud (np.ndarray): 输入点云数据，形状为 (N, 3)，每行表示一个 (x, y, z) 坐标点。

    返回:
    np.ndarray: 旋转矩阵，形状为 (3, 3)。
    """
    # 检查输入点云是否包含非法值
    if not np.isfinite(point_cloud).all():
        raise ValueError("点云数据中包含 NaN 或 Inf 值。")

    # 检查点云的方差是否为零
    if np.var(point_cloud[:, :2]) == 0:
        # print("点云在 x-y 平面上的方差为零，无法计算主方向。")
        return np.eye(3)  # 返回单位矩阵，表示不旋转

    # 将点云数据投影到 x-y 平面，忽略 z 轴
    xy_points = point_cloud[:, :2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # 使用 PCA 分析点云在 x-y 平面的主方向
        pca = PCA(n_components=2)
        pca.fit(xy_points)

    # 获取主要方向，即第一个主成分
    main_direction = pca.components_[0]

    # 计算将 main_direction 与 x 轴对齐的旋转角度
    angle = np.arctan2(main_direction[1], main_direction[0])

    # 创建绕 z 轴旋转的旋转矩阵
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])

    return rotation_matrix

def align_point_cloud_to_maximize_z(point_cloud):
    """
    将点云数据的主轴与 z 轴对齐，以使 z 轴的绝对值之和最大化，同时保持原始点的正负方向。

    参数:
    point_cloud (np.ndarray): 输入点云数据，形状为 (N, 3)，每行表示一个 (x, y, z) 坐标点。

    返回:
    np.ndarray: 旋转矩阵，形状为 (3, 3)。
    """
    # 检查输入点云是否包含非法值
    if not np.isfinite(point_cloud).all():
        raise ValueError("点云数据中包含 NaN 或 Inf 值。")

    # 检查点云的方差是否为零
    if np.var(point_cloud) == 0:
        # print("点云方差为零，无法计算主方向。")
        return np.eye(3)  # 返回单位矩阵，表示不旋转

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # 使用 PCA 分析点云的主方向
        pca = PCA(n_components=3)
        pca.fit(point_cloud)

    # 获取 PCA 的主成分方向向量
    principal_axes = pca.components_

    # 找到与 z 轴最接近的主轴
    z_axis = np.array([0, 0, 1])
    axis_projections = [np.abs(np.dot(principal_axis, z_axis)) for principal_axis in principal_axes]
    max_index = np.argmax(axis_projections)
    target_axis = principal_axes[max_index]

    # 确保 target_axis 的 z 方向为正
    if np.dot(target_axis, z_axis) < 0:
        target_axis = -target_axis  # 翻转方向以保持一致的正方向

    # 计算旋转矩阵，将目标主轴与 z 轴对齐
    rotation_matrix = align_vector_to_z(target_axis)
    return rotation_matrix

def align_vector_to_z(vector):
    """
    计算将指定向量旋转到 z 轴的旋转矩阵。

    参数:
    vector (np.ndarray): 要对齐的向量，形状为 (3,)

    返回:
    np.ndarray: 旋转矩阵，形状为 (3, 3)
    """
    # 检查输入向量是否包含非法值
    if not np.isfinite(vector).all():
        raise ValueError("输入向量包含 NaN 或 Inf 值。")

    # 计算向量的模并检查是否为零向量
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("输入向量的模为零，无法计算旋转矩阵。")

    # 单位化
    vector = vector / norm
    z_axis = np.array([0, 0, 1])

    # 计算旋转轴和角度
    axis = np.cross(vector, z_axis)
    angle = np.arccos(np.clip(np.dot(vector, z_axis), -1.0, 1.0))

    # 如果角度为零，说明已经对齐，无需旋转
    if np.isclose(angle, 0):
        return np.eye(3)

    # 归一化旋转轴
    axis = axis / np.linalg.norm(axis)

    # 使用 Rodrigues 旋转公式生成旋转矩阵
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return rotation_matrix

def compute_density(point_cloud_frames, radius=0.5):
    all_points = np.vstack(point_cloud_frames)
    tree = KDTree(all_points)
    density = tree.query_radius(all_points, r=radius, count_only=True)
    density_normalized = density / np.max(density)
    return all_points, density_normalized
def load_files(folder_path):
    if os.path.isdir(folder_path):
        # 获取文件夹中的所有图像文件名
        points_files = [f for f in os.listdir(folder_path)]
        # 按照文件名中的数字顺序排序
        points_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # 提取文件名中的数字进行排序
    else:
        points_files = []  # 如果不是有效的文件夹，清空列表
    return points_files
def load_json(json_file_path):

    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def label_points(points_file, label_file):
    filtered_point_cloud = []
    json_data = load_json(label_file)
    pcd_file_path = points_file
        # 使用open3d加载PCD文件
    point_cloud = o3d.io.read_point_cloud(pcd_file_path)

            # 提取第一个对象的点云
    if json_data["objects"] != []:
        object_info = json_data["objects"][0]
        filtered_point_cloud = extract_object_pointcloud_o3d(point_cloud, object_info)
        filtered_point_cloud = np.asarray(filtered_point_cloud.points)
    return filtered_point_cloud
def read_data(points, labels):
    first = 0
    first_2 = 0
    num = 0
    points_num = []
    seg_point_cloud = []
    points_files = load_files(points)
    labels_files = load_files(labels)
    files = zip(points_files, labels_files)
    for point, label in files:
        point_file = os.path.join(points, point)
        label_file = os.path.join(labels, label)
        points_seg = label_points(point_file, label_file)
        if points_seg == []:
            print("No points")
            print(point_file, label_file)
        centroid = points_seg.mean(axis=0)
        # 将点云平移，使质心位于原点
        points_seg = points_seg - centroid
        if points_seg is not None:
            if first == 0:
                rotation_matrix = align_point_cloud_to_maximize_z(points_seg)
                first = 1
            aligned_point_cloud = np.dot(points_seg, rotation_matrix.T)
            # 将质心平移到原点
            centroid = aligned_point_cloud.mean(axis=0)
            aligned_point_cloud -= centroid
            points_seg = aligned_point_cloud
            if first_2 == 0:
                rotation_matrix_2 = align_point_cloud_to_maximize_x_keepZ(points_seg)
                first_2 = 1
            points_seg = np.dot(points_seg, rotation_matrix_2.T)
            points_seg[:, 2] *= -1
            num = len(points_seg)
        points_num.append(num)
        seg_point_cloud.append(points_seg)
    return seg_point_cloud, points_num

def preprocess(folder_path):
    subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    len_points = []
    for subfolder in subfolders:
        path = os.path.join(folder_path, subfolder)
        points = os.path.join(path, 'pcd')
        labels = os.path.join(path, 'label')
        seg_point_cloud, _ = read_data(points, labels)
        all_points, density = compute_density(seg_point_cloud)
        len_points.append(len(all_points))
    return len_points

def farthest_point_sampling_with_density(points, densities, num_samples):
    """
    基于密度和最远距离的采样方法，同时记录每个采样点周围的点。

    :param points: 点云数据，形状为 (n, 3)。
    :param densities: 点云的密度，形状为 (n,)。
    :param num_samples: 需要采样的点数。

    :return: 采样点的索引列表以及每个采样点的邻域（每个采样点的半径为最小距离）。
    """
    # 创建一个权重数组，密度较高的点权重更大
    weights = densities / np.sum(densities)  # 归一化密度作为权重

    # 随机选择一个初始点
    sampled_points = [np.random.choice(len(points), p=weights)]

    # 计算每个点到已采样点的距离
    dist = np.ones(len(points)) * np.inf

    # 创建 BallTree 用于快速邻域搜索
    tree = BallTree(points)

    # 存储每个采样点的邻域点
    neighbors = {}
    all_neighbor_points = []
    # 初始采样点已经选择了，因此循环次数是 num_samples - 1
    for _ in range(num_samples):
        # 更新每个点到已采样点的最小距离
        for i in range(len(points)):
            dist[i] = min(dist[i], np.linalg.norm(points[i] - points[sampled_points[-1]]))

        # 选择最远的点，但需要按密度加权
        weighted_dist = dist * densities  # 权重乘以最远距离
        farthest_point_idx = np.argmax(weighted_dist)

        # 记录采样点
        sampled_points.append(farthest_point_idx)

        # 计算新的采样点与其他采样点的最小距离（即邻域的半径）
        distances_to_sampled = [np.linalg.norm(points[farthest_point_idx] - points[sampled_idx]) for sampled_idx in
                                sampled_points if sampled_idx != farthest_point_idx]
        min_distance = min(distances_to_sampled)

        # 使用 BallTree 进行邻域搜索
        # 找到所有在半径范围内的点
        indices = tree.query_radius([points[farthest_point_idx]], r=min_distance)

        # 记录邻域
        neighbors[farthest_point_idx] = indices[0]
        all_neighbor_points.extend(indices[0])
    sampled_points = sampled_points[:-1]
    # 返回采样点的索引和邻域
    return sampled_points, neighbors


def handle(folder_path):
    subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    len_points = []
    points_cloud = []
    for i, subfolder in enumerate(subfolders):
        path = os.path.join(folder_path, subfolder)
        points = os.path.join(path, 'pcd')
        labels = os.path.join(path, 'label')
        seg_point_cloud, _ = read_data(points, labels)
        if i == 1:
            seg_point_cloud = [point + np.array([0.5, 0, 0]) for point in seg_point_cloud]
        if i == 2:
            seg_point_cloud = [point + np.array([0, 0.5, 0]) for point in seg_point_cloud]
        points_cloud.extend(seg_point_cloud)
        # all_points, density = compute_density(seg_point_cloud)
        # len_points.append(len(all_points))
    all_points, density = compute_density(points_cloud)
    sampled_points, neighbors = farthest_point_sampling_with_density(all_points, density ,16)
    density = density[:, np.newaxis]
    points_cloud_frames = np.hstack((all_points, density))

    return points_cloud_frames, sampled_points, neighbors



if __name__ == '__main__':
    subfolders = 'points_video'
    points_cloud_frames, sampled_points, neighbors = handle(subfolders)
    visualize_point_cloud(points_cloud_frames, sampled_points)

    # # 计算平均值
    # mean_value = np.mean(len_points)
    #
    # # 统计高于和低于平均值的点的个数
    # above_mean_count = np.sum(len_points > mean_value)
    # below_mean_count = np.sum(len_points < mean_value)
    #
    # # 绘制折线图
    # plt.plot(len_points, label="Data")
    #
    # # 添加平均线
    # plt.axhline(y=mean_value, color='r', linestyle='--', label=f"Mean: {mean_value:.2f}")
    #
    # # 在图表上显示统计信息
    # plt.text(
    #     x=len(len_points) * 0.7,  # 文本显示位置的 x 坐标 (在右侧靠近数据尾部)
    #     y=mean_value * 1.1,  # 文本显示位置的 y 坐标 (稍微高于平均线)
    #     s=f"Above mean: {above_mean_count}\nBelow mean: {below_mean_count}",
    #     color='blue',
    #     fontsize=10,
    #     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')  # 添加背景框
    # )
    #
    # # 添加标题和标签
    # plt.title("Example Line Plot with Mean Value")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    #
    # # 添加图例
    # plt.legend()
    #
    # # 显示图表
    # plt.show()