from sklearn.decomposition import PCA
import numpy as np
from sklearn.neighbors import KDTree


def align_point_cloud_to_maximize_z(point_cloud):
    """
    将点云数据的主轴与 z 轴对齐，以使 z 轴的绝对值之和最大化，同时保持原始点的正负方向。

    参数:
    point_cloud (np.ndarray): 输入点云数据，形状为 (N, 3)，每一行表示一个 (x, y, z) 坐标点。

    返回:
    np.ndarray: 处理后的旋转矩阵，保证主轴与 z 轴对齐且不改变原始点的正负方向。
    """
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
    v = vector / np.linalg.norm(vector)
    z_axis = np.array([0, 0, 1])

    # 计算旋转轴和角度
    axis = np.cross(v, z_axis)
    angle = np.arccos(np.dot(v, z_axis))

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


def align_point_cloud_to_maximize_x_keepZ(point_cloud):
    """
    在保持 z 轴不变的情况下旋转点云，以最大化点云在 x 轴上的投影。

    参数:
    point_cloud (np.ndarray): 输入点云数据，形状为 (N, 3)，每行表示一个 (x, y, z) 坐标点。

    返回:
    np.ndarray: 处理后的点云数据。
    """
    # 将点云数据投影到 x-y 平面，忽略 z 轴
    xy_points = point_cloud[:, :2]

    # 使用 PCA 分析点云在 x-y 平面的主方向
    pca = PCA(n_components=2)
    pca.fit(xy_points)

    # 获取主要方向，即第一个主成分
    main_direction = pca.components_[0]

    # 计算将 main_direction 与 x 轴对齐的旋转角度
    x_axis = np.array([1, 0])
    angle = np.arctan2(main_direction[1], main_direction[0])

    # 创建绕 z 轴旋转的旋转矩阵
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])

    # 应用旋转矩阵到点云
    # aligned_point_cloud = np.dot(point_cloud, rotation_matrix.T)
    return rotation_matrix