import numpy as np
import plotly.graph_objs as go
from scipy.spatial import cKDTree


def compute_multiframe_activity_with_displacement(point_clouds, voxel_size=0.1, displacement_threshold=0.1):
    """
    计算多帧点云的活动频率，考虑位移
    :param point_clouds: 点云帧列表，每帧是一个 numpy 数组 (N, 3)
    :param voxel_size: 体素大小
    :param displacement_threshold: 位移阈值，用于判断是否为“活动”
    :return: 每个体素的活动频率映射，形式为 {(vx, vy, vz): 频率}
    """
    activity_map = {}

    # 遍历每一帧
    for frame_idx in range(1, len(point_clouds)):
        current_frame = point_clouds[frame_idx]
        previous_frame = point_clouds[frame_idx - 1]

        # 使用当前帧重新构建 KDTree
        kdtree = cKDTree(previous_frame)  # 构建前一帧的 KDTree

        # 对每一帧点云计算位移
        dist, idx = kdtree.query(current_frame, k=1)  # 查询当前帧点云的最近邻

        # 计算每个点的位移量
        for i, point in enumerate(current_frame):
            if idx[i] < len(previous_frame):  # 确保索引在前一帧范围内
                displacement = np.linalg.norm(point - previous_frame[idx[i]])  # 计算位移
                if displacement > displacement_threshold:
                    # 对于位移大于阈值的点，进行活动频率计数

                    # 处理 NaN 或 inf 值，避免无效转换
                    if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                        continue  # 跳过无效点

                    voxel_key = tuple(np.floor(point / voxel_size).astype(int))
                    activity_map[voxel_key] = activity_map.get(voxel_key, 0) + 1

    return activity_map

def map_activity_to_plotly_colors(activity_map, point_cloud, voxel_size=0.1):
    """
    将活动频率映射到 Plotly 可视化颜色
    :param activity_map: 体素的活动频率映射
    :param point_cloud: 合并后的所有点云数据 (numpy 数组)
    :param voxel_size: 体素大小
    :return: 返回颜色映射的点云数据，准备绘制
    """
    colors = np.zeros(len(point_cloud))
    max_frequency = max(activity_map.values()) if activity_map else 1

    # 根据体素的频率值来决定颜色
    for i, point in enumerate(point_cloud):
        # 处理 NaN 或 inf 值，避免无效转换
        if np.any(np.isnan(point)) or np.any(np.isinf(point)):
            colors[i] = 0  # 对于无效点，将其颜色设为黑色（或其他颜色）
            continue

        voxel_key = tuple(np.floor(point / voxel_size).astype(int))
        frequency = activity_map.get(voxel_key, 0)
        colors[i] = frequency / max_frequency  # 归一化频率

    # 将归一化的频率值映射到颜色值
    color_values = colors * 255  # 转换为 0-255 范围的颜色值

    return color_values

# # 示例点云数据：假设有三帧
# point_clouds = [np.random.rand(1000, 3) for _ in range(3)]  # 使用随机数据作为示例
#
# # 计算多帧的活动频率
# voxel_size = 0.1
# activity_map = compute_multiframe_activity(point_clouds, voxel_size=voxel_size)
#
# # 映射到颜色并显示（使用最后一帧进行可视化）
# colored_pcd = map_activity_to_colors_multiframe(activity_map, point_clouds[-1], voxel_size=voxel_size)
# o3d.visualization.draw_geometries([colored_pcd])
