import numpy as np
import matplotlib.pyplot as plt


def pixelate_point_cloud_fixed_ratio(point_cloud, pixel_size):
    """
    在保持原始空间比例的情况下对点云数据进行像素化处理。

    参数:
    - point_cloud: numpy 数组，形状为 (N, 3)，其中每一行代表一个点 (x, y, z)。
    - pixel_size: 每个像素的实际空间尺寸，定义为 (pixel_width, pixel_height)。

    返回:
    - pixelated_image: 像素化后的二维图像
    """
    # 获取点云的最大最小坐标范围
    min_x, min_y = np.min(point_cloud[:, :2], axis=0)
    max_x, max_y = np.max(point_cloud[:, :2], axis=0)

    # 计算网格尺寸，使得保持原始空间比例
    grid_width = int(np.ceil((max_x - min_x) / pixel_size[0]))
    grid_height = int(np.ceil((max_y - min_y) / pixel_size[1]))

    # 创建空的网格图像
    pixelated_image = np.zeros((grid_height, grid_width))

    # 计算每个点在网格中的坐标
    grid_x = ((point_cloud[:, 0] - min_x) / pixel_size[0]).astype(int)
    grid_y = ((point_cloud[:, 1] - min_y) / pixel_size[1]).astype(int)

    # 对应的像素位置增加计数
    for x, y in zip(grid_x, grid_y):
        if 0 <= x < grid_width and 0 <= y < grid_height:
            pixelated_image[y, x] += 1

    return pixelated_image