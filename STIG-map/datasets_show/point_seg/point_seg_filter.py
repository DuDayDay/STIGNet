import open3d as o3d
import numpy as np
import json

def load_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

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

def label_point_cloud(json_file_path):
    filtered_point_cloud = []
    json_data = load_json(json_file_path)
    pcd_file_path = json_data["path"]
    # 使用open3d加载PCD文件
    point_cloud = o3d.io.read_point_cloud(pcd_file_path)
    # 提取第一个对象的点云
    if json_data["objects"] != []:
        object_info = json_data["objects"][0]
        filtered_point_cloud = extract_object_pointcloud_o3d(point_cloud, object_info)
        filtered_point_cloud = np.asarray(filtered_point_cloud.points)
    return filtered_point_cloud