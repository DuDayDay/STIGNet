
import open3d as o3d
import pandas as pd
import numpy as np


class Pcd:
    def __init__(self, path):
        self.df = pd.read_csv(path, encoding='utf-8', low_memory=False)
        self.points = self.df.iloc[0:, 8:12].values

    def cut(self, x1, x2):
        points_part0 = self.points[x1:int(x2)]
        points_part = list(points_part0)
        return points_part0

    def produce(self, cutting):
        pcd = o3d.geometry.PointCloud()
        pcd_points = o3d.utility.Vector3dVector(cutting)
        return pcd_points

# function: 读入excel文件，返回帧
# param: 文件地址
# param: 分辨率
# param: 重叠率
# retval: 返回一个包含所有PCD的列表
def file_input(path, resolution, overlap):
    file0 = Pcd(path)
    i = 1
    pcd = []
    while(i + resolution <= len(file0.points)):
        cutting = file0.cut(i, i + resolution)
        pcd.append(cutting)
        i += overlap
    return pcd


def frame_divide(source_path, resolution, frame_num):
    frame_num = frame_num - 1
    file0 = Pcd(source_path)
    total_points = len(file0.points)
    step_size = (total_points - resolution) / frame_num if frame_num > 0 else total_points

    # 生成所有切割的起始索引
    indices = [int(i) for i in np.arange(0, total_points - resolution + 1, step_size)]

    pcde = []
    for start_index in indices:
        cutting = file0.cut(start_index, start_index + resolution)
        pcde.append(cutting)

    return pcde


def frame_save_2(source_path, folder_path, resolution, frame_num):
    # 读取 CSV 文件，避免 DtypeWarning
    df = pd.read_csv(source_path, encoding='utf-8', low_memory=False, dtype={'column_name': float})
    points_data = df.iloc[:, 8: 11].values
    reflectivity_data = df.iloc[:, 11: 14].values

    # 将反射率数据的第二和第三列置为 0
    reflectivity_data[:, 1] = 0
    reflectivity_data[:, 2] = 0

    total_points = len(points_data)
    step_size = (total_points - resolution) / (frame_num - 1) if frame_num > 1 else total_points
    indices = [int(i) for i in np.arange(0, total_points - resolution + 1, step_size)]

    # 生成点云并保存
    for j, start_index in enumerate(indices):
        end_index = start_index + resolution
        start_index = int(start_index)  # 确保是整数
        end_index = int(end_index)      # 确保是整数

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_data[start_index:end_index])
        pcd.colors = o3d.utility.Vector3dVector(reflectivity_data[start_index:end_index])
        o3d.io.write_point_cloud(f"{folder_path}/pcd_file{j + 1}.pcd", pcd)

