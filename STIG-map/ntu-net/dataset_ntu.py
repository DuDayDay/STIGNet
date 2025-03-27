import os
import time
from scipy.spatial import cKDTree
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import open3d as o3d
import matplotlib.cm as cm

def vision(all_points):
    points = all_points[:, :3]
    pcd = o3d.geometry.PointCloud()
    colormap = cm.get_cmap("jet")
    colors = colormap(all_points[:, 3])[:, :3]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def extract_number_from_filename(filename):
    """提取文件名中 Axxx 后的数字和相机信息"""
    match = re.search(r'A(\d+)', filename)
    camera = re.search(r'C(\d+)', filename)
    if match and camera:
        return int(match.group(1)), int(camera.group(1))  # 返回标签和相机编号
    else:
        raise ValueError("Filename does not contain a valid Axxx or Cxxx format")

# 数据集类
class NtuDataset(Dataset):
    def __init__(self, root, is_test=True):
        """
        初始化数据集

        Args:
            root (str): 数据集根目录
            num_points (int): 每个点云的点数
            is_test (bool): 如果是测试集，设置为True，默认加载训练集
        """
        self.is_test = is_test  # 是否加载测试集
        self.labels = []
        self.root = root
        rootdir = os.listdir(root)

        # 提取文件名中的标签和相机编号
        filenames_prefix = [extract_number_from_filename(filename) for filename in rootdir]

        # 根据相机编号划分数据集
        if is_test:
            self.files = [f for f, (label, camera) in zip(rootdir, filenames_prefix) if camera == 1]
            self.labels = [label for label, camera in filenames_prefix if camera == 1]
        else:
            self.files = [f for f, (label, camera) in zip(rootdir, filenames_prefix) if camera != 1]
            self.labels = [label for label, camera in filenames_prefix if camera != 1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx] - 1
        filename = self.files[idx]
        file = os.path.join(self.root, filename)
        all_points = np.load(os.path.join(file,'all_points_512_24.npy'))
        points_frames = np.load(os.path.join(file,'points_frames_512_24.npy'))
        # vision(all_points)
        #
        # for points in points_frames:
        #     vision(points)

        return all_points, points_frames,label

if __name__ == '__main__':
    start = time.time()
    data = NtuDataset(root='data/ntu/save')
    end = time.time()
    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    for all_points,  points_frames, points_frames_dist,label in dataloader:
        print(all_points)

        print(points_frames)
        print(points_frames_dist)
        print(label)