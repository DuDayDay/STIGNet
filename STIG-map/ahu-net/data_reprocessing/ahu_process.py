"""
this processing is used for datasets_ahu_noframes
"""
import os
from utils.data_preprocess import read_data
import os
import numpy as np
import random
import xml.etree.ElementTree as ET
from sklearn.neighbors import KDTree
def compute_density(point_cloud_frames, radius=0.5):
    all_points = np.vstack(point_cloud_frames)
    tree = KDTree(all_points)
    density = tree.query_radius(all_points, r=radius, count_only=True)
    density_normalized = density / np.max(density)
    return all_points, density_normalized
def parse_xml(xml_file):
    """
    解析 XML 文件并提取所需信息。
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    class_type = None
    frame_num = None

    # 提取 class 的 type 和 frame 的 num
    for elem in root:
        if elem.tag == "class" and "type" in elem.attrib:
            class_type = elem.attrib["type"]
        elif elem.tag == "frame" and "num" in elem.attrib:
            frame_num = elem.attrib["num"]

    return class_type, frame_num
def upsample(all_points, target_count, radius=0.08):
    """
    使用插值方法上采样点云到固定数量，同时处理特征。
    """
    # 检查输入点云数据的合法性
    if np.isnan(all_points).any() or np.isinf(all_points).any():
        raise ValueError("Input data contains NaN or Inf values.")

    coords = all_points[:, :3]  # 提取坐标部分
    density = all_points[:, 3]
    features = all_points[:, 3:]  # 提取特征部分

    # 确保 density 和 weights 的有效性
    density = np.array(density, dtype=np.float64)
    if density.sum() == 0 or np.isnan(density).any() or np.isinf(density).any():
        raise ValueError("Density values are invalid.")
    weights = density / density.sum()

    # 初始化 KDTree
    tree = KDTree(coords)

    # 计算需要新增的点数
    current_count = len(all_points)
    new_point_count = target_count - current_count
    if new_point_count <= 0:
        return all_points  # 如果目标点数小于当前点数，直接返回原始点云

    # 插值生成新点
    new_points = []
    while len(new_points) < new_point_count:
        idx = np.random.choice(len(coords), p=weights)
        base_point = coords[idx]
        base_feature = features[idx]

        # 查询邻居点并随机选一个邻居
        neighbor_idx = tree.query_radius([base_point], r=radius, return_distance=False)
        if len(neighbor_idx[0]) > 1:
            neighbor_idx = np.random.choice(neighbor_idx[0][1:])  # 随机选一个邻居点
            neighbor_point = coords[neighbor_idx]
            neighbor_feature = features[neighbor_idx]
        else:
            continue

        # 在线性插值生成新点
        t = np.random.uniform(0, 1)
        new_coord = (1 - t) * base_point + t * neighbor_point
        new_feature = neighbor_feature

        # 合并坐标和特征
        new_points.append(np.hstack((new_coord, new_feature)))

    # 检查插值点是否包含非法值
    new_points = np.array(new_points[:new_point_count])
    if np.isnan(new_points).any() or np.isinf(new_points).any():
        raise ValueError("Newly interpolated points contain NaN or Inf values.")

    # 合并原始点和新插值点
    upsampled_points = np.vstack([all_points, new_points])

    # 调整点数至目标数量
    if len(upsampled_points) < target_count:
        remaining_points = target_count - len(upsampled_points)
        extra_indices = np.random.choice(len(upsampled_points), remaining_points, replace=True)
        extra_points = upsampled_points[extra_indices]
        upsampled_points = np.vstack([upsampled_points, extra_points])
    elif len(upsampled_points) > target_count:
        excess_count = len(upsampled_points) - target_count
        delete_indices = np.random.choice(len(upsampled_points), excess_count, replace=False)
        upsampled_points = np.delete(upsampled_points, delete_indices, axis=0)

    # 返回前再次检查输出的合法性
    if np.isnan(upsampled_points).any() or np.isinf(upsampled_points).any():
        raise ValueError("Upsampled points contain NaN or Inf values.")

    return upsampled_points
def pc_normalize(pc):
    # 检查点云是否为空
    if pc.shape[0] == 0:
        raise ValueError("Point cloud is empty.")
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m == 0:
        # print("All points in the cloud overlap. Returning the original point cloud.")
        return pc
    pc = pc / m
    return pc
def downsample(points, num_samples):
    """
    根据密度进行点云采样，低密度区域更多保留，密度高的区域减少保留。

    参数:
        points (np.ndarray): 所有点云数据 (N, 5)。
        num_samples(float): 表示采样后点数。
    返回:
        sampled_points (np.ndarray): 采样后的点云数据。
    """
    all_points = points[:, :3]
    density = points[:, 3]
    # 为避免密度为0的问题，添加一个小值 epsilon
    epsilon = 1e-6
    sampling_prob = 1 / (density + epsilon)
    sampling_prob /= sampling_prob.sum()  # 归一化为概率分布
    # 根据采样概率选择点
    sampled_indices = np.random.choice(len(all_points), size=num_samples, replace=False, p=sampling_prob)
    sampled_points = points[sampled_indices]

    return sampled_points
def unit(points,labels,npoints=4096):
    point_clouds, points_num = read_data(points, labels)
    frame = [index + 1 for index, count in enumerate(points_num) for _ in range(count)]
    all_points, density = compute_density(point_clouds)
    density = density[:, np.newaxis]
    frame = np.array(frame)[:, np.newaxis]
    points_cloud_frames = np.hstack((all_points, density, frame))

    if len(points_cloud_frames) < npoints:
        sample_points = upsample(points_cloud_frames, npoints)
    elif len(points_cloud_frames) > npoints:
        sample_points = downsample(points_cloud_frames, npoints)
    else:
        sample_points = points_cloud_frames

    sample_points[:, 0:3] = pc_normalize(sample_points[:, 0:3])

    return sample_points
def process(path,output_dir):
    class_path = os.listdir(path)
    counter = {}
    for class_name in class_path:
        path_labels = os.listdir(os.path.join(path, class_name))
        for path_label in path_labels:
            point_path = os.path.join(path, class_name, path_label)
            points = os.path.join(point_path, 'pcd')
            labels = os.path.join(point_path, 'label')
            xml_file = os.path.join(point_path, 'point_clouds.xml')
            class_type, _ = parse_xml(xml_file)
            if class_name != 'data':
                num = counter.get(class_type, 0)
                filename = 'a' + f"{int(class_type):02d}" + '_e' + f"{num:02d}" + '_sdepth'
                counter[class_type] = num + 1
                seg_point_cloud, _ = read_data(points, labels)
                samples_points = unit(points, labels)
                np.savez_compressed(os.path.join(output_dir, os.path.basename(filename).split('.')[0] + '.npz'),
                                    point_clouds=samples_points)
                print('saved npz file:', filename)
            if class_name == 'data' and int(class_type) <= 3:
                num = counter.get(class_type, 0)
                filename = 'a' + f"{int(class_type):02d}" + '_e' + f"{num:02d}" + '_sdepth'
                counter[class_type] = num + 1
                samples_points = unit(points, labels)
                np.savez_compressed(os.path.join(output_dir, os.path.basename(filename).split('.')[0] + '.npz'),
                                    point_clouds=samples_points)
                print('saved npz file:', filename)
if __name__ == '__main__':
    path = 'data/ahu-origin/test_already'
    output_dir = 'data/ahu-origin/save_cls'
    process(path, output_dir)