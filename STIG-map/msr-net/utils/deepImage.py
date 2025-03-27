import numpy as np
import plotly.graph_objs as go
from scipy.spatial import cKDTree
from utils.data_preprocess import compute_density, align_point_cloud_to_maximize_z, align_point_cloud_to_maximize_x_keepZ
def read_header(file):
    """
    读取文件头，提取深度图的维度和帧数。
    :param file: 文件对象
    :return: dims (宽高), num_frames (帧数)
    """
    num_frames = np.frombuffer(file.read(4), dtype=np.uint32)[0]
    dims = (
        np.frombuffer(file.read(4), dtype=np.uint32)[0],  # 高度
        np.frombuffer(file.read(4), dtype=np.uint32)[0],  # 宽度
    )
    return dims, num_frames

def load_depth_map(path):
    """
    加载深度数据文件，并解析为深度图序列。
    :param path: .bin 文件路径
    :return: 深度图序列（列表，每帧是二维 numpy 数组）
    """
    with open(path, 'rb') as file:
        dims, num_frames = read_header(file)  # 读取文件头
        file_data = np.fromfile(file, dtype=np.uint32)  # 读取剩余所有数据

    depth_count_per_map = dims[0] * dims[1]  # 每帧像素数
    depth_map = []

    for i in range(num_frames):
        start = i * depth_count_per_map
        end = (i + 1) * depth_count_per_map
        current_depth_data = file_data[start:end]
        depth_map.append(current_depth_data.reshape(dims[1], dims[0]).T)  # 转为二维矩阵并转置

    return depth_map
def random_downsample_to_count(point_cloud, target_num_points):
    num_points = point_cloud.shape[0]
    if target_num_points >= num_points:
        raise ValueError("Target number of points must be less than the current number of points")

    indices = np.random.choice(num_points, target_num_points, replace=False)
    downsampled_point_cloud = point_cloud[indices]
    return downsampled_point_cloud
def interpolate_upsample_to_count(point_cloud, target_num_points):
    num_points = point_cloud.shape[0]
    if target_num_points <= num_points:
        raise ValueError("Target number of points must be greater than the current number of points")

    num_new_points = target_num_points - num_points
    tree = cKDTree(point_cloud)
    random_indices = np.random.choice(num_points, num_new_points)
    random_points = point_cloud[random_indices]

    distances, indices = tree.query(random_points, k=2)
    new_points = (point_cloud[indices[:, 0]] + point_cloud[indices[:, 1]]) / 2

    upsampled_point_cloud = np.vstack((point_cloud, new_points))
    return upsampled_point_cloud

def load_skeleton_data(file_path, num_joints=20):
    """
    加载骨架数据文件并拆分为每帧 20 个关节的数据。

    :param file_path: 骨架数据文件路径
    :param num_joints: 每帧的关节数量（默认为 20）
    :return: 每帧的关节数据，形状为 (num_frames, num_joints, 4)
    """
    # 加载骨架数据文件
    skeleton_data = np.loadtxt(file_path)  # 假设数据按行存储，每行4个数 (u, v, d, c)

    # 骨架数据的总行数应该是 num_frames * num_joints
    num_frames = skeleton_data.shape[0] // num_joints

    # 将数据重塑为 (num_frames, num_joints, 4)
    skeleton_data_reshaped = skeleton_data.reshape(num_frames, num_joints, 4)

    return skeleton_data_reshaped


import matplotlib.pyplot as plt
import time


def show_depth_sequence(depth_sequence):
    """
    显示深度图序列（以动画的形式）。
    :param depth_sequence: 深度图序列（列表，每帧是二维 numpy 数组）
    """
    plt.figure()
    for depth_map in depth_sequence:
        show_depth_map(depth_map)
        time.sleep(0.001)  # 暂停以显示帧

    plt.show()


def show_depth_map(depth_map):
    """
    显示单帧深度图。
    :param depth_map: 单帧深度图（二维 numpy 数组）
    """
    plt.imshow(depth_map, cmap='gray')
    plt.colorbar()
    plt.gca().invert_yaxis()  # MATLAB 中的 'YDir', 'reverse'
    plt.axis([0, depth_map.shape[1], 0, depth_map.shape[0]])
    plt.draw()
    plt.pause(0.001)  # 短暂暂停以刷新显示


def depth_image_to_pointcloud(depth_image, camera_intrinsics, num_points = 512):
    """
    将深度图像转换为点云，并剔除掉深度为0的点，最后对剩余点进行归一化。

    :param depth_image: 深度图像，形状为 (height, width)
    :param camera_intrinsics: 相机内参矩阵 K，形状为 (3, 3)
    :return: 3D 点云，形状为 (num_points, 3)
    """
    height, width = depth_image.shape

    # 相机内参
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))  # (width, height) 网格

    # 转换为三维空间坐标 (X, Y, Z)
    Z = depth_image.flatten()  # 深度值
    X = (u.flatten() - cx) * Z / fx
    Y = (v.flatten() - cy) * Z / fy

    # 创建一个 (X, Y, Z) 数组
    pointcloud = np.stack([X, Y, Z], axis=-1)

    # 剔除掉深度值为0的点
    valid_points = pointcloud[Z > 0]
    if valid_points.shape[0] < num_points:
        valid_points = interpolate_upsample_to_count(valid_points, num_points)
    elif valid_points.shape[0] > num_points:
        valid_points = random_downsample_to_count(valid_points, num_points)
    # 归一化
    min_vals = valid_points.min(axis=0)
    max_vals = valid_points.max(axis=0)

    # 归一化到 [0, 1] 范围
    normalized_points = (valid_points - min_vals) / (max_vals - min_vals)
    normalized_points[:, 1] *= 0.5
    normalized_points[:, 2] *= 0.5
    return normalized_points

def visualize_point_cloud(point_cloud):
    """
    使用 Plotly 可视化点云。
    :param point_cloud: 点云数据，包含坐标 (x, y, z) 和特征。
    """
    # 提取坐标和特征
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    features = point_cloud[:, 3] if point_cloud.shape[1] > 3 else np.zeros_like(x)

    # 创建散点图
    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,  # 点的大小
            color=features,  # 使用特征值作为颜色
            colorscale='Viridis',  # 配色方案
            colorbar=dict(title="Feature"),  # 显示颜色条
        ),
        name='Point Cloud'
    )

    # 设置布局
    layout = go.Layout(
        title="Point Cloud Visualization",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    # 创建并显示图表
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()


def uniform_sample_to_target_frames(data, target_frames=16,random=False):
    """
    将数据从原始的帧数均匀采样到目标帧数。

    :param data: 原始数据，形状为 (original_frames, height, width)
    :param target_frames: 目标帧数
    :return: 均匀采样后的数据，形状为 (target_frames, height, width)
    """
    original_frames = data.shape[0]
    # print("Original frames: ", original_frames)
    if random is False and original_frames - target_frames-1 > 0:
        start_frame = np.random.randint(0, int((original_frames - target_frames-1)/8)+1)
    else:
        start_frame = 0
    # 生成均匀分布的帧索引
    frame_indices = np.linspace(start_frame, original_frames-1, target_frames).astype(int)
    # frame_indices = np.clip(frame_indices, 0, original_frames - 1)

    # 根据生成的索引选择帧
    sampled_data = data[frame_indices]

    return sampled_data
def process_skeleton_and_depth( depth_images, camera_intrinsics, num_points = 512):
    """
    处理骨架数据和深度图像，生成每帧的点云。

    :param depth_images: 深度图像数据，形状为 (num_frames, height, width)
    :param camera_intrinsics: 相机内参矩阵 K，形状为 (3, 3)
    :return: 所有帧的点云数据，形状为 (num_frames, height * width, 3)
    """
    num_frames = depth_images.shape[0]
    all_pointclouds = []

    for i in range(num_frames):
        # 获取当前帧的深度图像
        depth_image = depth_images[i]

        # 将深度图像转换为点云
        pointcloud = depth_image_to_pointcloud(depth_image, camera_intrinsics, num_points)

        # 存储该帧的点云
        all_pointclouds.append(pointcloud)

    return np.array(all_pointclouds)

def load_pointcloud(file_path, target_frames = 16,num_points = 512,is_test=False):

    # 加载深度数据
    depth_map_array = np.array(load_depth_map(file_path))

    # 相机内参，示例为 Kinect 相机的内参
    camera_intrinsics = np.array([[525.0, 0.0, 319.5],  # fx, cx
                                  [0.0, 525.0, 239.5],  # fy, cy
                                  [0.0, 0.0, 1.0]])  # 常用 Kinect 参数
    pointclouds = process_skeleton_and_depth(depth_map_array, camera_intrinsics, num_points)
    pointclouds = uniform_sample_to_target_frames(pointclouds, target_frames=target_frames,random=is_test)
    videos_array = pointclouds
    seg_point_cloud = []
    first = 0
    first_2 = 0
    for video in videos_array:
        centroid = video.mean(axis=0)
        # 将点云平移，使质心位于原点
        points_seg = video - centroid
        # if points_seg is not None:
        #     if first == 0:
        #         rotation_matrix = align_point_cloud_to_maximize_z(points_seg)
        #         first = 1
        #     aligned_point_cloud = np.dot(points_seg, rotation_matrix.T)
        #     # 将质心平移到原点
        #     centroid = aligned_point_cloud.mean(axis=0)
        #     aligned_point_cloud -= centroid
        #     points_seg = aligned_point_cloud
        #     if first_2 == 0:
        #         rotation_matrix_2 = align_point_cloud_to_maximize_x_keepZ(points_seg)
        #         first_2 = 1
        #     points_seg = np.dot(points_seg, rotation_matrix_2.T)
        seg_point_cloud.append(points_seg)
    point_clouds = seg_point_cloud
    all_points, density = compute_density(point_clouds)
    frame = [i for i in range(1, target_frames + 1) for _ in range(num_points)]
    density = density[:, np.newaxis]
    frame = np.array(frame)[:, np.newaxis]
    points_cloud_frames = np.hstack((all_points, density, frame))
    # npoints = 4096
    # if len(points_cloud_frames) < npoints:
    #     sample_points = upsample(points_cloud_frames, npoints)
    # elif len(points_cloud_frames) > npoints:
    #     sample_points = downsample(points_cloud_frames, npoints)
    # else:
    #     sample_points = points_cloud_frames
    # 使用 Open3D 可视化
    # visualize_point_cloud(point_clouds[1])
    return points_cloud_frames, np.array(point_clouds)

if __name__ == '__main__':
    # load_pointcloud('msr_datasets/Depth/a10_s01_e01_sdepth.bin', target_frames=16, num_points=512)
    load_pointcloud('msr_datasets/Depth/a10_s01_e02_sdepth.bin', target_frames=16, num_points=512)