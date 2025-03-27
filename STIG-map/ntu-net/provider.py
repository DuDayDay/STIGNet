import numpy as np
from utils.data_preprocess import handle, visualize_point_cloud
def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def rotate_and_scale_point_cloud_x_axis(sample_points, points_frames, angle_range=(0, np.pi/10), scale_low=0.8, scale_high=1.25):
    """
    对点云进行随机缩放和沿 x 轴的随机旋转，确保 sample_points 和 points_frames 使用相同的旋转和缩放参数（批量并行处理）。
    :param sample_points: numpy 数组，形状为 [B, N, 5]，包含 (x, y, z) 坐标和特征信息。
    :param points_frames: numpy 数组，形状为 [B, F, P, 3]，包含 (x, y, z) 坐标。
    :param angle_range: 旋转角度范围 (theta_min, theta_max)，单位为弧度。
    :param scale_low: 缩放范围下限。
    :param scale_high: 缩放范围上限。
    :return: (rotated_scaled_sample_points, rotated_scaled_points_frames)
             - rotated_scaled_sample_points: 旋转和缩放后的 sample_points，形状为 [B, N, 5]。
             - rotated_scaled_points_frames: 旋转和缩放后的 points_frames，形状为 [B, F, P, 3]。
    """
    B, N, _ = sample_points.shape
    _, F, P, _ = points_frames.shape
    # visualize_point_cloud(sample_points[0])
    # 随机生成每个批次的旋转角度 θ 和缩放因子 scale
    thetas = np.random.uniform(angle_range[0], angle_range[1], size=B)  # [B]
    scales = np.random.uniform(scale_low, scale_high, size=B)          # [B]

    # 构造批量沿 X 轴的旋转矩阵，形状为 [B, 3, 3]
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    rotation_matrices = np.stack([
        np.stack([cos_thetas, np.zeros(B), sin_thetas], axis=-1),  # 第一行：X轴变换
        np.stack([np.zeros(B), np.ones(B), np.zeros(B)], axis=-1),  # 第二行：Y轴固定
        np.stack([-sin_thetas, np.zeros(B), cos_thetas], axis=-1)  # 第三行：Z轴变换
    ], axis=1)

    # 对 sample_points 处理
    coords_sample = sample_points[:, :, :3]  # 提取坐标部分，形状 [B, N, 3]
    rotated_sample_coords = np.einsum('bij,bnj->bni', rotation_matrices, coords_sample)  # [B, N, 3]
    scaled_sample_coords = rotated_sample_coords * scales[:, None, None]                # 缩放 [B, N, 3]

    # 保留特征部分，拼接旋转缩放后的坐标和特征
    rotated_scaled_sample_points = np.concatenate([scaled_sample_coords, sample_points[:, :, 3:]], axis=-1)

    # 对 points_frames 处理
    coords_frames = points_frames[:, :, :, :3]  # [B, F, P, 3]

    # 扩展旋转矩阵和缩放因子维度以支持广播
    rotation_expanded = rotation_matrices[:, None, None, :, :]  # [B, 1, 1, 3, 3]
    scales_expanded = scales[:, None, None, None]  # [B, 1, 1, 1]

    # 应用旋转矩阵到每个点的坐标（保持特征不变）
    rotated_frames_coords = np.einsum('bfpij,bfpj->bfpi', rotation_expanded, coords_frames)  # [B, F, P, 3]

    # 应用缩放
    scaled_frames_coords = rotated_frames_coords * scales_expanded  # [B, F, P, 3]

    # 拼接回原始特征（第4个通道）
    rotated_scaled_points_frames = np.concatenate([
        scaled_frames_coords,
        points_frames[:, :, :, 3:]  # 原始特征部分
    ], axis=-1)  # [B, F, P, 4]
    # visualize_point_cloud(sample_points[0])
    # visualize_point_cloud(rotated_scaled_sample_points[0])
    # a = rotated_scaled_points_frames.reshape(8,6144,4)
    # visualize_point_cloud(a[0])
    return rotated_scaled_sample_points, rotated_scaled_points_frames


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc
