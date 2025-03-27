import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention, MultiHeadSelfAttention, AttentionFusion, Attention, CrossAttention
from utils.data_preprocess import handle, visualize_point_cloud
import time
def nearest_power_of_two_numpy(n):
    """
    使用 numpy 找到离 n 最近的 2 的次方值。
    :param n: 输入整数或浮点数。
    :return: 最近的 2 的次方值。
    """
    if n <= 0:
        raise ValueError("输入必须是正数")

    # 计算 2 的次方
    log2_n = np.log2(n)
    lower_power = 2 ** np.floor(log2_n)  # 小于或等于 n 的最大 2 次方
    upper_power = 2 ** np.ceil(log2_n)  # 大于或等于 n 的最小 2 次方

    # 返回距离最近的 2 的次方值
    return lower_power if abs(n - lower_power) <= abs(n - upper_power) else upper_power
def single_farthest_point_sample(points, num_samples):
    """
    对点云数据进行最远距离采样 (FPS)。

    :param points: 输入点云，形状为 [n, 3]。
    :param num_samples: 需要采样的点数。

    :return: 最远距离采样的点的索引，形状为 [num_samples]。
    """
    n = points.shape[0]  # 点的数量
#     nn = nearest_power_of_two_numpy(n)
    # 初始化采样结果存储
    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=points.device)

    # 初始化距离：设置为无穷大，表示未采样点的初始最远距离
    dist = torch.full((n,), float('inf'), device=points.device)

    # 随机选择一个初始点作为第一个采样点
    sampled_indices[0] = torch.randint(0, n, (1,), device=points.device)

    for i in range(1, num_samples):
        # 取出上一个采样点的坐标
        last_point = points[sampled_indices[i - 1]].unsqueeze(0)  # [1, 3]

        # 计算所有点到最近采样点的距离
        dists_to_last = torch.norm(points - last_point, dim=1)

        # 更新每个点的最小距离
        dist = torch.minimum(dist, dists_to_last)

        # 选择最远的点作为下一个采样点
        sampled_indices[i] = torch.argmax(dist)

    return sampled_indices
def farthest_point_sample(xyz, npoint, cut=128):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    dist_idx_all = torch.zeros(B, 0, cut, dtype=torch.long).to(device)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N,dtype=torch.long).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist_sort = torch.argsort(dist, dim=1, descending=False)
        dist_sort = dist_sort[..., :cut].unsqueeze(1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        dist_idx_all = torch.cat([dist_idx_all, dist_sort], dim=1)
    return centroids, dist_idx_all



def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
def get_neighbors_points(points_tensor, neighbors_batch):
    """
    根据 neighbors_batch 中的索引找到对应的坐标和密度信息，并重组为形状为 [B, num_samples, num_per_sample, channels]。
    坐标和密度信息放在一起，channels = 4（3个坐标和1个密度）。

    :param points_tensor: Tensor格式的点云数据，形状为 [B, N, C]。
                          C 包括坐标和密度信息。
    :param neighbors_batch: 包含每个采样点邻域点索引的 batch，形状为 [B, num_samples, num_per_sample]。

    :return: 返回形状为 [B, num_samples, num_per_sample, C] 的 tensor。
    """
    device = points_tensor.device
    B, N, C = points_tensor.shape  # 点云的形状
    _, num_samples, num_per_sample = neighbors_batch.shape  # 邻域索引的形状

    # 为 batch 维度创建索引
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1, 1)  # [B, 1, 1]
    batch_indices = batch_indices.expand(-1, num_samples, num_per_sample)  # [B, num_samples, num_per_sample]

    # 使用高级索引提取邻域点
    neighbor_coords_and_densities = points_tensor[batch_indices, neighbors_batch]  # [B, num_samples, num_per_sample, C]

    return neighbor_coords_and_densities

def count_points_in_radius(xyz, centroids, radius):
    """
    统计每个采样点（球中心）内的点数

    Args:
        xyz: 原始点云数据, [B, N, 3]
        centroids: 采样点索引, [B, npoint]
        radius: 球的半径
    Returns:
        counts: 每个采样点内的点数, [B, npoint]
    """
    B, N, _ = xyz.shape
    _, npoint = centroids.shape

    # 获取采样点的坐标 [B, npoint, 3]
    sampled_points = xyz[torch.arange(B).unsqueeze(1), centroids]  # [B, npoint, 3]

    # 计算原始点云中每个点到采样点的距离 [B, npoint, N]
    distances = torch.cdist(sampled_points, xyz, p=2)

    # 判断距离是否小于半径，统计满足条件的点数
    counts = (distances <= radius).sum(dim=-1)  # [B, npoint]
    return counts
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist
def get_nearest_points(center, sample_points, sample_feature, neighbors, k=16):
    """
    提取每个类的特征及其对应的邻居索引。

    :param center: [B, n_sample, C]，每个类的中心点坐标。
    :param sample_points: [B, n_sample, n_per_sample, C]，每个类中的点坐标。
    :param sample_feature: [B, n_sample, n_per_sample, C_prime]，每个类中的点的特征。
    :param neighbors: [B, n_sample, n_per_sample]，每个类中的点在全局点云中的索引。
    :param k: 每个类返回的邻居数目。
    :return: [B, n_sample * k, C_prime]，提取的邻居特征。
    """
    B, n_sample, n_per_sample, C = sample_points.shape
    _, _, _, C_prime = sample_feature.shape
    _, _, n_neighbors = neighbors.shape

    # 计算每个类中的所有点到中心点的距离
    center_expanded = center.unsqueeze(2).expand(-1, -1, n_per_sample, -1)  # [B, n_sample, n_per_sample, C]
    sample_points_expanded = sample_points  # [B, n_sample, n_per_sample, C]

    # 计算每个点到中心的距离
    dist = torch.norm(sample_points_expanded - center_expanded, dim=-1)  # [B, n_sample, n_per_sample]

    # 找到最近的k个邻居
    _, topk_idx = torch.topk(dist, k, dim=-1, largest=True, sorted=False)  # [B, n_sample, k]

    # 提取对应的特征
    topk_sample_feature = torch.gather(sample_feature, 2,
                                       topk_idx.unsqueeze(-1).expand(-1, -1, -1, C_prime))  # [B, n_sample, k, C_prime]

    # 获取邻居点的全局索引
    topk_neighbors = torch.gather(neighbors, 2, topk_idx)  # [B, n_sample, k]

    # 重塑输出结果
    output = topk_sample_feature.view(B, n_sample * k, C_prime)  # [B, n_sample * k, C_prime]

    return output, topk_neighbors.view(B, n_sample*k)
def closest_points(kernel, xyz, new_xyz, density_weight = None):
    """
    Input:
        kernel: number of closest points to select (instead of radius)
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, kernel]
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx_density = None
    normalized_dist_density = None
    min_dist_density = None
    # 计算每个查询点到所有原始点的距离
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    group_idx = sqrdists.argsort(dim=-1)[:, :, :kernel]  # [B, S, kernel]
    group_distances = sqrdists.gather(2, group_idx)
    # normalized_dist = (group_distances - group_distances.mean(dim=-1, keepdim=True)) / torch.sqrt(group_distances.var(dim=-1, keepdim=True) + 1e-8)
    min_dist = group_distances.mean(dim=-1, keepdim=True)
    if density_weight is not None:
        # 确保 density_weight 的形状为 [B, N]
        if density_weight.shape[-1] == 1:
            density_weight = density_weight.squeeze(-1)  # [B, N]

        # 检查 density_weight 的大小是否与 xyz 的第二维匹配
        assert density_weight.shape[1] == N, \
            f"Density weight shape mismatch: expected {N}, but got {density_weight.shape[1]}"

        # 使用密度权重调整距离值
        density_adjusted = density_weight ** 2 # 避免除以零
        sqrdists = sqrdists * density_adjusted.unsqueeze(1)  # [B, S, N]
        group_idx_density = sqrdists.argsort(dim=-1)[:, :, :kernel]
        group_distances = sqrdists.gather(2, group_idx_density)
        # normalized_dist_density = (group_distances - group_distances.mean(dim=-1, keepdim=True)) / torch.sqrt(group_distances.var(dim=-1, keepdim=True) + 1e-8)
        min_dist_density = group_distances.mean(dim=-1, keepdim=True)
        # 获取每个查询点最近的 `kernel` 个点的索引
    # group_idx = sqrdists.argsort(dim=-1)[:, :, :kernel]  # [B, S, kernel]

    # return group_idx, group_idx_density, normalized_dist, normalized_dist_density
    return group_idx, group_idx_density, min_dist, min_dist_density
def find_nearest_neighbors(sample_points, points_frames, K):
    """
    从 points_frames 中找到 sample_points 中每个点最近的 K 个点。

    Args:
        sample_points (torch.Tensor): Tensor形状为 [B, N, 4]，包含坐标信息 (x, y, z) 和帧信息。
        points_frames (torch.Tensor): Tensor形状为 [B, F, P, 3]，包含帧数 (F)、每帧点云数量 (P) 的点云坐标。
        K (int): 最近邻点的个数。

    Returns:
        torch.Tensor: 返回形状为 [B, N, K, 4] 的张量，其中最后一维为最近邻点的 (x, y, z) 坐标和帧信息。

    """


    B, N, _ = sample_points.shape
    _, F, P, _ = points_frames.shape

    # 分离 sample_points 的坐标信息和帧信息
    sample_coords = sample_points[:, :, :3]  # [B, N, 3]
    density = sample_points[:, :, 3]
    sample_frames = sample_points[:, :, -1].long() - 1  # [B, N]

    # 初始化结果张量
    nearest_neighbors = torch.zeros(B, N, K, 5, device=sample_points.device)

    # 构造帧索引掩码，提取每个点对应的帧数据
    frame_indices = sample_frames.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, P, 4)  # [B, N, P, 3]
    selected_points_frames = torch.gather(points_frames, 1, frame_indices).float()  # [B, N, P, 3]

    # 计算每个点与其对应帧点云的欧几里得距离
    sample_coords_expanded = sample_coords.unsqueeze(2)  # [B, N, 1, 3]
    dists = torch.cdist(sample_coords_expanded, selected_points_frames[...,:3], p=2)  # [B, N, P]

    # 找到最近的 K 个点
    nearest_indices = torch.topk(dists, K, dim=-1, largest=False).indices  # [B, N, K]
    nearest_indices_expanded = nearest_indices.permute(0, 1, 3, 2)


    # 根据最近邻索引提取点云数据
    nearest_points = torch.gather(selected_points_frames, 2, index =  nearest_indices_expanded.expand(-1, -1, -1, 4))

    # 合并帧信息
    nearest_neighbors[:, :, :, :4] = nearest_points  # 最近点的坐标 [B, N, K, 3]
    nearest_neighbors[...,:3] = nearest_neighbors[...,:3] - sample_coords_expanded
    nearest_neighbors[:, :, :, -1] = sample_frames.unsqueeze(-1).expand(-1, -1, K)  # 帧信息 [B, N, K]
    # nearest_neighbors[:, :, :, 3] = density.unsqueeze(-1).expand(-1, -1, K)  # 帧信息 [B, N, K]
    # a = nearest_neighbors.view(B, N*36, 5)
    # b = a[0]
    # visualize_point_cloud(b)
    return nearest_neighbors
def visualize(pixel_features, num_samples=4, channel=0):
    """
    可视化像素化后的特征图。

    参数：
    - pixel_features: [B, H, W, C]，像素化后的特征张量。
    - num_samples: int，显示的样本数量。
     - channel: int，展示特定通道的特征。
    """
    # 转为 numpy 进行可视化
    pixel_features = pixel_features[:, :, :, channel].detach().cpu().numpy()

    # 限制样本数量
    B = pixel_features.shape[0]
    num_samples = min(num_samples, B)

    # 创建可视化子图
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(pixel_features[i], cmap='hot', interpolation='nearest')
        ax.set_title(f"Sample {i+1} (Channel {channel})")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
def pixelize_point_cloud_batch(points, resolution=64):
    """
    将点云数据批量投影到二维像素网格上（并行处理）。

    参数：
    - points: [B, N, 3] 的点云张量，B 是批量大小，N 是点的数量，每个点包含 (x, y, z)。
    - resolution: 图像的分辨率（像素网格大小），默认 256x256。

    返回：
    - pixel_maps: [B, resolution, resolution] 的张量，表示每个点云的像素化结果。
    """
    B, N, _ = points.shape

    # 取出 x 和 y 坐标用于投影
    xy = points[:, :, :2]  # [B, N, 2]

    # 归一化到 [0, 1] 范围（每个 batch 独立计算归一化）
    min_xy = xy.amin(dim=1, keepdim=True)  # [B, 1, 2]
    max_xy = xy.amax(dim=1, keepdim=True)  # [B, 1, 2]
    normalized_xy = (xy - min_xy) / (max_xy - min_xy + 1e-6)  # [B, N, 2]

    # 将归一化坐标映射到 [0, resolution-1] 范围，并转为整数索引
    pixel_coords = (normalized_xy * (resolution - 1)).long()  # [B, N, 2]

    # 初始化像素网格
    pixel_maps = torch.zeros(B, resolution, resolution, dtype=torch.float32, device=points.device)

    # 扁平化索引用于 batch 并行
    batch_indices = torch.arange(B, device=points.device).view(-1, 1).repeat(1, N)  # [B, N]
    x_coords = pixel_coords[:, :, 0]  # [B, N]
    y_coords = pixel_coords[:, :, 1]  # [B, N]

    # 使用 scatter_add 在批量中并行累加点的数量
    pixel_maps.index_put_((batch_indices.flatten(), x_coords.flatten(), y_coords.flatten()),
                          torch.ones(B * N, device=points.device), accumulate=True)

    return pixel_maps
def pixelize(points, features, resolution=256, pooling="mean", shadow='xy'):
    """
    将点云和其对应的特征映射到像素网格。
    :param points: [B, N, 3]，点云坐标
    :param features: [B, N, C]，点云特征
    :param resolution: int，像素网格分辨率
    :param pooling: str，冲突处理方式，"mean" 或 "max"
    :return: [B, H, W, C] 的像素化特征
    """
    B, N, _ = points.shape  # B: batch size, N: num points
    _, _, C = features.shape  # C: feature dimension
    if shadow == 'xy':
        # --- 1. 归一化点云到 [0, 1]
        min_vals = points[:, :, :2].amin(dim=1, keepdim=True)  # [B, 1, 2] 获取 x 和 y 最小值
        max_vals = points[:, :, :2].amax(dim=1, keepdim=True)  # [B, 1, 2] 获取 x 和 y 最大值
        normalized_xy = (points[:, :, :2] - min_vals) / (max_vals - min_vals + 1e-6)  # [B, N, 2] 归一化 x 和 y 坐标
    elif shadow == 'xz':
        # --- 1. 归一化点云到 [0, 1]
        min_vals = points[:, :, [0, 2]].amin(dim=1, keepdim=True)  # [B, 1, 2] 获取 x 和 z 最小值
        max_vals = points[:, :, [0, 2]].amax(dim=1, keepdim=True)  # [B, 1, 2] 获取 x 和 z 最大值
        normalized_xy = (points[:, :, [0, 2]] - min_vals) / (max_vals - min_vals + 1e-6)  # [B, N, 2] 归一化 x 和 z 坐标
    elif shadow == 'yz':
        # --- 1. 归一化点云到 [0, 1]
        min_vals = points[:, :, [1, 2]].amin(dim=1, keepdim=True)  # [B, 1, 2] 获取 x 和 y 最小值
        max_vals = points[:, :, [1, 2]].amax(dim=1, keepdim=True)  # [B, 1, 2] 获取 x 和 y 最大值
        normalized_xy = (points[:, :, [1, 2]] - min_vals) / (max_vals - min_vals + 1e-6)  # [B, N, 2] 归一化 x 和 y 坐标
    # --- 2. 映射到像素网格 [0, resolution-1]
    pixel_coords = (normalized_xy * (resolution - 1)).long()  # [B, N, 2]
    pixel_coords = torch.clamp(pixel_coords, min=0, max=resolution - 1)  # 确保索引范围合法

    # --- 3. 计算像素线性索引
    pixel_indices = pixel_coords[:, :, 1] * resolution + pixel_coords[:, :, 0]  # [B, N]

    # --- 4. 初始化特征容器
    device = points.device
    pixel_features = torch.zeros(B, resolution * resolution, C, device=device)  # [B, H*W, C]
    pixel_counts = torch.zeros(B, resolution * resolution, 1, device=device)  # [B, H*W, 1]

    # --- 5. 使用 scatter_add_ 聚合特征
    pixel_features.scatter_add_(1, pixel_indices.unsqueeze(-1).expand(-1, -1, C), features)  # 特征累加
    pixel_counts.scatter_add_(1, pixel_indices.unsqueeze(-1), torch.ones_like(pixel_indices, device=device).unsqueeze(-1).float())  # 计数累加

    # --- 6. 根据 pooling 方式处理冲突
    if pooling == "mean":
        pixel_features = pixel_features / (pixel_counts + 1e-6)  # 平均值
    elif pooling == "max":
        # 使用 scatter_reduce 的最大值实现
        raise NotImplementedError("Max pooling 需要 PyTorch 2.0 的 scatter_reduce")
    elif pooling != "sum":
        raise ValueError(f"不支持的 pooling 方式: {pooling}")

    # --- 7. 恢复到 [B, H, W, C] 的形状
    pixel_features = pixel_features.view(B, resolution, resolution, C)
    if shadow == 'xy':
        pixel_features = pixel_features.permute(0, 2, 1, 3)
    return pixel_features

def plot_point_counts(counts):
    """
    绘制每批次的采样点计数图（按点数从小到大排序）


    Args:
        counts: 每个采样点内的点数, [B, npoint]
    """
    B, npoint = counts.shape
    heights_to_mark = [16, 32, 64, 128, 256]  # 标记的高度位置

    for b in range(B):
        # 对点数从小到大排序
        sorted_counts, indices = torch.sort(counts[b])

        # 绘制柱状图
        plt.figure(figsize=(12, 6))
        plt.bar(range(npoint), sorted_counts.cpu().numpy(), color="skyblue")

        # 添加水平标记线
        for h in heights_to_mark:
            plt.axhline(h, color="red", linestyle="--", linewidth=1)
            plt.text(npoint - 10, h + 5, f"{h}", color="red", fontsize=10)

        # 添加标题和标签
        plt.title(f"Batch {b+1}: Points in Each Sphere (Sorted)", fontsize=14)
        plt.xlabel("Sample Index (Sorted)", fontsize=12)
        plt.ylabel("Number of Points", fontsize=12)
        plt.show()
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def farthest_point_sampling_with_density(points_tensor, num_samples):
    """
    基于密度和最远距离的采样方法，同时记录每个采样点周围的点。
    在 Tensor 上直接操作，不转为 NumPy。

    :param points_tensor: Tensor格式的点云数据，形状为 [B, N, C]。
                          前3个维度是点坐标，最后一维是密度。
    :param num_samples: 每帧需要采样的点数。

    :return: 采样点的索引列表
    """
    # 解构输入 Tensor
    B, N, C = points_tensor.shape
    points = points_tensor[:, :, :3]  # 点坐标，形状 [B, N, 3]
    densities = points_tensor[:, :, 3]  # 点密度，形状 [B, N]

    # 初始化结果存储
    sampled_points_batch = torch.zeros(B, num_samples, dtype=torch.long, device=points.device)

    # 初始化每个 batch 的最远距离
    dist = torch.full((B, N), float('inf'), device=points.device)  # 初始最远距离（正无穷）

    # 遍历每个 batch
    for i in range(num_samples):
        # 计算权重（密度归一化）
        weights = densities / densities.sum(dim=1, keepdim=True)

        # 对每个 batch 单独进行采样
        if i == 0:
            # 随机选择初始点
            sampled_points = torch.multinomial(weights, 1)  # 随机选择第一个点，形状为 [B, 1]
            sampled_points_batch[:, i] = sampled_points.squeeze()  # 记录第一个采样点
        else:
            last_point = points[torch.arange(B), sampled_points_batch[:, i - 1]]  # 取出每个batch最后一个采样点 [B, 3]
            dists_to_last = torch.norm(points - last_point.unsqueeze(1), dim=2)  # 计算每个点到最后一个采样点的距离 [B, N]

            dist = torch.min(dist, dists_to_last)  # 更新每个点到当前采样点的最小距离

            # 密度加权
            weighted_dist = dist * weights
            farthest_point_idx = torch.argmax(weighted_dist, dim=1)  # 找到加权距离最大的点 [B]

            # 记录采样点
            sampled_points_batch[:, i] = farthest_point_idx

    return sampled_points_batch
def process_mask(a,sampled_center,num=5000):
    batch_size, num_nodes, num_points = a.shape
    device = a.device

    # 生成点云索引张量 [8, 16, 2048]
    idx = torch.arange(num_points, device=device).view(1, 1, -1).expand(batch_size, num_nodes, -1)

    # 计算累积和以确定前1000个True的位置
    counts = a.int()
    cum_counts = counts.cumsum(dim=-1)
    selected_mask = (cum_counts <= num) & a

    # 替换未选中的索引为num_points（超出实际范围的值）
    filled_idx = torch.where(selected_mask, idx, num_points)

    # 对索引进行排序，有效索引将排在前面
    sorted_idx, _ = filled_idx.sort(dim=-1)
    selected_indices = sorted_idx[..., :num]  # 取前1000个

    # 计算每个节点选中的有效数量
    counts_per_node = selected_mask.sum(dim=-1).clamp(min=1)  # 确保至少有一个

    # 生成需要替换的位置掩码
    pos = torch.arange(num, device=device).view(1, 1, -1).expand_as(selected_indices)
    replace_mask = pos >= counts_per_node.unsqueeze(-1)
    # 替换超出有效数量的位置
    final_indices = torch.where(replace_mask, sampled_center.unsqueeze(-1), selected_indices)

    return final_indices
def get_nearest_points_within_radius_parallel(sampled_center, points_tensor, n_per_sample=64, cut=256):
    """
    批量并行计算每个采样点到其他采样点的最近距离作为半径，
    并将落在该半径中的点的索引进行汇总。

    :param sampled_points: 采样点的索引，形状为 [B, num_samples]。
    :param points_tensor: 原始点云数据，形状为 [B, N, C]，前 3 个维度是坐标，最后一个维度是密度。

    :return: 每个采样点在其半径内的点的索引，形状为 [B, num_samples, N_nearest]。
    """
    B, N, C = points_tensor.shape
    _,S = sampled_center.shape
    # 获取所有点的坐标，形状 [B, N, 3]
    all_points = points_tensor[:, :, :3]
    sampled_coords = index_points(all_points, sampled_center)
    # 计算每个采样点到所有点的距离，形状 [B, num_samples, N]
    dists = torch.norm(sampled_coords.unsqueeze(2) - sampled_coords.unsqueeze(1) , dim=-1)
    dist_other = torch.norm(all_points.unsqueeze(2) - sampled_coords.unsqueeze(1), dim=3)# [B, num_samples, N]
    second_min_dists, _ = torch.topk(dists, 2, dim=1, largest=False, sorted=False)  # 获取前2小的距离
    # 使用第二小的距离作为半径
    radius = second_min_dists[:, -1, :] * 1.5  # 第二小的距离，形状 [B, num_samples]
    dist_other = dist_other.permute(0, 2, 1)
    within_radius = dist_other <= radius.unsqueeze(2).expand(-1, -1, N)# 生成坐标索引，形状为 [8, 16, 2048, 3]
    # ntu-60 used num=8000
    neighbors_idx = process_mask(within_radius,sampled_center, num=6144)

    gathered_points = all_points[torch.arange(B).unsqueeze(1).unsqueeze(2), neighbors_idx]
    gathered_points_flat = gathered_points.view(B*S, 6144, 3)
    FPS_gathered_points, dist_idx_all = farthest_point_sample(gathered_points_flat, n_per_sample,cut=cut)
    FPS_index = FPS_gathered_points.view(B, S, n_per_sample)
    dist_idx_all = dist_idx_all.view(B, S, n_per_sample,-1)
    local_idx = torch.gather(neighbors_idx.unsqueeze(2).expand(-1, -1, n_per_sample, -1), dim=3, index=dist_idx_all.long())
    neighbors_idx = neighbors_idx.unsqueeze(-1)
    result = torch.gather(neighbors_idx, dim=2, index=FPS_index.unsqueeze(-1)).squeeze(-1)
    # visualize_point_cloud(points_tensor[1],sampled_center[1])
    # local_idx = local_idx.view(B,S*n_per_sample*cut)
    # for i in range(B):
    #     visualize_point_cloud(points_tensor[i],local_idx[i])
    # for i in range(16):
    #     visualize_point_cloud(points_tensor[2],local_idx[2,i,:])
    return sampled_coords,result,local_idx
def time_encoding(grouped_features,frames):
    time_indices = grouped_features[..., 4]
    freq = 2 * torch.pi / frames
    sin_enc = torch.sin(freq * time_indices)
    cos_enc = torch.cos(freq * time_indices)
    time_encoding = torch.stack([cos_enc, sin_enc], dim=-1)
    grouped_features = torch.cat((grouped_features[..., :4], time_encoding), dim=-1)
    return grouped_features
class PointFeatureNet(nn.Module):
    def __init__(self, n_points, kernel_list, in_channel, mlp_list, attention=False):
        super(PointFeatureNet, self).__init__()
        self.attention = attention
        self.n_points = n_points
        self.kernel_list = kernel_list
        self.in_channel = in_channel
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.att = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            if attention:
                self.att.append(MultiHeadSelfAttention(mlp_list[i][-1], attention_dim=512, num_heads=4))
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, features):
        """
                Input:
                    xyz: input points position data, [B, N, C]
                    points: input points data, [B, D, N]
                Return:
                    new_xyz: sampled points position data, [B, N, S, C]
                    new_points_concat: sample points feature data, [B, D', S]
                """
        xyz = xyz.to(torch.float32)
        B,N,C = xyz.shape
        centroids, dist_idx_all = farthest_point_sample(xyz, self.n_points)
        xyz_new = index_points(xyz, centroids)
        new_features_list = []
        for i, kernel in enumerate(self.kernel_list):
            # group_idx, _, _, _ = closest_points(kernel, xyz, xyz_new)
            group_idx = dist_idx_all[..., :kernel]
            grouped_xyz = index_points(xyz, group_idx)
            # grouped_xyz -= xyz_new.view(B,self.n_points,1,C)
            if features is not None:
                grouped_features = index_points(features, group_idx)
                grouped_features = torch.cat([grouped_features, grouped_xyz], dim=-1)
            else:
                grouped_features = grouped_xyz
            grouped_features = grouped_features.permute(0, 3, 2, 1)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_features = F.relu(bn(conv(grouped_features)))
            new_feature = torch.max(grouped_features, 2)[0]  # [B, D', S]
            new_feature = new_feature.permute(0, 2, 1)
            if self.attention:
                new_feature, _ = self.att[i](new_feature)
            new_features_list.append(new_feature)
            new_feature_concat = torch.cat(new_features_list, dim=2)
        return xyz_new, new_feature_concat
class PointNetSetAbstraction(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        new_xyz, new_points = sample_and_group_all(xyz, points)
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class CrossPointNet(nn.Module):
    def __init__(self, in_channels, num_sample, global_sample, kernel_list, mlp_list, n_per_sample=64, out_channels=128, Fusion=True, DFPS = False, time_coding=True):
        super(CrossPointNet, self).__init__()
        self.time_coding = time_coding
        self.out_channels = out_channels
        self.kernel_list = kernel_list
        self.mlp_list = mlp_list
        self.num_sample = num_sample
        self.global_sample = global_sample
        self.n_per_sample = n_per_sample
        self.in_channels = in_channels
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.fusion = Fusion
        self.att2 = AttentionFusion(num_sample*global_sample, sum([row[-1] for row in mlp_list]), self.out_channels, 1.0 )
        self.att = nn.ModuleList()
        self.DFPS = DFPS
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            if self.time_coding:
                last_channel = in_channels + 4
            else:
                last_channel = in_channels + 3

            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
                # self.att.append(MultiHeadSelfAttention(mlp_list[i][-1], attention_dim=512, num_heads=8))
            self.att.append(CrossAttention(mlp_list[i][-1]))
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
    def forward(self,points,points_frames):
        points = points.to(torch.float32)
        B, N, C = points.shape
        _,frames,_,_ = points_frames.shape
        sampled_points_idx = farthest_point_sampling_with_density(points, self.num_sample)
        sampled_center, neighbors, local_idx =get_nearest_points_within_radius_parallel(sampled_points_idx, points, n_per_sample=self.n_per_sample)#0.4
        local_idx = local_idx.view(B,self.n_per_sample*self.num_sample,-1)
        sampled_points = get_neighbors_points(points, neighbors)
        sampled_points_concat = sampled_points.reshape(B, self.num_sample * self.n_per_sample, C)
        mix_features_list = []
        att_features_list = []
        for i, kernel in enumerate(self.kernel_list):
            if self.DFPS is False:
                # group_idx, _, dist, _ = closest_points(kernel, xyz, sampled_points_concat[:, :, 0:3])
                grouped_features = find_nearest_neighbors(sampled_points_concat, points_frames, kernel)
            else:
                nearest_neighbors = find_nearest_neighbors(sampled_points_concat, points_frames, kernel) #0.03s
                # group_idx, _, dist, _ = closest_points(kernel, xyz, sampled_points_concat[:, :, 0:3])
                group_idx =  local_idx[..., :kernel]
                grouped_features = index_points(points, group_idx)
                anchor_xyz = sampled_points_concat[...,:3]
                grouped_features[...,:3] = grouped_features[...,:3] - anchor_xyz.view(B, self.num_sample*self.n_per_sample,1,3)
                if self.time_coding == True:
                    grouped_features = time_encoding(grouped_features, frames)
                    nearest_neighbors = time_encoding(nearest_neighbors, frames)
                grouped_features = torch.cat([grouped_features, nearest_neighbors], -2)
            grouped_features = grouped_features.permute(0, 3, 2, 1)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_features = F.relu(bn(conv(grouped_features)))
            grouped_features = grouped_features.permute(0, 3, 2, 1)
            mix_feature = torch.max(grouped_features[:, :, 0:kernel, :], dim=2)[0]
            unmix_feature = torch.max(grouped_features[:, :, kernel:, :], dim=2)[0]
            att_feature = self.att[i](mix_feature, unmix_feature, unmix_feature) + unmix_feature
            mix_features_list.append(mix_feature)
            att_features_list.append(att_feature)
            new_feature_concat = torch.cat(mix_features_list, dim=2)
            att_feature_concat = torch.cat(att_features_list, dim=2)
        sampled_features = new_feature_concat.view(B, self.num_sample, self.n_per_sample, -1)
        global_features, idx = get_nearest_points(sampled_center, sampled_points[:, :, :, 0:3], sampled_features, neighbors, k=self.global_sample)
        if self.fusion:
            global_features = self.att2(global_features, sampled_features)

        features = global_features.view(B, self.num_sample*self.n_per_sample, -1)
        features = torch.cat([features, att_feature_concat],-1)
        # print(sampled_points_idx.shape)
        # visualize_point_cloud(np.array(points[0]), np.array(sampled_points_idx[0].view(-1)))
        # visualize_point_cloud(np.array(points[0]), np.array(neighbors[0].view(-1)))
        # visualize_point_cloud(np.array(points[0]), np.array(idx[0]))
        # visualize_point_cloud(np.array(points[0]), np.array(group_idx.reshape(B, -1)[0]))
        # visualize_point_cloud(sampled_points_concat[0])
        return sampled_points_concat[:, :, 0:3], features
class DirPredictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DirPredictor, self).__init__()

        # 定义卷积-激活-池化模块
        self.conv_block_3x3 = self._make_conv_block_3x3(in_channels*3, out_channels=256)
        self.conv_block_3x3_2 = self._make_conv_block_3x3(256, out_channels=64)
        self.conv_block_5x5 = self._make_conv_block_5x5(64, out_channels=out_channels)
        self.conv_block = nn.Sequential(
            self.conv_block_3x3,
            self.conv_block_3x3_2,
            self.conv_block_5x5
        )
    def _make_conv_block_3x3(self, in_channels, out_channels=256):
        """构建卷积-激活-池化模块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def _make_conv_block_5x5(self, in_channels, out_channels=256):
        """构建卷积-激活-池化模块"""
        return nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, xy, xz, yz):
        xyz = torch.cat([xy, xz, yz], dim=-1)
        branch_xyz = self.conv_block(xyz.permute(0, 3, 1, 2))
        features = branch_xyz.reshape(branch_xyz.size(0), branch_xyz.size(1), -1).permute(0, 2, 1)
        features = torch.max(features, dim=1)[0]
        return features

if "__main__" == __name__:
    # data = PointCloudDataset("data")
    # DataLoader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True)
    # for point, label in DataLoader:
    # xyz = point[:, :, 0:3].to(torch.float32)
    # features = point[:, :, 3:].to(torch.float32)
    subfolders = 'points_video'
    points_cloud_frames, sampled_points, neighbors = handle(subfolders)
    points_cloud = torch.from_numpy(points_cloud_frames)
    print(points_cloud.shape)
    points_cloud = points_cloud.unsqueeze(0).expand(3, -1, -1)
    print(points_cloud.shape)
    model = CrossPointNet(1, 16, 16, [64],  [[32, 32, 64]], 64, DFPS=True)
    model.forward(points_cloud)
    # npoint = 512
    # kernel = 32
    # xyz_new = index_points(xyz, farthest_point_sample(xyz, npoint))
    # group_idx = closest_points(kernel, xyz, xyz_new)
    # print(group_idx.shape)
    # grouped_xyz = index_points(xyz, group_idx)
    # print(grouped_xyz.shape)
    # grouped_features = index_points(features, group_idx)
    # print(grouped_features.shape)
    # grouped_features = torch.cat([grouped_features, grouped_xyz], dim=-1)
    # print(grouped_features.shape)
    # grouped_features = grouped_features.permute(0, 3, 2, 1)
    # print(grouped_features.shape)
    # in_channel = 2
    # sa1 = PointFeatureNet(512, [32, 128, 256], in_channel,
    #                     [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
    # xyz_new, new_feature = sa1(xyz, features)
    # print(xyz_new, new_feature)
    # # 选择其中一个点云进行可视化，例如第一个点云
    # point_cloud = xyz_new[0]
    # visualize_point_cloud(point_cloud)




