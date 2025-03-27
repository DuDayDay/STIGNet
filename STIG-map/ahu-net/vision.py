from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
import torch
from datasets_ahu import MSRAction3D
from torch.utils.data import DataLoader
from model import CrossModule

def get_xyz_and_features(model, data_loader):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        xyz_new_list = []
        features_list = []
        points_list = []
        label_list = []
        for points, points_frames, labels in data_loader:  # 假设数据集返回 (points, labels)
            xyz_new, features = model.CP1(points, points_frames)
            # 使用模型进行推理
            label_list.append(labels)
            points_list.append(points)  # 原始点云
            xyz_new_list.append(xyz_new)  # 采样后的点云
            features_list.append(features)  # 特征
        # 合并所有batch的数据
        points = torch.cat(points_list, dim=0)  # (B * N, 3)
        xyz_new = torch.cat(xyz_new_list, dim=0)  # (B * N, 3)
        features = torch.cat(features_list, dim=0)  # (B * N, C)
        labels = torch.cat(label_list)
    return points, xyz_new, features, labels

def visualize_point_cloud_with_features(points, xyz_new, features,labels, rows=2, cols=2):
    B, N, _ = xyz_new.shape
    _, _, C = features.shape

    # 创建多个子图
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Batch {i+1}" for i in range(B)],
        vertical_spacing=0.1, horizontal_spacing=0.05,
        specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)]  # 指定为3D子图
    )

    # 对每个batch进行可视化
    for batch_idx in range(B):
        # 使用原始点云数据 (points) 和降维特征数据
        original_xyz = points[batch_idx].cpu().numpy()  # (N, 3)
        xyz = xyz_new[batch_idx].cpu().numpy()  # (N, 3)
        feature = features[batch_idx].cpu().numpy()  # (N, C)
        label = labels[batch_idx].cpu().numpy()
        color = feature[:, 254]
        # 将降维后的特征作为颜色
        # 增强颜色对比度
        color_min = np.min(color)
        color_max = np.max(color)
        color_range = color_max - color_min
        color = (color - color_min) / color_range  # 将颜色标准化到 [0, 1] 范围内
        # 1. 绘制原始点云
        original_trace = go.Scatter3d(
            x=original_xyz[:, 0], y=original_xyz[:, 1], z=original_xyz[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='lightgrey',  # 使用灰色表示原始点云
                opacity=0.6
            ),
            name=f"Original Batch {batch_idx+1}"
        )

        # 2. 绘制特征点云（降维后的特征作为颜色）
        feature_trace = go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=color,  # 使用降维后的特征作为颜色值
                # colorscale='',  # 颜色映射
            ),
            name=f"Feature Batch {batch_idx+1},{label}"
        )

        row = (batch_idx // cols) + 1
        col = (batch_idx % cols) + 1

        # 添加原始点云和特征点云到同一个子图
        fig.add_trace(original_trace, row=row, col=col)
        fig.add_trace(feature_trace, row=row, col=col)


    fig.update_layout(
        title="Point Cloud and Feature Visualization",
        showlegend=True,
        autosize=True,  # 自适应大小
        height=1000,  # 或者设置为更大的值
        width=1500,   # 或者设置为更大的值
        margin=dict(l=0, r=0, t=40, b=0)  # 移除边距，最大化利用空间
    )

    fig.show()

if __name__ == "__main__":
    batch_size = 8
    num_points = 2048  # 假设你需要的点云大小
    data_path = 'data/ahu-origin/test/test2'  # 替换为实际的路径
    num_class = 20  # 假设你有40个类别
    # 初始化数据集
    test_dataset = MSRAction3D(root=data_path)
    testDataLoader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    # 加载模型
    model = CrossModule(num_class, 2)  # 加载模型模块
    checkpoint = torch.load('log/classification/2025-03-26_22-12/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    points, xyz_new, features, labels = get_xyz_and_features(model, testDataLoader)
    # 可视化
    visualize_point_cloud_with_features(points, xyz_new, features, labels)