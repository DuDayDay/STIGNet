import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import PointFeatureNet, PointNetSetAbstraction, CrossPointNet, pixelize_point_cloud_batch, pixelize, \
    visualize
import time
import os
import sys
from provider import rotate_and_scale_point_cloud_x_axis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
class CrossModule(nn.Module):
    def __init__(self, num_class, in_channels, anchors=16, out_channels=2):
        super(CrossModule, self).__init__()
        self.num_class = num_class
        #  DFPS = True, attention=True
        anchors = int(anchors)
        self.CP1 = CrossPointNet(in_channels, anchors, int(512/anchors), [36, 64],  [[32, 32, 64], [32, 32, 64]], int(1024/anchors), out_channels, DFPS=True, time_encoding=False)
        """
         in_channels = out_channels + n_per_sample
        """
        #  attention = True
        self.fe2 = PointFeatureNet(64, [56, 128], 256+out_channels, [[64, 64, 128], [128, 128, 256]])
        # self.fe2 = PointFeatureNet(128, [256], 192+64, [[128, 128, 256]])
        self.fe3 = PointNetSetAbstraction(384 + 3, [256, 1024])
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.drop1 = nn.Dropout(0.6)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, points, points_frames):
        xyz_new, features = self.CP1(points,points_frames)
        B, _, _ = xyz_new.shape
        f_xyz2, f_features2 = self.fe2(xyz_new, features)
        f_xyz3, f_features3 = self.fe3(f_xyz2, f_features2)
        x = f_features3.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss
class get_loss_MultiCls(nn.Module):
    def __init__(self, class_counts):
        super(get_loss_MultiCls, self).__init__()
        # 计算正样本权重
        pos_weight = 1 / (class_counts + 1e-6)  # 防止除零，使用小常数
        # 将 pos_weight 转换为 torch.Tensor
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32).cuda()  # 放到 GPU 上

        # 创建损失函数，使用 pos_weight
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, pred, target):
        total_loss = self.loss_fn(pred, target)
        return total_loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): 类别权重参数，默认值是 0.25，通常用于调节正负样本的影响。
            gamma (float): Focal Loss 的调节因子，默认值是 2.0，用于控制难易样本的权重。
            reduction (str): 损失归约方式，'mean' 或 'sum'。默认为 'mean'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): 模型的输出，未经过 softmax 的 raw logits，形状为 [batch_size, num_classes]。
            targets (Tensor): 目标标签，形状为 [batch_size, num_classes]，通常是 one-hot 编码。
        """
        # 使用 softmax 获得概率分布
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 计算预测的概率
        p_t = torch.exp(-BCE_loss)  # p_t 是正样本的概率

        # 计算 Focal Loss 的加权
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss




