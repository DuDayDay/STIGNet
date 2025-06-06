import time

import torch
import importlib
from tqdm import tqdm
import numpy as np
from dataset_ntu import NtuDataset
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from model import CrossModule


def plot_confusion_matrix(y_true, y_pred, labels, normalize=False):
    """
    绘制混淆矩阵的热力图
    Args:
        y_true: 实际类别
        y_pred: 预测类别
        labels: 类别标签列表
        normalize: 是否对混淆矩阵进行归一化
    """
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1)
        cm = np.divide(cm.astype('float'), row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)
        cm = np.nan_to_num(cm)  # 将可能存在的 NaN 替换为 0

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=False, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    norm = ax.collections[0].norm
    for text in ax.texts:
        value = float(text.get_text())
        text.set_color("white" if norm(value > 0.5) else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    data_path ='ntu/ntu-60'
    batch_size = 32
    # test_dataset = MSRAction(root=data_path, frame_clip=16, num_points=256, is_test=False)
    # # test_dataset = PointCloudDataset(root=data_path, split='test')
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False,
    #                                              num_workers=4)
    train_dataset = train_dataset = NtuDataset(root=data_path, is_test=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    # num_class = 4  # 分类数量
    model = CrossModule(60, 2)  # 加载模型模块
    # classifier = model.CrossModule(20, 2)
    # classifier = model.ClsModel(num_class, 2)  # 初始化模型
    checkpoint = torch.load('log/classification/2025-03-09_94.29/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cuda')
    with torch.no_grad():
        start = time.time()
        mean_correct = []
        # classifier = model.eval()

        # total_params = sum(p.numel() for p in model.parameters())
        # total_params_m = total_params / 1e6  # 转换为 M
        # print(f"Total parameters: {total_params_m:.2f}M")
        class_acc = np.zeros((60, 3))

        # 计算 FLOPs

        # 用于保存所有测试样本的预测类别
        all_predictions = []
        all_targets = []

        for j, (points, points_frames, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # torch.onnx.export(model, points, points_frames, "model.onnx")
            classifier = model.eval()
            points, points_frames, target = points.cuda(), points_frames.cuda(), target.cuda()
            # flops, _ = profile(model, inputs=(points, points_frames))

            # flops_g = flops / 1e9  # 转换为 G
            # print(f"FLOPs: {flops_g:.2f}G")
            # pred = classifier(points[:, :, 0:3], points[:, :, 3:])
            pred = classifier(points, points_frames)
            pred_choice = pred.data.max(1)[1]  # 获取预测的类别索引

            # 记录所有预测类别和真实类别
            all_predictions.append(pred_choice.cpu().numpy())  # 转为 numpy 并存储
            all_targets.append(target.cpu().numpy())

            # 计算类别准确率
            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1

            # 计算批量实例准确率
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        # 计算类别准确率
        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2])
        instance_acc = np.mean(mean_correct)

        # 将所有预测结果和真实标签展平
        all_predictions = np.concatenate(all_predictions, axis=0)  # 所有预测类别
        all_targets = np.concatenate(all_targets, axis=0)
        end = time.time()
        print(str(end-start))
        print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        print(f'predictions: {all_predictions}, targets: {all_targets}')

        # 获取类别标签
        class_labels = [str(i) for i in range(60)]  # 假设类别为 0-19
        # 绘制混淆矩阵
        plot_confusion_matrix(all_targets, all_predictions, labels=class_labels)
if __name__ == '__main__':
    main()