import datetime
from pathlib import Path
import torch
import importlib
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import os
import numpy as np
from dataset_ntu import NtuDataset
from thop import profile
# from provider import *
from torch.utils.data import random_split
from provider import rotate_and_scale_point_cloud_x_axis
import sys

import os



def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
def lr_lambda(epoch):
    if epoch < 25:
        return 0.7 ** epoch
    elif 25 <= epoch < 45:
        return (0.7 ** 25) * (0.95 ** (epoch - 25))
    else:
        return (0.7 ** 25) * (0.95 ** 20) * (0.99 ** (epoch - 45))

def parse_args():
    parser = argparse.ArgumentParser(description="Train a point cloud classification model.")
    # 添加参数
    parser.add_argument('--data_path', type=str, default='ntu/ntu-60-512', help='Path to the dataset')
    parser.add_argument('--lr_warmup_epochs', type=int, default=10, help='Path to the dataset')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Path to the dataset')
    parser.add_argument('--lr_milestones', nargs='+', default=[20, 30], help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training and testing')
    parser.add_argument('--num_class', type=int, default=60, help='Number of classification categories')
    parser.add_argument('--feature_num', type=int, default=2, help='Number of input features')
    parser.add_argument('--epochs', type=int, default=450
                        , help='Number of training epochs')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of point clouds')
    parser.add_argument('--learning_rate', type=float, default=0.001
                        , help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for the optimizer')
    parser.add_argument('--pretrained', type=str, default='log/classification/2025-03-18_22-56/checkpoints/best_model.pth', help='Path to a pre-trained model (optional)')
    parser.add_argument('--save_dir', type=str, default='./log/classification',
                        help='Directory to save logs and checkpoints')

    return parser.parse_args()
def test(model, loader, criterion, num_class=4):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    total_loss = 0  # 新增变量记录总损失
    total_batches = 0

    for j, (points, points_frames,target)in tqdm(enumerate(loader), total=len(loader)):

        points, points_frames, target = points.cuda(), points_frames.cuda(), target.cuda()
        # pred = classifier(points[:, :, 0:3], points[:, :, 3:])
        pred = classifier(points,points_frames)
        loss = criterion(pred, target.long())  # 计算损失
        total_loss += loss.item()
        total_batches += 1

        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    avg_loss = total_loss / total_batches  # 计算平均损失

    return instance_acc, class_acc, avg_loss

def train():
    args = parse_args()

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 手动指定 GPU
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    '''CREATE DIR'''
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    exp_dir = Path(args.save_dir) / timestr
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

    '''DATA LOADING'''
    data_path = args.data_path
    train_dataset = NtuDataset(root=data_path,is_test=False)
    test_dataset = NtuDataset(root=data_path, is_test=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    '''MODEL LOADING'''
    num_class = args.num_class
    feature_num = args.feature_num
    epochs = args.epochs

    model = importlib.import_module('model')  # 加载模型模块
    classifier = model.CrossModule(num_class, feature_num)  # 初始化模型
    criterion = model.get_loss()
    classifier.apply(inplace_relu)  # 初始化 ReLU
    classifier = classifier.cuda()
    criterion = criterion.cuda()
    tatal_paramater = sum(p.numel() for p in classifier.parameters())
    tatal_paramater_m =  tatal_paramater/1e6
    print(tatal_paramater_m)
    if args.pretrained:
        try:
            checkpoint = torch.load(args.pretrained)
            # print(checkpoint['epoch'])
            # start_epoch = checkpoint['epoch']
            start_epoch = 0
            classifier.load_state_dict(checkpoint['model_state_dict'])
            print(checkpoint['instance_acc'])
            print("Using pre-trained model...")
        except Exception as e:
            print(f"Failed to load pre-trained model: {e}")
            print("Training from scratch...")
            start_epoch = 0
    else:
        print("No pre-trained model found. Training from scratch...")
        start_epoch = 0

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_epoch = 0
    best_instance_acc = 0.0

    # Metrics to track
    train_instance_acc_list = []
    train_loss_list = []
    test_instance_acc_list = []
    test_loss_list = []

    '''TRAINING'''
    print("Start training...")
    for epoch in range(start_epoch, epochs):
        classifier.train()

        mean_correct = []
        loss_correct = []
        for batch_id, (points, points_frames,target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            # points = points.data.numpy()
            # points_frames = points_frames.numpy()
            # points , points_frames = rotate_and_scale_point_cloud_x_axis(points, points_frames)
            # # points = random_point_dropout(points)
            # # points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
            # points = torch.Tensor(points)
            # points_frames = torch.Tensor(points_frames)
            points, points_frames, target = points.cuda(), points_frames.cuda(), target.cuda()
            flops, _ = profile(classifier, inputs=(points, points_frames))
            flops_g = flops/1e9
            print(f"FLOPS:{flops_g:.2f}G")
            # pred = classifier(points[:, :, 0:3], points[:, :, 3:])
            pred = classifier(points, points_frames)
            loss = criterion(pred, target.long())
            loss_correct.append(loss.item())
            pred_choice = pred.data.max(1)[1
            ]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / points.size(0))

            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        loss_train = np.mean(loss_correct)
        train_instance_acc_list.append(train_instance_acc)
        train_loss_list.append(loss_train)
        print(f'Epoch {epoch + 1}, Train Instance Accuracy: {train_instance_acc:.4f}, Train Loss: {loss_train:.4f}')

        with torch.no_grad():
            # 调用 test 函数，获取测试准确率和损失
            instance_acc, _, test_loss = test(classifier.eval(), test_loader, criterion, num_class=num_class)
            test_instance_acc_list.append(instance_acc)
            test_loss_list.append(test_loss)
            print(f'Test Instance Accuracy: {instance_acc:.4f}, Test Loss: {test_loss:.4f}')

            if instance_acc > best_instance_acc:
                best_instance_acc = instance_acc
                savepath = checkpoints_dir / 'best_model.pth'

                print(f'Saving best model to {savepath}')
                state = {
                    'epoch': epoch + 1,
                    'instance_acc': instance_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
        global_epoch += 1
        scheduler.step()

    print("Training completed.")

    # Plot metrics
    plt.figure(figsize=(12, 6))
    plt.plot(train_instance_acc_list, label='Train Instance Accuracy', color='b', linestyle='-', marker='.')
    plt.plot(train_loss_list, label='Train Loss', color='r', linestyle='--', marker='x')
    plt.plot(test_instance_acc_list, label='Test Instance Accuracy', color='g', linestyle='-', marker='o')
    plt.plot(test_loss_list, label='Test Loss', color='purple', linestyle='--', marker='s')

    plt.title('Training and Testing Metrics Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Metrics', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig(exp_dir / 'training_metrics.png')
    plt.show()

if __name__ == '__main__':
    train()