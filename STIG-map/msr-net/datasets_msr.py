import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
from utils.deepImage import load_pointcloud
import os

def extract_group_number(filename):
    """
    从文件名中提取出 'a' 后面的数字。

    :param filename: 文件名，形如 'a01_s01_e01_sdepth.bin'
    :return: 提取的数字
    """
    # 使用正则表达式匹配 'a' 后面的数字部分
    pattern = r'a(\d{1,2})'  # 匹配 'a' 后面跟着 1 到 2 位数字
    match = re.search(pattern, filename)
    camera = re.search(r's(\d+)', filename).group(1)
    if match:
        return int(match.group(1)), int(camera)  # 提取并返回数字部分，转换为整数
    else:
        return None

# 数据集类
class MSRAction(Dataset):
    def __init__(self, root,  frame_clip=16, num_points=128, is_test=False):
        self.frame_clip = frame_clip
        self.is_test = is_test
        self.num_points = num_points
        self.labels = []
        self.root = root
        rootdir = os.listdir(root)
        filenames_prefix = [extract_group_number(filename) for filename in rootdir]

        if is_test is False:
            self.files = [f for f, (label, camera) in zip(rootdir, filenames_prefix) if camera <= 5]
            self.labels = [label for label, camera in filenames_prefix if camera <= 5]
        else:
            self.files = [f for f, (label, camera) in zip(rootdir, filenames_prefix) if camera > 5]
            self.labels = [label for label, camera in filenames_prefix if camera > 5]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx] - 1
        filename = self.files[idx]
        file = os.path.join(self.root, filename)
        points_clouds, points_frames = load_pointcloud(file, self.frame_clip, self.num_points, self.is_test)

        return points_clouds, self.frame_clip, label
if __name__ == '__main__':
    start = time.time()
    data = MSRAction(root='msr_datasets/Depth', num_points=300, is_test=True)
    end = time.time()
    dataloader = DataLoader(data, batch_size=8, shuffle=True, num_workers=4)
    for point, frames, label in dataloader:
        print(point.shape)
        print(label.shape)

