import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
from utils.deepImage import load_pointcloud


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
import os

# class MSRAction(Dataset):
#     def __init__(self, root, frames_per_clip=16, frame_interval=2, num_points=256, train=False):
#         super(MSRAction, self).__init__()
#
#         self.videos = []
#         self.labels = []
#         self.index_map = []
#         index = 0
#         for video_name in os.listdir(root):
#             if train and (int(video_name.split('_')[1].split('s')[1]) <= 5):
#                 video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
#                 self.videos.append(video)
#                 label = int(video_name.split('_')[0][1:])-1
#                 self.labels.append(label)
#
#                 nframes = video.shape[0]
#                 for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
#                     self.index_map.append((index, t))
#                 index += 1
#
#             if not train and (int(video_name.split('_')[1].split('s')[1]) > 5):
#                 video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
#                 self.videos.append(video)
#                 label = int(video_name.split('_')[0][1:])-1
#                 self.labels.append(label)
#
#                 nframes = video.shape[0]
#                 for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
#                     self.index_map.append((index, t))
#                 index += 1
#
#         self.frames_per_clip = frames_per_clip
#         self.frame_interval = frame_interval
#         self.num_points = num_points
#         self.train = train
#         self.num_classes = max(self.labels) + 1
#
#
#     def __len__(self):
#         return len(self.index_map)
#
#     def __getitem__(self, idx):
#         index, t = self.index_map[idx]
#
#         video = self.videos[index]
#         label = self.labels[index]
#
#         clip = [video[t+i*self.frame_interval] for i in range(self.frames_per_clip)]
#         for i, p in enumerate(clip):
#             if p.shape[0] > self.num_points:
#                 r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
#             else:
#                 repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
#                 r = np.random.choice(p.shape[0], size=residue, replace=False)
#                 r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
#             clip[i] = p[r, :]
#         clip = np.array(clip)
#
#         if self.train:
#             # scale the points
#             scales = np.random.uniform(0.9, 1.1, size=3)
#             clip = clip * scales
#
#         clip = clip / 300
#         all_points, density = compute_density(clip, radius=0.5)
#         frame = [i for i in range(1, self.frames_per_clip + 1) for _ in range(self.num_points)]
#         density = density[:, np.newaxis]
#         frame = np.array(frame)[:, np.newaxis]
#         points_cloud_frames = np.hstack((all_points, density, frame))
#         # visualize_point_cloud(points_cloud_frames)
#         return points_cloud_frames,clip.astype(np.float32),  label

if __name__ == '__main__':
    dataset = MSRAction(root='msr_datasets/msr', frames_per_clip=16,train=False)
    clip, label, video_idx = dataset[160]
    print(clip)
    print(label)
    print(video_idx)
    print(dataset.num_classes)
