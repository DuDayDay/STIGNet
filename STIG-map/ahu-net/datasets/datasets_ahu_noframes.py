import os
import sys
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MSRAction3D(Dataset):
    def __init__(self, root,points_frames=16):
        self.files = []
        self.labels = []
        self.points_frames = points_frames
        for video_name in os.listdir(root):
            self.files.append(os.path.join(root, video_name))
            label = int(video_name.split('_')[0][1:])
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        index = self.labels[idx]
        file = self.files[idx]
        points_cloud_frames = np.load(file, allow_pickle=True)['point_clouds']
        # visualize_point_cloud(points_cloud_frames)
        return points_cloud_frames, self.points_frames, index

if __name__ == '__main__':
    dataset = MSRAction3D(root='data/ahu-origin/save_cls',points_frames=16)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for all_points, points_frames, label in dataloader:
        print(all_points.shape)
        print(points_frames.shape)