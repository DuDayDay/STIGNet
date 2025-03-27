import os
import sys
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_preprocess import compute_density, visualize_point_cloud

class MSRAction3D(Dataset):
    def __init__(self, root, num_points=256, frames_per_clip=16):
        self.files = []
        self.labels = []
        for video_name in os.listdir(root):
            self.files.append(os.path.join(root, video_name))
            label = int(video_name.split('_')[0][1:])
            self.labels.append(label)

        self.num_points = num_points
        self.frames_per_clip = frames_per_clip


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        index = self.labels[idx]
        file = self.files[idx]
        video = np.load(file, allow_pickle=True)['point_clouds']
        video = np.array(video, dtype=np.float32)
        all_points, density = compute_density(video)
        frame = [i for i in range(1, self.frames_per_clip + 1) for _ in range(self.num_points)]
        density = density[:, np.newaxis]
        frame = np.array(frame)[:, np.newaxis]
        points_cloud_frames = np.hstack((all_points, density, frame))
        # visualize_point_cloud(points_cloud_frames)
        return points_cloud_frames, np.array(video), index

if __name__ == '__main__':
    dataset = MSRAction3D(root='ahu',num_points=256, frames_per_clip=16)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for all_points, points_frames, label in dataloader:
        print(all_points.shape)
        print(points_frames.shape)