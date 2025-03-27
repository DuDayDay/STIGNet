import os
import time
import numpy as np
import random
import imageio
import sys
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from utils.data_preprocess import visualize_point_cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utilis'))
'''
To reduce the training time cost, the proposaled appearace raw depth are first converted into point data,
then the points data can be send to PointNet++

For multi-appearance streams, each stream is to represent one temporal segment PoinNet++

Addtionally, in training process, the appearance frames should be random sampled from all video frames.
But to save storage space, only M frames in each segment are sampled to convert to points data. one of the M frames points is final send to PointNet++.
'''

save_path1 = 'data/ntu/save'

root_path = 'data/ntu/test'

seg_num = 1  #
M = 24  #
fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347
SAMPLE_NUM = 512


def save_npy(data, save_path, filename):
    file = os.path.join(save_path, filename)
    if not os.path.isfile(file):
        np.save(file, data)

def farthest_point_sampling_fast(pc, sample_num):
    pc_num = pc.shape[0]
    sample_idx = np.zeros(shape=[sample_num, 1], dtype=np.int32)
    sample_idx[0] = np.random.randint(0, pc_num)

    cur_sample = np.tile(pc[sample_idx[0], :], (pc_num, 1))
    diff = pc - cur_sample
    min_dist = (diff * diff).sum(axis=1)
    for cur_sample_idx in range(1, sample_num):
        sample_idx[cur_sample_idx] = np.argmax(min_dist)
        # print(cur_sample_idx,np.argmax(min_dist))
        if cur_sample_idx < sample_num - 1:
            diff = pc - np.tile(pc[sample_idx[cur_sample_idx], :], (pc_num, 1))
            # print(diff.shape,min_dist[valid_idx].shape)
            min_dist = np.concatenate((min_dist.reshape(pc_num, 1), (diff * diff).sum(axis=1).reshape(pc_num, 1)),
                                      axis=1).min(axis=1)  ##?
    return sample_idx
def load_depth_from_img(depth_path):
    depth_im = imageio.v2.imread(depth_path)  # im is a numpy array
    return depth_im
def depth_to_pointcloud(depth_im):
    rows, cols = depth_im.shape
    xx, yy = np.meshgrid(range(0, cols), range(0, rows))

    valid = depth_im > 0
    xx = xx[valid]
    yy = yy[valid]
    depth_im = depth_im[valid]

    X = (xx - cx) * depth_im / fx
    Y = (yy - cy) * depth_im / fy
    Z = depth_im

    points3d = np.array([X.flatten(), Y.flatten(), Z.flatten()])

    return points3d
def compute_density(all_points, radius=0.5):
    tree = KDTree(all_points)
    density = tree.query_radius(all_points, r=radius, count_only=True)
    density_normalized = density / np.max(density)
    return density_normalized

start_time = time.time()

sub_Files = os.listdir(root_path)
sub_Files.sort()
for s_fileName in sub_Files:

    videoPath = os.path.join(root_path, s_fileName, 'nturgb+d_depth_masked')
    if os.path.isdir(videoPath):
        print(s_fileName, '-------')
        video_Files = os.listdir(videoPath)
        # print(video_Files)
        video_Files.sort()
        for video_FileName in video_Files:
            pngPath = os.path.join(videoPath, video_FileName)
            imgNames = os.listdir(pngPath)
            imgNames.sort()
            frame_indices = np.linspace(0, len(imgNames) - 1, M).astype(int)
            points_frames = []
            points_dist_frames = []
            for frame_index in frame_indices:
                imgPath = os.path.join(pngPath, imgNames[frame_index])
                png = load_depth_from_img(imgPath)
                raw_points_xyz = depth_to_pointcloud(png)
                points_num = raw_points_xyz.shape[1]
                all_sam = np.arange(points_num)
                if points_num < SAMPLE_NUM:
                    if points_num < SAMPLE_NUM / 2:
                        index = random.sample(list(all_sam), SAMPLE_NUM - points_num - points_num)
                        index.extend(list(all_sam))
                        index.extend(list(all_sam))
                    else:
                        index = random.sample(list(all_sam), SAMPLE_NUM - points_num)
                        index.extend(list(all_sam))
                else:
                    index = random.sample(list(all_sam), SAMPLE_NUM)
                points = raw_points_xyz[:, index].T
                points = points/1000
                np.random.shuffle(points)
                density = compute_density(points)
                density = density[:, np.newaxis]
                points_d = np.hstack((points, density))
                # distances = cdist(points, points)
                # # 对每个点按距离排序，返回排序后的索引
                # sorted_indices = np.argsort(distances, axis=1)
                # visualize_point_cloud(points_d)
                points_frames.append(points_d)
                # points_dist_frames.append(sorted_indices)

            all_points = np.vstack(points_frames)
            points_frames = np.array(points_frames)
            points_dist_frames = np.array(points_dist_frames)
            all_points = all_points[:, 0:3]
            frame = [i for i in range(1, M + 1) for _ in range(SAMPLE_NUM)]
            frame = np.array(frame)[:, np.newaxis]
            density = compute_density(all_points)
            density = density[:, np.newaxis]
            all_points = np.hstack((all_points, density, frame))
            # distances = cdist(all_points, all_points)
            # # 对每个点按距离排序，返回排序后的索引
            # sorted_indices_all = np.argsort(distances, axis=1)
            # visualize_point_cloud(all_points)
            filename_frames_points = 'points_frames_' + str(SAMPLE_NUM) + '_' + str(M)
            filename_all_points = 'all_points_' + str(SAMPLE_NUM) + '_' + str(M)
            # filename_frames_points_dist = 'points_frames_dist_' + str(SAMPLE_NUM) + '_' + str(M)
            # filename_all_points_dist = 'all_points_dist_' + str(SAMPLE_NUM) + '_' + str(M)
            save_path = save_path1 + '/' + video_FileName
            print(video_FileName)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_npy(points_frames, save_path, filename_frames_points)
            save_npy(all_points, save_path, filename_all_points)
            # save_npy(points_dist_frames, save_path, filename_frames_points_dist)
end_time = time.time()
print(end_time - start_time)
