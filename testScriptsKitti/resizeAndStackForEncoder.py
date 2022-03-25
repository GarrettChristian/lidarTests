"""
"""

import numpy as np

import glob, os
import struct
import open3d as o3d



trainLidar = np.array([])

path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/"
for file in glob.glob(path + "**/0000*.bin", recursive = True):
    print(file)
    pcd = np.fromfile(file, dtype=np.float32)
    # print(np.shape(pcd))

    # pcd = pcd.reshape((int(np.shape(pcd)[0]) // 4, 4))
    pcd = np.pad(pcd, (0,2070272 - int(np.shape(pcd)[0])), 'constant', constant_values=(0))
    # print(np.shape(pcd))
    # print(pcd)

    if (np.size(trainLidar)):
        trainLidar = np.vstack((trainLidar, pcd))
    else:
        trainLidar = np.array([pcd])

