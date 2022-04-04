


import numpy as np

import glob, os
import struct
import open3d as o3d

binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/16/velodyne/000000.bin"

pcdi = np.fromfile(binFileName, dtype=np.float32)
pcdi = pcdi.reshape((int(np.shape(pcdi)[0]) // 4, 4))
pcdi = np.delete(pcdi, np.s_[2::], 1)
pcdi = pcdi.reshape(np.size(pcdi),)
pcdi = np.pad(pcdi, (0,1035136 - int(np.shape(pcdi)[0])), 'constant', constant_values=(0))


pcdi = pcdi.reshape(int(np.shape(pcdi)[0]) // 2, 2)
new_col = pcdi.sum(1)[...,None]

all_data = np.hstack((pcdi, new_col))

print(np.shape(all_data))


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_data)
o3d.visualization.draw_geometries([pcd])