


import numpy as np

import glob, os
import struct
import open3d as o3d

# https://github.com/PRBonn/semantic-kitti-api/blob/master/visualize_voxels.py
def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1

  return uncompressed


# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/16/velodyne/000000.bin"
binFileName = "/Volumes/Extreme SSD/semKitti/dataset/sequences/00/voxels/000000.bin"

voxelGrid = np.fromfile(binFileName, dtype=np.int16)
# pcdi = pcdi.reshape((int(np.shape(pcdi)[0]) // 4, 4))
# pcdi = np.delete(pcdi, np.s_[2::], 1)
# pcdi = pcdi.reshape(np.size(pcdi),)
# pcdi = np.pad(pcdi, (0,1035136 - int(np.shape(pcdi)[0])), 'constant', constant_values=(0))

# pcdi = pcdi.reshape(int(np.shape(pcdi)[0]) // 2, 2)
# new_col = pcdi.sum(1)[...,None]

# all_data = np.hstack((pcdi, new_col))

# print(np.shape(all_data))


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(all_data)
# o3d.visualization.draw_geometries([pcd])

print(np.shape(voxelGrid))
print(voxelGrid)
cat_indices = np.where(voxelGrid == 1)
print(cat_indices)
print(np.shape(cat_indices))

buffer_data = unpack(np.fromfile(binFileName, dtype=np.uint8)).astype(np.float32)
print(np.shape(buffer_data))
print(buffer_data)

# pcd = o3d.geometry.PointCloud(); 
# pcd.points = o3d.utility.Vector3dVector(cat_indices)