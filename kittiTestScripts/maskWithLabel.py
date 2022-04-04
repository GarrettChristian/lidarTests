"""
"""

import numpy as np

import glob, os
import struct
import open3d as o3d


labelsFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/labels/000000.label"
binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/velodyne/000000.bin"
pcd_arr = np.fromfile(binFileName, dtype=np.float32)
print(np.shape(pcd_arr))

pcd_arr = pcd_arr.reshape((int(np.shape(pcd_arr)[0]) // 4, 4))
print(np.shape(pcd_arr))
pcd_arr = np.delete(pcd_arr, 3, 1)
print(np.shape(pcd_arr))

label_arr = np.fromfile(labelsFileName, dtype=np.int32)

print(np.shape(label_arr))

# upper_half = label >> 16      # get upper half for instances
# lower_half = label & 0xFFFF   # get lower half for semantics
# 0xFFFF = 65535

# lowerAnd = np.full(np.shape(label_arr), 65535)
# semantics = np.bitwise_and(label_arr, 65535)
semantics = label_arr & 0xFFFF
labelInstance = label_arr >> 16 
print(semantics)
print(labelInstance)

# carMask = ma.masked_equal(semantics, 10)
# carMask = np.ma.masked_where(semantics == 10, semantics)
# print(carMask)
# row_mask = (semantics == 10).all(axis=1)
# new_pcd_arr = np.ma.masked_where(np.ma.getmask(carMask), pcd_arr)

mask = (semantics == 10)
pcd_arr = pcd_arr[mask, :]


# half = (int(np.shape(pcd_arr)[0]) // 2)
# pcd_arr = np.delete(pcd_arr, np.s_[half::], 0)

# mask = (pcd_arr[:, 0] < 0) # X AXIS CUT
# mask = (pcd_arr[:, 1] < 0) # Y AXIS CUT
# # mask = (pcd_arr[:, 2] < 0) # Z AXIS CUT
# pcd_arr = pcd_arr[mask, :]

# print(np.shape(pcd_arr))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_arr)
o3d.visualization.draw_geometries([pcd])

# pcd_arr = pcd_arr.flatten()
# print(np.shape(pcd_arr))
# pcd_arr.tofile("removeHalfByYval.bin")