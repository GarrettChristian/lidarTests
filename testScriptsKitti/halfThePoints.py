"""
"""

import numpy as np

import glob, os
import struct
import open3d as o3d


binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
np_arr = np.fromfile(binFileName, dtype=np.float32)
print(np.shape(np_arr))

np_arr = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
print(np.shape(np_arr))
np_arr = np.delete(np_arr, 3, 1)
print(np.shape(np_arr))
half = (int(np.shape(np_arr)[0]) // 2)
np_arr = np.delete(np_arr, np.s_[half::], 0)

# mask = (np_arr[:, 0] < 0) # X AXIS CUT
# mask = (np_arr[:, 1] < 0) # Y AXIS CUT
# mask = (np_arr[:, 2] < 0) # Z AXIS CUT
# np_arr = np_arr[mask, :]

print(np.shape(np_arr))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np_arr)
o3d.visualization.draw_geometries([pcd])
