"""
"""

import numpy as np

import glob, os
import struct
import open3d as o3d


binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
rawSyncFile = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/2011_09_26 2/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin"
rawUnsyncFile = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/2011_09_26 2unsync/2011_09_26_drive_0001_extract/velodyne_points/data/0000000000.txt"
rawUnsyncFile2 = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/2011_09_26 2unsync/2011_09_26_drive_0001_extract/velodyne_points/data/0000000001.txt"

np_arr = np.fromfile(rawSyncFile, dtype=np.float32)
print(np.shape(np_arr))

np_arr = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
print(np.shape(np_arr))
np_arr = np.delete(np_arr, 3, 1)
print(np.shape(np_arr))
# half = (int(np.shape(np_arr)[0]) // 2)
# np_arr = np.delete(np_arr, np.s_[half::], 0)

# mask = (np_arr[:, 0] < 0) # X AXIS CUT
# mask = (np_arr[:, 1] < 0) # Y AXIS CUT
# mask = (np_arr[:, 2] < 0) # Z AXIS CUT
# np_arr = np_arr[mask, :]

print(np.shape(np_arr))

with open(rawUnsyncFile) as file:
    lines = file.readlines()

print(lines[0])
print(len(lines))
print(np_arr[0])

with open(rawUnsyncFile2) as file:
    lines2 = file.readlines()

print(lines2[0])
print(len(lines2))

unsyncData = []
for line in lines:
    for xyzi in line.split():
        # print(xyzi)
        unsyncData.append(xyzi)

print(len(unsyncData))
unsyncDataStr = np.array(unsyncData)
print(np.shape(unsyncDataStr))
unsyncDataFloat = np.asarray(unsyncDataStr, dtype = np.float32)
unsyncDataFloat = unsyncDataFloat.reshape((int(np.shape(unsyncDataFloat)[0]) // 4, 4))
print(np.shape(unsyncDataFloat))
unsyncDataFloat = np.delete(unsyncDataFloat, 3, 1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(unsyncDataFloat)
o3d.visualization.draw_geometries([pcd])
