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
# pcd_arr = np.delete(pcd_arr, 3, 1)
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


# Only cars
mask = (semantics == 10)
cars = pcd_arr[mask, :]
# cars removed
mask = (semantics != 10)
noCars = pcd_arr[mask, :]
# move cars z up by 10
cars[::3] = cars[::3] + 2 

# Rejoin cars
pcd_arr = np.vstack((noCars, cars))

print(np.shape(pcd_arr))

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pcd_arr)
# o3d.visualization.draw_geometries([pcd])

pcd_arr = pcd_arr.flatten()
print(np.shape(pcd_arr))
pcd_arr.tofile("carsUp2Z00.bin")