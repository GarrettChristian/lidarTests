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
# mask = (semantics == 10)
# cars = pcd_arr[mask, :]
# print(np.shape(cars))

# cars removed
mask = (semantics != 40)
noCars = pcd_arr[mask, :]

# move cars z up by 10
# cars[:, 2] = cars[:, 2] + 1
# print(np.shape(cars))

# Rejoin cars
# pcd_arr = np.vstack((noCars, cars))
# pcd_arr = cars

print(np.shape(noCars))

# np_arr = noCars.reshape((int(np.shape(noCars)[0]) // 4, 4))
np_arr = np.delete(noCars, 3, 1)


mask = ((np_arr[:, 0] < 25.6) & (np_arr[:, 0] > -25.6) 
    & (np_arr[:, 1] < 25.6) & (np_arr[:, 1] > -25.6) 
    & (np_arr[:, 2] < 3.4) & (np_arr[:, 2] > -3.4))


np_arr = np_arr[mask, :]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np_arr)

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.2)

voxGrid = []
for voxel in voxel_grid.get_voxels():

    x = voxel.grid_index[0]
    y = voxel.grid_index[1]
    z = voxel.grid_index[2]

    if (x < 256 and y < 256 and z < 32):
        voxGrid.append(x)
        voxGrid.append(y)
        voxGrid.append(z)

np_vox2 = np.asarray(voxGrid)
np_vox2 = np_vox2.reshape((int(np.shape(np_vox2)[0]) // 3, 3))

pcd.points = o3d.utility.Vector3dVector(np_vox2)
o3d.visualization.draw_geometries([pcd])

np_vox2 = np_vox2.flatten()
print(np.shape(np_vox2))
np_vox2 = np_vox2.astype(np.ubyte)
np_vox2.tofile("noRoad00Vox2.bin")

# -------





# ogVoxArray = []
# for x in range (256):
#     for y in range (256):
#           if (np_vox2[x][y][0] > 0.5):
#                 ogVoxArray.append(x)
#                 ogVoxArray.append(y)
#                 ogVoxArray.append(0)


# np_vox2og = np.asarray(ogVoxArray)
# np_vox2og = np_vox2og.reshape((int(np.shape(np_vox2og)[0]) // 3, 3))

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np_vox2og)
# o3d.visualization.draw_geometries([pcd])