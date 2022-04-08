"""
"""

import numpy as np
import open3d as o3d
import sys


binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
np_arr = np.fromfile(binFileName, dtype=np.float32)
print(np.shape(np_arr))

np_arr = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
print(np.shape(np_arr))
np_arr = np.delete(np_arr, 3, 1)
print(np.shape(np_arr))
# half = (int(np.shape(np_arr)[0]) // 2)
# np_arr = np.delete(np_arr, np.s_[half::], 0)

"""
25.6 to each side, 3.4 up and 3.4 down
based off of
We select a volume of 51.2 m ahead of the car, 25.6 m
to every side and 6.4 m in height with a voxel resolution of
0.2 m, which results in a volume of 256×256×32 voxels to
predict.
"""
mask = ((np_arr[:, 0] < 25.6) & (np_arr[:, 0] > -25.6) 
    & (np_arr[:, 1] < 25.6) & (np_arr[:, 1] > -25.6) 
    & (np_arr[:, 2] < 3.4) & (np_arr[:, 2] > -3.4))




np_arr = np_arr[mask, :]

print(np.shape(np_arr))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np_arr)
# o3d.visualization.draw_geometries([pcd])

# np_arr = np_arr.flatten()
# print(np.shape(np_arr))
# np_arr.tofile("removeHalfByYval.bin")

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.2)


print(len(voxel_grid.get_voxels()))

minX4 = sys.maxsize
maxX4 = sys.maxsize * -1
minY4 = sys.maxsize
maxY4 = sys.maxsize * -1
minZ4 = sys.maxsize
maxZ4 = sys.maxsize * -1

for voxel in voxel_grid.get_voxels():
    maxX4 = max(maxX4, voxel.grid_index[0])
    minX4 = min(minX4, voxel.grid_index[0])
    maxY4 = max(maxY4, voxel.grid_index[1])
    minY4 = min(minY4, voxel.grid_index[1])
    maxZ4 = max(maxZ4, voxel.grid_index[2])
    minZ4 = min(minZ4, voxel.grid_index[2])

print("min X 4: ", minX4)
print("max X 4: ", maxX4)
print("min Y 4: ", minY4)
print("max Y 4: ", maxY4)
print("min Z 4: ", minZ4)
print("max Z 4: ", maxZ4)


grid = np.zeros((256, 256, 32))

dropped = 0


for voxel in voxel_grid.get_voxels():
    x = voxel.grid_index[0]
    y = voxel.grid_index[1]
    z = voxel.grid_index[2]

    if (x < 256 and y < 256 and z < 32):
        grid[x][y][z] = 1
    else:
        dropped += 1


# voxGrid = np.asarray(voxelGridExtract)
# grid = grid.flatten()

# print(np.shape(grid))
# print(dropped)

# grid.tofile(saveAt)
