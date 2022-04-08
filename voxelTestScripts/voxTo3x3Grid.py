"""
"""

import numpy as np

import glob, os
import struct
import open3d as o3d


# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/velodyne/000000.bin"
np_arr = np.fromfile(binFileName, dtype=np.float32)
print(np.shape(np_arr))

# np_arr = np.pad(np_arr, (0,2070272 - int(np.shape(np_arr)[0])), 'constant', constant_values=(0))
# print(np.shape(np_arr))

np_arr = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
print(np.shape(np_arr))
np_arr = np.delete(np_arr, 3, 1)
print(np.shape(np_arr))

np_arr = np_arr[np_arr[:, 2].argsort()]  
np_arr = np_arr[np_arr[:, 1].argsort(kind='mergesort')] 
np_arr = np_arr[np_arr[:, 0].argsort(kind='mergesort')]

# for x in np_arr:
#     print(x)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np_arr)


print('voxelization')
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.4)

print(voxel_grid)
# print(voxel_grid.get_voxels())

grid = np.zeros((400, 400, 500))


print("w/o bounds")
count = 0

voxelGridExtract = []
for voxel in voxel_grid.get_voxels():
    # print(voxel.grid_index[0])
    voxelGridExtract.append([voxel.grid_index[0], voxel.grid_index[1], voxel.grid_index[2]])
    count += 1
print("Voxels", count)
print("dim", voxel_grid.dimension())
print("max", voxel_grid.get_max_bound())
print("min", voxel_grid.get_min_bound())


# o3d.visualization.draw_geometries([voxel_grid])


voxGrid = np.asarray(voxelGridExtract)
print(np.shape(voxGrid))
print(voxGrid[0])



# voxelExtractionFunc = lambda voxel: 

# voxelExtractionFunc(voxGrid)


print(np.shape(voxGrid))
print(voxGrid[0])
# Sort by x, y, z
# voxGrid = np.lexsort((voxGrid[:,0], voxGrid[:,1], voxGrid[:,2]))

# Sort by x y z (this needs to be done in reverse)
voxGrid = voxGrid[voxGrid[:, 2].argsort()]  
voxGrid = voxGrid[voxGrid[:, 1].argsort(kind='mergesort')] 
voxGrid = voxGrid[voxGrid[:, 0].argsort(kind='mergesort')]

print(np.shape(voxGrid))
# print("Here ", voxGrid[1])

for xyz in voxGrid:
    grid[xyz[0]][xyz[1]][xyz[2]] = 1
    print(xyz)


# for x in range (512):
#     for y in range (512):
#         for z in range (64):



# Get point cloud back from vox grid
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(voxGrid)
# o3d.visualization.draw_geometries([pcd])


