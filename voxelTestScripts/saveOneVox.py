
import numpy as np

import glob, os
import struct
import open3d as o3d


binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"


# print(count, file)
np_arr = np.fromfile(binFileName, dtype=np.float32)
np_arr = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
np_arr = np.delete(np_arr, 3, 1)

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

np_vox = np.asarray(voxGrid)

np_vox = np_vox.astype(np.ubyte)

np_vox.tofile("test.bin")

# ----

fromFile = np.fromfile("test.bin", dtype=np.ubyte)

xyzArray = fromFile.reshape((int(np.shape(fromFile)[0]) // 3, 3))

grid = np.zeros((256, 256, 32), dtype=np.ubyte)

for xyz in xyzArray:
    grid[xyz[0]][xyz[1]][xyz[2]] = 1

# turn it back into a vector array 

voxArray = []

for x in range (256):
    for y in range (256):
        for z in range (32):
            if (grid[x][y][z] == 1):
                voxArray.append(x)
                voxArray.append(y)
                voxArray.append(z)


np_vox2 = np.asarray(voxGrid)

np_vox2 = np_vox2.astype(np.ubyte)

np_vox2 = np_vox2.reshape((int(np.shape(np_vox2)[0]) // 3, 3))

pcd.points = o3d.utility.Vector3dVector(np_vox2)
o3d.visualization.draw_geometries([pcd])


