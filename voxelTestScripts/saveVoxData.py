"""
"""

import numpy as np

import glob, os
import struct
import open3d as o3d
import sys


path = "/Volumes/Extreme SSD/semKitti/dataset/sequences/"
saveBase = "voxels4"


# Set up directories if they don't exist
isExist = os.path.exists(saveBase)
if not isExist:
  os.makedirs(saveBase)
  
for x in range(0, 22):
    folderNum = str(x).rjust(2, '0')
    isExist = os.path.exists(saveBase + "/" + folderNum)
    if not isExist:
        os.makedirs(saveBase + "/" + folderNum)

for x in range(0, 22):
    
    folderNum = str(x).rjust(2, '0')
    currPath = path + folderNum

    files = np.array(glob.glob(currPath + "/velodyne/*.bin", recursive = True))
    print("parsing ", currPath)
    
    for file in files:

        fileName = os.path.basename(file)
        
        saveAt = saveBase + "/" + folderNum + "/" + fileName
        isExist = os.path.exists(saveAt)
        if not isExist:

            # print(count, file)
            np_arr = np.fromfile(file, dtype=np.float32)
            np_arr = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
            np_arr = np.delete(np_arr, 3, 1)

            mask = ((np_arr[:, 0] < 25.6) & (np_arr[:, 0] > -25.6) 
                & (np_arr[:, 1] < 25.6) & (np_arr[:, 1] > -25.6) 
                & (np_arr[:, 2] < 3.4) & (np_arr[:, 2] > -3.4))

            np_arr = np_arr[mask, :]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_arr)

            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.4)

            voxGrid = []

            for voxel in voxel_grid.get_voxels():

                x = voxel.grid_index[0]
                y = voxel.grid_index[1]
                z = voxel.grid_index[2]

                if (x < 128 and y < 128 and z < 16):

                    voxGrid.append(x)
                    voxGrid.append(y)
                    voxGrid.append(z)
                    

            np_vox = np.asarray(voxGrid)
            np_vox = np_vox.astype(np.ubyte)

            np_vox.tofile(saveAt)





