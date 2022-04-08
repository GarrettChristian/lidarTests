"""
"""

import numpy as np

import glob, os
import struct
import open3d as o3d
import sys



minSize2 = sys.maxsize
maxSize2 = sys.maxsize * -1
avgSize2 = 0
minX2 = sys.maxsize
maxX2 = sys.maxsize * -1
minY2 = sys.maxsize
maxY2 = sys.maxsize * -1
minZ2 = sys.maxsize
maxZ2 = sys.maxsize * -1
minSize4 = sys.maxsize
maxSize4 = sys.maxsize * -1
avgSize4 = 0
minX4 = sys.maxsize
maxX4 = sys.maxsize * -1
minY4 = sys.maxsize
maxY4 = sys.maxsize * -1
minZ4 = sys.maxsize
maxZ4 = sys.maxsize * -1
minSize5 = sys.maxsize
maxSize5 = sys.maxsize * -1
avgSize5 = 0
minX5 = sys.maxsize
maxX5 = sys.maxsize * -1
minY5 = sys.maxsize
maxY5 = sys.maxsize * -1
minZ5 = sys.maxsize
maxZ5 = sys.maxsize * -1
count = 0

totalVoxels = 0
dropped = 0

path = "/Volumes/Extreme SSD/semKitti/dataset/sequences/"
saveBase = "voxels4"

xdim = 512
ydim = 512
zdim = 64

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

            count += 1
            # print(count, file)
            np_arr = np.fromfile(file, dtype=np.float32)
            np_arr = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
            np_arr = np.delete(np_arr, 3, 1)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_arr)

            voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.2)

            maxSize2 = max(maxSize2, len(voxel_grid2.get_voxels()))
            minSize2 = min(minSize2, len(voxel_grid2.get_voxels()))
            avgSize2 += len(voxel_grid2.get_voxels())

            grid = np.zeros((xdim, ydim, zdim))

            for voxel in voxel_grid2.get_voxels():
                maxX2 = max(maxX2, voxel.grid_index[0])
                minX2 = min(minX2, voxel.grid_index[0])
                maxY2 = max(maxY2, voxel.grid_index[1])
                minY2 = min(minY2, voxel.grid_index[1])
                maxZ2 = max(maxZ2, voxel.grid_index[2])
                minZ2 = min(minZ2, voxel.grid_index[2])

            
            voxel_grid4 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.4)

            maxSize4 = max(maxSize4, len(voxel_grid4.get_voxels()))
            minSize4 = min(minSize4, len(voxel_grid4.get_voxels()))
            avgSize4 += len(voxel_grid4.get_voxels())

            for voxel in voxel_grid4.get_voxels():
                maxX4 = max(maxX4, voxel.grid_index[0])
                minX4 = min(minX4, voxel.grid_index[0])
                maxY4 = max(maxY4, voxel.grid_index[1])
                minY4 = min(minY4, voxel.grid_index[1])
                maxZ4 = max(maxZ4, voxel.grid_index[2])
                minZ4 = min(minZ4, voxel.grid_index[2])

                x = voxel.grid_index[0]
                y = voxel.grid_index[1]
                z = voxel.grid_index[2]

                if (x < xdim and y < ydim and z < zdim):
                    grid[x][y][z] = 1
                else:
                    dropped += 1
                

            voxel_grid5 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.5)

            maxSize5 = max(maxSize5, len(voxel_grid5.get_voxels()))
            minSize5 = min(minSize5, len(voxel_grid5.get_voxels()))
            avgSize5 += len(voxel_grid5.get_voxels())

            for voxel in voxel_grid5.get_voxels():
                maxX5 = max(maxX5, voxel.grid_index[0])
                minX5 = min(minX5, voxel.grid_index[0])
                maxY5 = max(maxY5, voxel.grid_index[1])
                minY5 = min(minY5, voxel.grid_index[1])
                maxZ5 = max(maxZ5, voxel.grid_index[2])
                minZ5 = min(minZ5, voxel.grid_index[2])

            # Save 4
            # voxelGridExtract = []
            # for voxel in voxel_grid4.get_voxels():
            #     voxelGridExtract.append([voxel.grid_index[0], voxel.grid_index[1], voxel.grid_index[2]])

            # voxGrid = np.asarray(voxelGridExtract)
            # grid.flatten()

            # grid.tofile(saveAt)


print("count ", count)
print("------")
print("avg 2: ", avgSize2 / count)
print("min 2: ", minSize2)
print("max 2: ", maxSize2)
print("min X 2: ", minX2)
print("max X 2: ", maxX2)
print("min Y 2: ", minY2)
print("max Y 2: ", maxY2)
print("min Z 2: ", minZ2)
print("max Z 2: ", maxZ2)
print("------")
print("avg 4: ", avgSize4 / count)
print("min 4: ", minSize4)
print("max 4: ", maxSize4)
print("min X 4: ", minX4)
print("max X 4: ", maxX4)
print("min Y 4: ", minY4)
print("max Y 4: ", maxY4)
print("min Z 4: ", minZ4)
print("max Z 4: ", maxZ4)
print("------")
print("avg 5: ", avgSize5 / count)
print("min 5: ", minSize5)
print("max 5: ", maxSize5)
print("min X 5: ", minX5)
print("max X 5: ", maxX5)
print("min Y 5: ", minY5)
print("max Y 5: ", maxY5)
print("min Z 5: ", minZ5)
print("max Z 5: ", maxZ5)
print("------")
print("total voxels: ", (avgSize4 * 3))
print("dropped: ", dropped)

# path = "/Volumes/Extreme SSD/semKitti/dataset/sequences/"
# for file in glob.glob(path + "00/velodyne/000*.bin", recursive = True):
#     count += 1
#     print(count, file)
#     np_arr = np.fromfile(file, dtype=np.float32)
#     np_arr = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
#     np_arr = np.delete(np_arr, 3, 1)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np_arr)

#     voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.2)

#     maxSize2 = max(maxSize2, len(voxel_grid2.get_voxels()))
#     minSize2 = min(minSize2, len(voxel_grid2.get_voxels()))
#     avgSize2 += len(voxel_grid2.get_voxels())
    
#     voxel_grid4 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.4)

#     maxSize4 = max(maxSize4, len(voxel_grid4.get_voxels()))
#     minSize4 = min(minSize4, len(voxel_grid4.get_voxels()))
#     avgSize4 += len(voxel_grid4.get_voxels())

#     voxel_grid5 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.5)

#     maxSize5 = max(maxSize5, len(voxel_grid5.get_voxels()))
#     minSize5 = min(minSize5, len(voxel_grid5.get_voxels()))
#     avgSize5 += len(voxel_grid5.get_voxels())

#     pcd_arr1 = pcd_arr1.flatten()
#     print(np.shape(pcd_arr1))
#     pcd_arr1.tofile("")



