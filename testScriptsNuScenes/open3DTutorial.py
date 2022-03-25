# http://www.open3d.org/docs/0.9.0/tutorial/Basic/pointcloud.html

# https://stackoverflow.com/questions/60506331/conversion-of-binary-lidar-data-bin-to-point-cloud-data-pcd-format

# examples/Python/Basic/pointcloud.py

from stat import filemode
import numpy as np
import open3d as o3d
import struct
import data_classes

file_path = './../data/sets/nuscenes/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402928147847.pcd.bin'

# if __name__ == "__main__":

#     print("Load a ply point cloud, print it, and render it")
#     pcd = o3d.io.read_point_cloud("../../TestData/fragment.ply")
#     print(pcd)
#     print(np.asarray(pcd.points))
#     o3d.visualization.draw_geometries([pcd])


# Load binary point cloud
# bin_pcd = np.fromfile(file_path, dtype=np.float32)

# # # Reshape and drop reflection values
# # points = bin_pcd.reshape((-1, 4))[:, 0:3]
# points = bin_pcd.reshape((-1, 5))[:, :4]

# # # Convert to Open3D point cloud
# o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

# # # Save to whatever format you like
# o3d.io.write_point_cloud("pointcloud.pcd", o3d_pcd)

# list_pcd = []

x = data_classes.LidarPointCloud.from_file(file_path)
pointz = [x.points]
print(pointz[0][0])
print(len(pointz[0][0]))
# pcdV2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointz))
# o3d.io.write_point_cloud("pointcloud.pcd", pcdV2)

# with open(file_path, "rb") as f:
#     number = f.read(4)
#     while number != b"":
#         print(np.frombuffer(number, dtype=np.float32))
#         number = f.read(4)

# with open (file_path, "rb") as f:
#     byte = f.read(8*4)
#     while byte:
#         x,y,z,intensity = struct.unpack("ffff", byte)
#         list_pcd.append([x, y, z])
#         byte = f.read(8*4)
# np_pcd = np.asarray(list_pcd)
# pcd = o3d.geometry.PointCloud()
# v3d = o3d.utility.Vector3dVector
# pcd.points = v3d(np_pcd)

# print("Load a ply point cloud, print it, and render it")
# pcd = o3d.io.read_point_cloud("pointcloud.pcd")
# print(pcd)
# print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd])
# print(np.asarray(o3d_pcd))
# o3d.visualization.draw_geometries([o3d_pcd])