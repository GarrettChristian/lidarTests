import numpy as np
import struct
import sys
import open3d as o3d


def bin_to_pcd(binFileName):
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 5)
        while byte:
            x, y, z, intensity, ringIndex = struct.unpack("fffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 5)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    print(np.shape(np_pcd))
    return pcd

def main():
    # binFileName = './../data/sets/nuscenes/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402928147847.pcd.bin'
    # binFileName = '/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/nuscenes/samples/LIDAR_TOP/n008-2018-08-30-15-16-55-0400__LIDAR_TOP__1535657119649820.pcd.bin'
    binFileName = '/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/nuscenes/samples/LIDAR_TOP/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542800862447900.pcd.bin'

    pcd = bin_to_pcd(binFileName)
    print(np.asarray(pcd))
    o3d.visualization.draw_geometries([pcd])


    # o3d.io.write_point_cloud(pcdFileName, pcd)

if __name__ == "__main__":
    main()