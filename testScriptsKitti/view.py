import numpy as np
import struct
import sys
import open3d as o3d


def bin_to_pcd(binFileName):
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
            print(intensity)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd

def main():
    # binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
    binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000001.bin"

    pcd = bin_to_pcd(binFileName)
    print(np.asarray(pcd))
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])


    # o3d.io.write_point_cloud(pcdFileName, pcd)

if __name__ == "__main__":
    main()