import struct
import numpy as np

# https://stackoverflow.com/questions/8710456/reading-a-binary-file-with-python

filePath = "./../data/sets/nuscenes/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402928147847.pcd.bin"

data = np.fromfile(filePath)


print(data[0])
print(data.ndim)
print(data.shape)

# -5.158165553740423e-06
# 1
# (86800,)








