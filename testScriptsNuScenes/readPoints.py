# https://github.com/nutonomy/nuscenes-devkit/issues/17

import numpy as np

file_path = './../data/sets/nuscenes/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402928147847.pcd.bin'

with open(file_path, "rb") as f:
    number = f.read(4)
    while number != b"":
        print(np.frombuffer(number, dtype=np.float32))
        number = f.read(4)


