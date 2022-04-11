
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import open3d as o3d

from PIL import Image
from PIL import ImageOps

# ------------------------------------

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Load model
autoencoder = keras.models.load_model('1pcdModel')

# Get test images
# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
# testImage = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/00/000000.png"
# testImage = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/00/000000.png"
# testImage = "unrealistic1.png"
# testImage = "/Volumes/Extreme SSD/rangeimgs/21/000000.png"
# testImage = "/Volumes/Extreme SSD/rangeimgs/21/000000.png"
binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/voxels2/00/000000.bin"
# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/voxels2/21/000000.bin"
# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/uncomforming/vox2/noRoad00Vox2.bin"

# binFileName = "test.bin"


fromFile = np.fromfile(binFileName, dtype=np.ubyte)

xyzArray = fromFile.reshape((int(np.shape(fromFile)[0]) // 3, 3))

grid = np.zeros((256, 256), dtype=np.float32)
#        grid = np.zeros((256, 256, 32), dtype=np.float32)s

for xyz in xyzArray:
    grid[xyz[0]][xyz[1]] = 1

test_arr = np.expand_dims(grid, axis=2)

print(np.shape(test_arr))
test_arr = np.array([test_arr])
print(np.shape(test_arr))

ogVoxArray = []
for x in range (256):
    for y in range (256):
          if (test_arr[0][x][y][0] > 0.5):
                ogVoxArray.append(x)
                ogVoxArray.append(y)
                ogVoxArray.append(0)


np_vox2og = np.asarray(ogVoxArray)
np_vox2og = np_vox2og.reshape((int(np.shape(np_vox2og)[0]) // 3, 3))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np_vox2og)
o3d.visualization.draw_geometries([pcd])

# ---

decoded_data = autoencoder.predict([test_arr])

print(decoded_data)
print(np.shape(decoded_data))
print(np.shape(decoded_data[0]))

decoded_data0 = decoded_data[0]

# print(decoded_data0)

voxArray = []
for x in range (256):
    for y in range (256):
          if (decoded_data0[x][y][0] > 0.5):
                voxArray.append(x)
                voxArray.append(y)
                voxArray.append(0)


np_vox2 = np.asarray(voxArray)

np_vox2 = np_vox2.astype(np.ubyte)

np_vox2 = np_vox2.reshape((int(np.shape(np_vox2)[0]) // 3, 3))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np_vox2)
o3d.visualization.draw_geometries([pcd])




# decoded_data_unnorm = decoded_data0.astype('float32') * 255
# undoExpandDims = decoded_data_unnorm.reshape(64, 1024)

# decodedImage = Image.fromarray(undoExpandDims)
# print("dec")
# decodedImage.show()
# print("imgGrey")
# imageGrey.show()


# print(np.shape(test_arr))
# print(np.shape(decoded_data))
print(SSIMLoss(test_arr, decoded_data))

# For later:
# https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
