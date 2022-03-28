import matplotlib.pyplot as plt
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


def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

# ------------------------------------

autoencoder = keras.models.load_model('pcdModel')

# -------------------------------------


# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
binFileName = "/media/garrett/Extreme SSD/semKitti/dataset/sequences/00/velodyne/000000.bin"
np_arr = np.fromfile(binFileName, dtype=np.float32)
print(np.shape(np_arr))


pcdi = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
pcdi = np.delete(pcdi, np.s_[2::], 1)
pcdi = pcdi.reshape(np.size(pcdi),)
pcdi = np.pad(pcdi, (0,1035136 - int(np.shape(pcdi)[0])), 'constant', constant_values=(0))
test_arr = np.array([pcdi])

# np_arr = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
# print(np.shape(np_arr))
# np_arr = np.delete(np_arr, 3, 1)
# print(np.shape(np_arr))

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np_arr)
# o3d.visualization.draw_geometries([pcd])


# ---

encoded_data = autoencoder.encoder([test_arr]).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

print(np.shape(decoded_data))
print(np.shape(decoded_data[0]))

# dec_reshape = decoded_data[0]
# dec_reshape = dec_reshape.reshape((int(np.shape(dec_reshape)[0]) // 4, 4))
# print(np.shape(dec_reshape))
# dec_reshape = np.delete(dec_reshape, 3, 1)
# print(np.shape(dec_reshape))

dec_reshape = decoded_data[0]
dec_reshape = dec_reshape.reshape((int(np.shape(dec_reshape)[0]) // 2), 2)
new_col = dec_reshape.sum(1)[...,None]

all_data = np.hstack((dec_reshape, new_col))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_data)
o3d.visualization.draw_geometries([pcd])

print(all_data)
