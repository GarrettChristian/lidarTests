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
binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/16/velodyne/000000.bin"
np_arr = np.fromfile(binFileName, dtype=np.float32)
print(np.shape(np_arr))

np_arr = np.pad(np_arr, (0,1552704 - int(np.shape(np_arr)[0])), 'constant', constant_values=(0))
print(np.shape(np_arr))
test_arr = np.array([np_arr])

print(np.shape(test_arr))


# np_arr = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
# print(np.shape(np_arr))
# np_arr = np.delete(np_arr, 3, 1)
# print(np.shape(np_arr))

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np_arr)
# o3d.visualization.draw_geometries([pcd])


# ---

encoded_data = autoencoder.encoder(test_arr).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

print(np.shape(decoded_data))
print(np.shape(decoded_data[0]))

dec_reshape = decoded_data[0]
dec_reshape = dec_reshape.reshape((int(np.shape(dec_reshape)[0]) // 4, 4))
print(np.shape(dec_reshape))
dec_reshape = np.delete(dec_reshape, 3, 1)
print(np.shape(dec_reshape))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(dec_reshape)
o3d.visualization.draw_geometries([pcd])


