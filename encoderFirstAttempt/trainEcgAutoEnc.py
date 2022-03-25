"""
https://www.tensorflow.org/tutorials/generative/autoencoder

"""


import atexit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import glob, os
import struct

from anomolyDetector import AnomalyDetector

autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam', loss='mae')

# -------------------------------------

dataframe = pd.read_csv('ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()

# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

# normalize data to [0, 1]

# print(train_data)
# print(np.shape(train_data))

# trainLidar = np.array([])

# path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/"
# for file in glob.glob(path + "**/0000*.bin", recursive = True):
#     print(file)
#     size_float = 4
#     list_pcd = []
#     with open(file, "rb") as f:
#         byte = f.read(size_float * 4)
#         for x in range(0, 2070272):
#         # while byte:
#             if byte:
#                 x, y, z, intensity = struct.unpack("ffff", byte)
#                 list_pcd.append([x, y, z])
#                 byte = f.read(size_float * 4)
#             else:
#                 list_pcd.append([0, 0, 0])
#     np_pcd = np.asarray(list_pcd)
#     pcd = np.array([np_pcd])
#     # print("shape new", np.shape(pcd))
#     # print(pcd)
#     # print("shape new", np.shape(trainLidar))
#     # print(trainLidar)
#     if (np.size(trainLidar)):
#         trainLidar = np.vstack((trainLidar, pcd))
#     else:
#         trainLidar = np.array([np_pcd])
        
# print("shape new", np.shape(trainLidar))

trainLidar = np.array([])

path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/"
for file in glob.glob(path + "**/0000*.bin", recursive = True):
    print(file)
    pcd = np.fromfile(file, dtype=np.float32)
    # print(np.shape(pcd))

    # pcd = pcd.reshape((int(np.shape(pcd)[0]) // 4, 4))
    pcd = np.pad(pcd, (0,2070272 - int(np.shape(pcd)[0])), 'constant', constant_values=(0))
    # print(np.shape(pcd))
    # print(pcd)

    if (np.size(trainLidar)):
        trainLidar = np.vstack((trainLidar, pcd))
    else:
        trainLidar = np.array([pcd])

        
print("shape new", np.shape(trainLidar))

testLidar = np.array([])

path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/"
for file in glob.glob(path + "**/00001*.bin", recursive = True):
    print(file)
    size_float = 4
    list_pcd = []
    with open(file, "rb") as f:
        byte = f.read(size_float * 4)
        for x in range(0, 2070272):
        # while byte:
            if byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 4)
            else:
                list_pcd.append([0, 0, 0])
    np_pcd = np.asarray(list_pcd)
    pcd = np.array([np_pcd])
    # print("shape new", np.shape(pcd))
    # print(pcd)
    # print("shape new", np.shape(trainLidar))
    # print(trainLidar)
    if (np.size(testLidar)):
        testLidar = np.vstack((testLidar, pcd))
    else:
        testLidar = np.array([np_pcd])
        
print("shape new", np.shape(testLidar))


# min_val = tf.reduce_min(train_data)
# max_val = tf.reduce_max(train_data)

# train_data = (train_data - min_val) / (max_val - min_val)
# test_data = (test_data - min_val) / (max_val - min_val)

# train_data = tf.cast(train_data, tf.float32)
# test_data = tf.cast(test_data, tf.float32)

# # You will train the autoencoder using only the normal rhythms, which are labeled in this dataset as 1. 
# # Separate the normal rhythms from the abnormal rhythms.

# train_labels = train_labels.astype(bool)
# test_labels = test_labels.astype(bool)

# normal_train_data = train_data[train_labels]
# normal_test_data = test_data[test_labels]

# anomalous_train_data = train_data[~train_labels]
# anomalous_test_data = test_data[~test_labels]

# Notice that the autoencoder is trained using only the normal ECGs, 
# but is evaluated using the full test set.

history = autoencoder.fit(trainLidar, trainLidar, 
          epochs=20, 
          batch_size=512,
          validation_data=(testLidar, testLidar),
          shuffle=True)


autoencoder.save("ecgModel")