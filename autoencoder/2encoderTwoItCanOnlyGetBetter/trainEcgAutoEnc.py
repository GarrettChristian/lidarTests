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

lidar_data = np.array([])

path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/"
for file in glob.glob(path + "**/000*.bin", recursive = True):
    print(file)
    pcd = np.fromfile(file, dtype=np.float32)
    # print(np.shape(pcd))
    
    pcd = pcd.reshape((int(np.shape(pcd)[0]) // 4, 4))
    pcd = np.delete(pcd, 3, 1)
    pcd = pcd.reshape(np.size(pcd),)
    pcd = np.pad(pcd, (0,1552704 - int(np.shape(pcd)[0])), 'constant', constant_values=(0))
    
    # print(np.shape(pcd))
    # print(pcd)

    if (np.size(lidar_data)):
        lidar_data = np.vstack([lidar_data, pcd])
    else:
        lidar_data = np.array([pcd])

print(lidar_data)
print(np.shape(lidar_data))

labels = np.ones(np.shape(lidar_data)[0])
# print(labels)
# print(np.size(labels))

train_data, test_data, train_labels, test_labels = train_test_split(
    lidar_data, labels, test_size=0.2, random_state=21
)


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

history = autoencoder.fit(train_data, train_data, 
          epochs=20, 
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)


autoencoder.save("pcdModel")