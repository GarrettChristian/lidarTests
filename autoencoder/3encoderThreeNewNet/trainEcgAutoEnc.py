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
from anomolyDetector import DataGenerator

# -------------------------------------

autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')


path = "/media/garrett/Extreme SSD/semKitti/dataset/sequences/00/"
# path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/"

files = np.array(glob.glob(path + "**/00*.bin", recursive = True))
print(np.shape(files))

# Parameters
params = {'dim': (2070272,),
          'batch_size': 100,
          'n_channels': 1,
          'shuffle': True}


# Datasets
labels = np.ones(np.shape(files)[0]) # Labels

train_data, test_data, train_labels, test_labels = train_test_split(
    files, labels, test_size=0.2, random_state=21
)

# Generators
training_generator = DataGenerator(train_data, train_labels, **params)
validation_generator = DataGenerator(test_data, test_labels, **params)

history = autoencoder.fit(training_generator, validation_data=validation_generator, epochs=20)

autoencoder.save("pcdModel")