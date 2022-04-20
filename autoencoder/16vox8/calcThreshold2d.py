
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

import matplotlib.pyplot as plt

import glob
import random

from PIL import Image
from PIL import ImageOps

# ------------------------------------

# https://medium.com/analytics-vidhya/image-anomaly-detection-using-autoencoders-ae937c7fd2d1
def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def convertVox(path):
    fromFile = np.fromfile(path, dtype=np.ubyte)
    xyzArray = fromFile.reshape((int(np.shape(fromFile)[0]) // 3, 3))
    grid = np.zeros((64, 64), dtype=np.float32)
    for xyz in xyzArray:
        grid[xyz[0]][xyz[1]] = 1

    vox2dArray = np.expand_dims(grid, axis=2)
    return vox2dArray

def getSampleSet(basePath, sampleCount):
    results = np.array([])
    files = np.array(glob.glob(basePath + "*/*.bin", recursive = True))

    for i in range(sampleCount):
        fileLoc = random.choice(files)
        vox2D = convertVox(fileLoc)

        if (np.size(results)):
            results = np.vstack((results, [vox2D]))
        else:
            results = np.array([vox2D])

    return results




# ------------------------------------

# Load model
modelName = '4pcdModel'
autoencoder = keras.models.load_model(modelName)
print("Info for ", modelName)


# Get test images
# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
# path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/"
# path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/"
# path = "/Volumes/Extreme SSD/rangeimgs/"
path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/voxels8/"
samples = getSampleSet(path, 1000)

# print(samples)
print(np.shape(samples))

reconstructions = autoencoder.predict([samples])
# train_loss = tf.keras.losses.mae(reconstructions, samples)

# print(train_loss)
# print(np.shape(train_loss))

# ssim = tf.image.ssim(reconstructions, samples, max_val=1.0)
# print(ssim)
ssimLoss = SSIMLoss(samples, reconstructions)

print("----------------------------------------")
print("1000 random of train/val section")
print("Loss: ", ssimLoss)

print("----------------------------------------")
pathHidden = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/hiddenVox8/"
samplesHidden = getSampleSet(path, 1000)

reconstructionsHidden = autoencoder.predict([samplesHidden])
ssimLossHidden = SSIMLoss(samples, reconstructionsHidden)

print(ssimLoss)
print("1000 random of train/val section")
print("Loss: ", ssimLossHidden)

print("----------------------------------------")

path00 = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/voxels8/00/000000.bin"
controlImage = convertVox(path00)
control = np.array([controlImage])

reconstructionsControl = autoencoder.predict([control])
ssimLossControl = SSIMLoss(control, reconstructionsControl)

print("Loss on 00/000000")
print("Loss: ", ssimLossControl)

print("----------------------------------------")

grid = np.ones((64, 64), dtype=np.float32)
for x in range(0, 64):
    for y in range(0, 64):
        if (x % 2 == 0 and y % 2 == 0):
            grid[x][y] = 0

un2Image = np.expand_dims(grid, axis=2)
un2 = np.array([un2Image])

reconstructionsUn2 = autoencoder.predict([un2])
ssimLossUn2 = SSIMLoss(un2, reconstructionsUn2)

print("Loss on noisy")
print("Loss: ", ssimLossUn2)







