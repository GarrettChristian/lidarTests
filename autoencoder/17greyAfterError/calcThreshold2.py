
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


import matplotlib.pyplot as plt

import glob
import random

from PIL import Image
from PIL import ImageOps

# ------------------------------------

# https://medium.com/analytics-vidhya/image-anomaly-detection-using-autoencoders-ae937c7fd2d1
def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def convertImage(imagePath):
    image = Image.open(imagePath)
    imageGrey = ImageOps.grayscale(image)
    # newSize = (512, 32)
    # imageGreyResized = imageGrey.resize(newSize)
    imageGreyArray = np.array(imageGrey)
    imageGreyArrayNorm = imageGreyArray.astype('float32') / 255
    image_arr = np.expand_dims(imageGreyArrayNorm, axis=2) 
    # print(np.shape(image_arr))
    return image_arr

def getSampleSet(basePath, sampleCount):
    results = np.array([])
    files = np.array(glob.glob(basePath + "*/*.png", recursive = True))

    for i in range(sampleCount):
        fileLoc = random.choice(files)
        npImage = convertImage(fileLoc)

        if (np.size(results)):
            results = np.vstack((results, [npImage]))
        else:
            results = np.array([npImage])

    return results

# ------------------------------------

# Load model
autoencoder = keras.models.load_model('v35/pcdModel')

# Get test images
# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
# path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/"
# path = "/p/lidarrealism/data/range2TrainVal/"
path = "/p/lidarrealism/data/rangeimgs/"
samples = getSampleSet(path, 1000)

# print(samples)
print(np.shape(samples))

reconstructions = autoencoder.predict([samples])

ssimLoss = SSIMLoss(samples, reconstructions)

print("----------------------------------------")
print("train")
print(ssimLoss)

print("----------------------------------------")
# pathHidden = "/p/lidarrealism/data/range2Test/"
pathHidden = "/p/lidarrealism/data/testRanImg/"
samplesHidden = getSampleSet(pathHidden, 1000)

reconstructionsHidden = autoencoder.predict([samplesHidden])
ssimLoss = SSIMLoss(samplesHidden, reconstructionsHidden)
print("test")
print(ssimLoss)

print("----------------------------------------")



path00 = "/p/lidarrealism/data/rangeimgs/00/000000.png"
controlImage = convertImage(path00)
control = np.array([controlImage])

reconstructionsControl = autoencoder.predict([control])
ssimLossControl = SSIMLoss(control, reconstructionsControl)

print("Loss on 00/000000")
print("Loss: ", ssimLossControl)

print("----------------------------------------")

pathUn1 = "/p/lidarrealism/data/uncomforming/png/carsUp1Z00.png"
un1Image = convertImage(pathUn1)
un1 = np.array([un1Image])

reconstructionsUn1 = autoencoder.predict([un1])
ssimLossUn1 = SSIMLoss(un1, reconstructionsUn1)

print("Loss on floating cars")
print("Loss: ", ssimLossUn1)

print("----------------------------------------")

pathUn2 = "/p/lidarrealism/data/uncomforming/png/removeRoad00.png"
un2Image = convertImage(pathUn2)
un2 = np.array([un2Image])

reconstructionsUn2 = autoencoder.predict([un2])
ssimLossUn2 = SSIMLoss(un2, reconstructionsUn2)

print("Loss on no road")
print("Loss: ", ssimLossUn2)


