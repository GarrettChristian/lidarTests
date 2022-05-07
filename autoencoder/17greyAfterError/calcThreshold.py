
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
    newSize = (512, 32)
    imageGreyResized = imageGrey.resize(newSize)
    imageGreyArray = np.array(imageGreyResized)
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
autoencoder = keras.models.load_model('v3/pcdModel')

# Get test images
# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
# path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/"
path = "/p/lidarrealism/data/range2TrainVal/"
samples = getSampleSet(path, 1000)

# print(samples)
print(np.shape(samples))

reconstructions = autoencoder.predict([samples])

ssimLoss = SSIMLoss(samples, reconstructions)

print("----------------------------------------")
print("train")
print(ssimLoss)

print("----------------------------------------")
pathHidden = "/p/lidarrealism/data/range2Test/"
samplesHidden = getSampleSet(pathHidden, 1000)

reconstructionsHidden = autoencoder.predict([samplesHidden])
ssimLoss = SSIMLoss(samplesHidden, reconstructionsHidden)
print("test")
print(ssimLoss)

print("----------------------------------------")







