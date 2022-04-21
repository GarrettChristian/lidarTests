
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
    imageGreyArray = np.array(imageGrey)
    imageGreyArrayNorm = imageGreyArray.astype('float32') / 255
    image_arr = np.expand_dims(imageGreyArrayNorm, axis=2) 
    # print(np.shape(image_arr))
    return image_arr

def getSampleSet(basePath, sampleCount):
    results = np.array([])
    # files = np.array(glob.glob(basePath + "*/*.png", recursive = True))
    files = np.array(glob.glob(basePath + "*.png", recursive = True))

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
encoder = tf.keras.models.load_model("encoderVaeModel") 
decoder = tf.keras.models.load_model("decoderVaeModel")

# Get test images
# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
# path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/"
# path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/00/"
# path = "/Volumes/Extreme SSD/rangeimgs/01/"
#path = "/p/lidarrealism/data/range2TrainVal/"
# path = "/p/lidarrealism/data/range2Test/"
path = "/Volumes/Extreme SSD/rangeimgs2/"

samples = getSampleSet(path, 100)

# print(samples)
print(np.shape(samples))

z_mean, _, _ = encoder.predict([samples])
print(np.shape(z_mean))
# print(z_mean)
reconstructions = decoder.predict(z_mean)
# train_loss = tf.keras.losses.mae(reconstructions, samples)

# print(train_loss)
# print(np.shape(train_loss))

# ssim = tf.image.ssim(reconstructions, samples, max_val=1.0)
# print(ssim)
ssimLoss = SSIMLoss(samples, reconstructions)


print("----------------------------------------")
print("100 random of train/val section")
print("Loss: ", ssimLoss)

print("----------------------------------------")
# pathHidden = "/Volumes/Extreme SSD/hiddenRangeImgs/"
pathHidden = "/Volumes/Extreme SSD/hiddenRangeImgs2/"
samplesHidden = getSampleSet(path, 1000)

z_mean, _, _ = encoder.predict([samplesHidden])
reconstructionsHidden = decoder.predict(z_mean)
ssimLossHidden = SSIMLoss(samples, reconstructionsHidden)

print(ssimLoss)
print("100 random of train/val section")
print("Loss: ", ssimLossHidden)

print("----------------------------------------")

path00 = "/Volumes/Extreme SSD/rangeimgs/00/000000.png"
controlImage = convertImage(path00)
control = np.array([controlImage])

z_mean, _, _ = encoder.predict([control])
reconstructionsControl = decoder.predict(z_mean)
ssimLossControl = SSIMLoss(control, reconstructionsControl)

print("Loss on 00/000000")
print("Loss: ", ssimLossControl)

print("----------------------------------------")

pathUn1 = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/uncomforming/png/carsUp1Z00.png"
un1Image = convertImage(pathUn1)
un1 = np.array([un1Image])

z_mean, _, _ = encoder.predict([un1])
reconstructionsUn1 = decoder.predict(z_mean)
ssimLossUn1 = SSIMLoss(un1, reconstructionsUn1)

print("Loss on floating cars")
print("Loss: ", ssimLossUn1)

print("----------------------------------------")

pathUn2 = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/uncomforming/png/removeRoad00.png"
un2Image = convertImage(pathUn2)
un2 = np.array([un2Image])

z_mean, _, _ = encoder.predict([un2])
reconstructionsUn2 = decoder.predict(z_mean)
ssimLossUn2 = SSIMLoss(un2, reconstructionsUn2)

print("Loss on no road")
print("Loss: ", ssimLossUn2)






