
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
path = "/Volumes/Extreme SSD/rangeimgs/01/"
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
print(ssimLoss)






