
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

from PIL import Image
from PIL import ImageOps

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

# ------------------------------------

# Load model
autoencoder = keras.models.load_model('pcdModel')

# Get test image
# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
testImage = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/00/000000.png"
image = Image.open(testImage)
imageGrey = ImageOps.grayscale(image)
imageGreyArray = np.array(imageGrey)
imageGreyArrayNorm = imageGreyArray.astype('float32') / 255
test_arr = np.expand_dims(imageGreyArrayNorm, axis=2) 
print(np.shape(test_arr))
test_arr = np.array([test_arr])

# ---

decoded_data = autoencoder.predict([test_arr])

print(decoded_data)
print(np.shape(decoded_data))
print(np.shape(decoded_data[0]))

decoded_data0 = decoded_data[0]
decoded_data_unnorm = decoded_data0.astype('float32') * 255
undoExpandDims = decoded_data_unnorm.reshape(64, 1024)

decodedImage = Image.fromarray(undoExpandDims)
print("dec")
decodedImage.show()
print("imgGrey")
imageGrey.show()


# For later:
# https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
