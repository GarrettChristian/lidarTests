import numpy as np
import tensorflow as tf


from tensorflow.keras import layers
from tensorflow import keras
from keras.models import Sequential

from PIL import Image
from PIL import ImageOps


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

import glob, os
import struct

import random

# ----------------------------------------------------------------------------------------------------

# Based on:
# https://towardsdatascience.com/using-skip-connections-to-enhance-denoising-autoencoder-algorithms-849e049c0ac9

def create_model():

  input_img = layers.Input(shape=(64, 1024, 1))

  # Encoder
  y = layers.Conv2D(32, (3, 3), padding='same',strides =(2,2))(input_img)
  y = layers.LeakyReLU()(y)
  y = layers.Conv2D(64, (3, 3), padding='same',strides =(2,2))(y)
  y = layers.LeakyReLU()(y)
  y1 = layers.Conv2D(128, (3, 3), padding='same',strides =(2,2))(y) # skip-1
  y = layers.LeakyReLU()(y1)
  y = layers.Conv2D(256, (3, 3), padding='same',strides =(2,2))(y)
  y = layers.LeakyReLU()(y)
  y2 = layers.Conv2D(256, (3, 3), padding='same',strides =(2,2))(y)# skip-2
  y = layers.LeakyReLU()(y2)
  y = layers.Conv2D(512, (3, 3), padding='same',strides =(2,2))(y)
  y = layers.LeakyReLU()(y)
  y = layers.Conv2D(1024, (3, 3), padding='same',strides =(2,2))(y)
  y = layers.LeakyReLU()(y)
  #Flattening for the bottleneck
  x = layers.Flatten()(y)
  latent = layers.Dense(16384, activation='relu')(x)

  # Decoder 
  y = layers.Dense(16384, activation='relu')(latent)
  y = layers.Reshape((2, 32, 256))(y)
  y = layers.Conv2DTranspose(1024, (3,3), padding='same')(y)
  y = layers.LeakyReLU()(y)
  y = layers.Conv2DTranspose(512, (3,3), padding='same',strides=(2,2))(y)
  y = layers.LeakyReLU()(y)
  y = layers.Conv2DTranspose(256, (3,3), padding='same',strides=(2,2))(y)
  y = layers.Add()([y2, y]) # second skip connection added here
  y = layers.LeakyReLU()(y)
  y = layers.BatchNormalization()(y)
  y = layers.Conv2DTranspose(256, (3,3), padding='same',strides=(2,2))(y)
  y = layers.LeakyReLU()(y)
  y = layers.Conv2DTranspose(128, (3,3), padding='same',strides=(2,2))(y)
  y = layers.Add()([y1, y]) # first skip connection added here
  y = layers.LeakyReLU()(y)
  y = layers.BatchNormalization()(y)
  y = layers.Conv2DTranspose(64, (3,3), padding='same',strides=(2,2))(y)
  y = layers.LeakyReLU()(y)
  y = layers.Conv2DTranspose(32, (3,3), padding='same',strides=(2,2))(y)
  y = layers.LeakyReLU()(y)
  y = layers.Conv2DTranspose(3, (3,3), activation='sigmoid', padding='same',strides=(2,2))(y)

  return Model(input_img,y)

  


# ----------------------------------------------------------------------------------------------------

# Adapted from this tutorial:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://gist.github.com/twolodzko/aa4f4ad52f16c293df40342929b025a4?short_path=4d6ba34
# one above is really cool example of how the noise version works
class DataGenerator(keras.utils.Sequence):
  def __init__(self, list_IDs, labels, batch_size=100, dim=(2070272), n_channels=1, shuffle=True):
      'Initialization'
      self.dim = dim
      self.batch_size = batch_size
      self.labels = labels
      self.list_IDs = list_IDs
      self.n_channels = n_channels
      self.shuffle = shuffle
      self.on_epoch_end()

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    # X = np.empty((self.batch_size, *self.dim))
    # y = np.empty((self.batch_size, *self.dim))
    # y = np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        # X[i,] = np.load(ID)
        
        image = Image.open(ID)


        # 1 out of 10 times flip the image
        # if (random.random() % 10 == 0):
        #        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        imageGrey = ImageOps.grayscale(image)

        imageGreyArray = np.array(imageGrey)
        imageGreyArrayNorm = imageGreyArray.astype('float32') / 255
        
        X[i,] = np.expand_dims(imageGreyArrayNorm, axis=2)
        # print(np.shape(X[i,]))
        
        # Store class
        # y[i] = self.labels[ID]
        # y[i] = 1

    # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    # return X, y
    return X, X

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_IDs_temp)

    return X, y


# ----------------------------------------------------------------------------------------------------

def main():

    # PATH TO THE TRAINING FILES
  # path = "/media/garrett/Extreme SSD/rangeimgs/00/"
  # path = "/Volumes/Extreme SSD/rangeimgs/00/"
  # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/"
  # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/00/"
  path = "/p/lidarrealism/data/rangeimgs/"
  # path = "/p/lidarrealism/data/range2TrainVal/"
  # path = "/p/lidarrealism/data/rangeimgs/00/"
  # path = "/p/lidarrealism/data/voxel4/00/"
  # path = "/p/lidarrealism/data/voxel4/"
  # path = "/p/lidarrealism/data/voxels2/"
  # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/voxelTestScripts/voxels4/00/"
  # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/voxels2/00/"
  # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/voxels2/00/"
  # path = ""

  files = np.array(glob.glob(path + "*/*.png", recursive = True))
  # files = np.array(glob.glob(path + "*.png", recursive = True))
  print(np.shape(files))

  # Parameters
  # 'dim': (65536,),
  # 'dim': (64, 1024, 1)
  params = {'dim': (64, 1024),
            'batch_size': 64,
            'n_channels': 1,
            'shuffle': True}


  # Datasets
  labels = np.ones(np.shape(files)[0]) # Labels we don't actually use these 

  train_data, test_data, train_labels, test_labels = train_test_split(
      files, labels, test_size=0.2, random_state=21
  )

  print(np.shape(train_data))

  # Generators
  training_generator = DataGenerator(train_data, train_data, **params)
  validation_generator = DataGenerator(test_data, test_data, **params)

  # autoencoder = AnomalyDetector()
  autoencoder = create_model()
  autoencoder.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')
  print(autoencoder.summary())

  history = autoencoder.fit(training_generator, validation_data=validation_generator, epochs=100)
  # history = autoencoder.fit(training_generator, validation_data=validation_generator, epochs=50, use_multiprocessing=True, workers=6)

  autoencoder.save("pcdModel")


if __name__ == '__main__':
    main()
