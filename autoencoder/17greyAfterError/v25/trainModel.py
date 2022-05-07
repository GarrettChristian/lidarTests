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

# https://blog.keras.io/building-autoencoders-in-keras.html
# https://ai.stackexchange.com/questions/19891/how-to-add-a-dense-layer-after-a-2d-convolutional-layer-in-a-convolutional-autoe
# Kernel size: https://medium.com/analytics-vidhya/how-to-choose-the-size-of-the-convolution-filter-or-kernel-size-for-cnn-86a55a1e2d15#:~:text=Smaller%20kernel%20sizes%20consists%20of,to%20three%20weeks%20in%20training.
# CNN models to examine AlexNet, VGGNet, GoogLeNet, and ResNet
def create_model():
  model = Sequential()

  # Encoder
  model.add(layers.Input(shape=(64, 1024, 1)))
  model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

  model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

  model.add(layers.Flatten())
  model.add(layers.Dense(8192, activation="sigmoid"))
  
  # Decoder 
  model.add(layers.Dense(8192,activation='sigmoid'))
  model.add(layers.Reshape((2, 32, 128)))

  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.UpSampling2D(size=(2, 2)))

  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.UpSampling2D(size=(2, 2)))

  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.UpSampling2D(size=(2, 2)))

  model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.UpSampling2D(size=(2, 2)))

  model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
  model.add(layers.UpSampling2D(size=(2, 2)))  
  

  model.add(layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', padding='same'))

#  model.add(layers.Input(shape=(64, 1024, 1)))
#   model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

#   model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
  
#   model.add(layers.Flatten())
#   model.add(layers.Dense(32768, activation="sigmoid"))
  
#   # Decoder 
#   model.add(layers.Dense(32768,activation='sigmoid'))
#   model.add(layers.Reshape((2, 32, 512)))


#   model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.UpSampling2D(size=(2, 2)))

#   model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.UpSampling2D(size=(2, 2)))

#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.UpSampling2D(size=(2, 2)))

#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.UpSampling2D(size=(2, 2)))

#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#   model.add(layers.UpSampling2D(size=(2, 2)))

#   model.add(layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', padding='same'))


  return model

  


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
            'batch_size': 32,
            'n_channels': 1,
            'shuffle': True}


  # Datasets
  labels = np.ones(np.shape(files)[0]) # Labels we don't actually use these 

  train_data, test_data, train_labels, test_labels = train_test_split(
      files, labels, test_size=0.3, random_state=21
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
  # history = autoencoder.fit(training_generator, validation_data=validation_generator, epochs=100, use_multiprocessing=True, workers=6)

  autoencoder.save("pcdModel")


if __name__ == '__main__':
    main()
